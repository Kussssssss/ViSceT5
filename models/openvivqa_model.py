"""ViT5 answer generation over question, CLIP image tokens and OCR features (+ TWC)."""

import math
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    CLIPImageProcessor,
    GenerationConfig,
    PreTrainedModel,
    T5ForConditionalGeneration,
)
from transformers.models.t5.modeling_t5 import T5LayerNorm

from configs.model_config import OpenViVQAConfig
from models.modules.ocr_consformer import OCREncoder
from models.modules.ocr_spatial import SemanticOCREmbedding
from models.modules.qa_clip import QACLIPEncoder

RELATED_FIELDS = ("ocr_rel_input_ids", "ocr_rel_attention_mask", "ocr_rel_to_word_map",
                  "ocr_rel_char_ids", "ocr_rel_char_mask")


@contextmanager
def _seeded(seed):
    """Initialise one module group from its own seed, independent of which other
    groups a configuration builds (ablations share the init of common modules)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed % (2**63))
        yield


class OpenViVQAModel(PreTrainedModel):
    config_class = OpenViVQAConfig
    base_model_prefix = "openvivqa"
    _tied_weights_keys = [
        "vit5.encoder.embed_tokens.weight",
        "vit5.decoder.embed_tokens.weight",
        "vit5.lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        seed = torch.initial_seed()
        self.vit5 = T5ForConditionalGeneration.from_pretrained(config.vit5_name)
        self.d_model = self.vit5.config.d_model
        for name in ("pad_token_id", "eos_token_id", "decoder_start_token_id"):
            value = getattr(config, name)
            if value is None:
                value = getattr(self.vit5.config, name)
            setattr(config, name, value)
            setattr(self.vit5.config, name, value)
        norm_eps = self.vit5.config.layer_norm_epsilon

        # Module không dùng thì không dựng: không tham số, không state_dict, không optimizer.
        image_size = config.image_size or 224
        if config.ablation_use_image:
            with _seeded(seed + 1):
                self.qa_clip = QACLIPEncoder.from_pretrained(
                    config.clip_vision_name,
                    instruction_dim=config.qa_clip_d_text or self.d_model,
                    # QA-ViT: câu hỏi vào 6 lớp cuối qua adapter học được; tắt = CLIP thường.
                    integration_point="late" if config.ablation_use_qaclip else "none",
                    gated_joint=config.qavit_gated_joint,
                    freeze_clip=True,
                    attn_implementation="eager",
                )
            image_size = config.image_size or self.qa_clip.config.image_size
        self.image_processor = CLIPImageProcessor(
            do_resize=config.do_resize,
            do_center_crop=config.do_center_crop,
            size={"width": image_size, "height": image_size},
            crop_size={"width": image_size, "height": image_size},
            image_mean=config.image_mean,
            image_std=config.image_std,
        )

        if config.ablation_use_ocr or config.ocr_full_features:
            # Đặc trưng OCR: chữ (ConsFormer, hoặc embedding token ViT5 khi tắt Consformer)
            # + ký tự + box/det/rec. Cùng thứ tự dựng như trước → bật Consformer khởi tạo y hệt.
            # With the ConsFormer gate, ConsFormer gets its own seed after the other OCR modules, so
            # these start exactly as in the run without Consformer (gate 0 = that run).
            gated = config.ablation_use_ocr and config.consformer_gate
            with _seeded(seed + 2):
                if config.ablation_use_ocr and not gated:
                    self.ocr_encoder = self._build_consformer(config)
                self.semantic_ocr_embedding = SemanticOCREmbedding(
                    SimpleNamespace(
                        d_model=self.d_model,
                        d_det=config.ocr_d_det,
                        d_rec=config.ocr_d_rec,
                        layer_norm_epsilon=norm_eps,
                        norm_type=config.ocr_feature_norm,
                    ),
                )
                self.char_embedding = nn.Embedding(config.char_num, self.d_model)
                self.char_position_embedding = nn.Embedding(config.char_max_num, self.d_model)
                self.ocr_char_layernorm = T5LayerNorm(self.d_model, eps=norm_eps)
            if gated:
                with _seeded(seed + 5):
                    self.ocr_encoder = self._build_consformer(config)
            if config.ablation_use_ocr:
                self.ocr_encoder.set_word_embed_proxy(lambda ids: self.get_input_embeddings()(ids))
        else:
            # Baseline: OCR text-only ở mức token (embedding ViT5 → RMSNorm → Linear).
            with _seeded(seed + 3):
                self.ocr_lite_text_ln = T5LayerNorm(self.d_model, eps=norm_eps)
                self.ocr_lite_text_proj = nn.Linear(self.d_model, self.d_model)
            with torch.no_grad():
                self.ocr_lite_text_proj.weight.copy_(torch.eye(self.d_model))
                self.ocr_lite_text_proj.bias.zero_()

        if config.use_twc and config.twc_contrastive:
            # Nhiệt độ học được τ của TWC (CLIP/TWA): exp(logit_scale) = 1/τ lúc đầu.
            self.twc_logit_scale = nn.Parameter(torch.tensor(math.log(1 / config.twc_temperature_init)))
            if config.twc_siglip:
                # SigLIP (Zhai et al., ICCV'23): logits = t·cos + b with a learnable bias, b0 = -t0, so a
                # pair at cos = 1 starts at logit 0 (t0 = 10, b0 = -10 is SigLIP's initialisation).
                self.twc_bias = nn.Parameter(torch.tensor(-1.0 / config.twc_temperature_init))
        if config.twc_text_only and config.use_twc and not hasattr(self, "semantic_ocr_embedding"):
            raise ValueError("twc_text_only needs the OCR text features (Consformer or ocr_full_features).")
        if config.twc_text_only and not config.twc_text_labels:
            # Equal strings have equal text features: as negatives they could never be separated.
            raise ValueError("twc_text_only needs twc_text_labels.")
        if config.qavit_ground and not (config.ablation_use_image and config.ablation_use_qaclip
                                        and config.qavit_gated_joint):
            raise ValueError("qavit_ground supervises the relevance logit of QA-ViT with qavit_gated_joint.")
        if (config.consformer_span or config.consformer_geometry or config.consformer_word_link) \
                and not config.ablation_use_ocr:
            raise ValueError("consformer_span / consformer_geometry / consformer_word_link need Consformer.")
        if config.consformer_span and config.consformer_constituent != "tree":
            raise ValueError("consformer_span supervises the constituent links of constituent='tree'.")
        if config.ocr_visual_pool:
            if not (config.ablation_use_image and hasattr(self, "semantic_ocr_embedding")):
                raise ValueError("ocr_visual_pool needs the image and the OCR features.")
            # OCR token += W LayerNorm(mean of the CLIP patch features under its box), W = 0 at the start
            # (ControlNet zero-convolution): the run starts as the run without it, dL/dW != 0.
            with _seeded(seed + 6):
                self.ocr_visual_norm = nn.LayerNorm(self.d_model, eps=norm_eps)
                self.ocr_visual_proj = nn.Linear(self.d_model, self.d_model)
            with torch.no_grad():
                self.ocr_visual_proj.weight.zero_()
                self.ocr_visual_proj.bias.zero_()

        self.generation_config = GenerationConfig(
            max_new_tokens=config.generation_max_new_tokens,
            num_beams=config.generation_num_beams,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            decoder_start_token_id=config.decoder_start_token_id,
            do_sample=False,
        )
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Weight loading re-creates parameters with requires_grad=True, which silently
        # unfroze CLIP (7 → 46 trainable tensors) and broke optimizer resume.
        loaded = super().from_pretrained(*args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if hasattr(model, "qa_clip"):
            model.qa_clip._apply_freeze()
        return loaded

    @property
    def target_dtype(self):
        return self.get_input_embeddings().weight.dtype

    def get_input_embeddings(self):
        return self.vit5.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        self.vit5.set_input_embeddings(embeddings)

    def get_output_embeddings(self):
        return self.vit5.get_output_embeddings()

    def tie_weights(self, *args, **kwargs):
        if hasattr(self, "vit5"):
            self.vit5.tie_weights()

    def get_encoder(self):
        return self.vit5.get_encoder()

    def get_decoder(self):
        return self.vit5.get_decoder()

    def sync_tokenizer_ids(self, tokenizer):
        for name in ("pad_token_id", "eos_token_id"):
            value = getattr(tokenizer, name)
            if value is None:
                raise ValueError(f"Tokenizer must define {name}.")
            for config in (self.config, self.vit5.config, self.generation_config):
                setattr(config, name, value)

    def _encode_image(self, pixel_values, input_ids, attention_mask, aux=None):
        """aux (dict, training with answer spans): receives the QA-ViT relevance logits
        [B, question layers, H, 1 + patches] for the grounding loss."""
        instruction = instruction_mask = None
        if self.config.ablation_use_qaclip:
            instruction = self.vit5.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
            instruction_mask = attention_mask
        relevance = aux is not None and self.config.qavit_ground
        outputs = self.qa_clip(
            pixel_values=pixel_values,
            text_emb=instruction,
            text_mask=instruction_mask,
            output_attentions=False,
            return_dict=True,
            output_relevance=relevance,
        )
        if relevance:
            outputs, relevances = outputs
            aux["relevance"] = torch.stack(relevances, dim=1)
        tokens = outputs.last_hidden_state[:, 1:].to(self.target_dtype)
        mask = torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device)
        return tokens, mask

    def _char_features(self, ids, mask):
        positions = torch.arange(ids.shape[-1], device=ids.device)
        features = self.char_embedding(ids) + self.char_position_embedding(positions)
        return self.ocr_char_layernorm((features * mask.unsqueeze(-1)).mean(dim=-2))

    @staticmethod
    def _patch_overlap(boxes, grid):
        """Fraction of each grid cell covered by each box: [B, W, grid * grid] (boxes xyxy in [0, 1];
        cells row-major, the order of CLIP's patch sequence). The image is resized to the square
        input without cropping (do_center_crop False), so normalised boxes map linearly to it."""
        edges = torch.linspace(0, 1, grid + 1, device=boxes.device, dtype=torch.float32)
        boxes = boxes.float()

        def cover(low, high):  # [B, W, grid]
            inside = torch.minimum(high[..., None], edges[1:]) - torch.maximum(low[..., None], edges[:-1])
            return inside.clamp_min(0) * grid

        x, y = cover(boxes[..., 0], boxes[..., 2]), cover(boxes[..., 1], boxes[..., 3])
        return (y[..., :, None] * x[..., None, :]).flatten(-2)

    def _link_inputs(self, info, word_map, token_mask):
        """Inputs of the ConsFormer links between adjacent tokens k, k+1: same_word [B, L-1] (both are
        pieces of one OCR word) and link_features [B, L-1, 4] = (horizontal gap, vertical offset of the
        centres, log height ratio, same word), lengths in units of the mean box height in pixels."""
        if not (self.config.consformer_geometry or self.config.consformer_word_link):
            return {}
        valid = (word_map >= 0) & token_mask.bool()
        pair = valid[:, 1:] & valid[:, :-1]
        same = pair & (word_map[:, 1:] == word_map[:, :-1])
        inputs = {}
        if self.config.consformer_word_link:
            inputs["same_word"] = same
        if self.config.consformer_geometry:
            device = word_map.device
            size = torch.tensor([[float(item["width"]), float(item["height"])] for item in info],
                                device=device, dtype=torch.float32).repeat(1, 2)
            boxes = torch.stack([item["boxes"] for item in info]).to(device, torch.float32) * size[:, None, :]
            index = word_map.clamp(0, boxes.shape[1] - 1).unsqueeze(-1).expand(-1, -1, 4)
            tokens = torch.gather(boxes, 1, index)
            left, right = tokens[:, :-1], tokens[:, 1:]
            height_left = (left[..., 3] - left[..., 1]).clamp_min(1.0)
            height_right = (right[..., 3] - right[..., 1]).clamp_min(1.0)
            height = 0.5 * (height_left + height_right)
            gap = (right[..., 0] - left[..., 2]) / height
            offset = 0.5 * (right[..., 1] + right[..., 3] - left[..., 1] - left[..., 3]) / height
            ratio = torch.log(height_right / height_left)
            features = torch.stack([gap, offset, ratio], dim=-1).clamp(-5.0, 5.0)
            features = torch.cat([features, same.unsqueeze(-1).float()], dim=-1)
            inputs["link_features"] = features * pair.unsqueeze(-1)
        return inputs

    def _encode_ocr(self, info, ids, token_mask, word_map, char_ids, char_mask, image_tokens=None, aux=None):
        """image_tokens: CLIP patch tokens [B, P, D] (ocr_visual_pool). aux (dict): receives the
        ConsFormer links of this pass (break_probs) for the answer-span loss."""
        if not (self.config.ablation_use_ocr or self.config.ocr_full_features):
            # Older text-only baseline (configs without ocr_full_features).
            tokens = self.get_input_embeddings()(ids)
            features = self.ocr_lite_text_proj(self.ocr_lite_text_ln(tokens))
            return features * token_mask.unsqueeze(-1).to(features.dtype)
        char_features = self._char_features(char_ids, char_mask)
        valid = (word_map >= 0) & token_mask.bool()
        indices = word_map.clamp(0, char_ids.shape[1] - 1)
        boxes = torch.stack([item["boxes"] for item in info]).to(ids.device, self.target_dtype)
        if self.config.ablation_use_ocr:
            text_features, break_probs = self.ocr_encoder(ids, token_mask, **self._link_inputs(info, word_map, token_mask))
            if aux is not None and self.config.consformer_span:
                aux["break_probs"] = break_probs
        else:  # Consformer off: the ViT5 token embeddings take the place of the ConsFormer output
            text_features = self.get_input_embeddings()(ids)
        visual = None
        if self.config.ocr_visual_pool:
            if image_tokens is None:
                raise ValueError("ocr_visual_pool needs the image tokens.")
            grid = math.isqrt(image_tokens.shape[1])
            if grid * grid != image_tokens.shape[1]:
                raise ValueError("ocr_visual_pool needs a square patch grid.")
            weights = self._patch_overlap(boxes, grid)  # [B, W, P]; an empty / PAD box has weight 0
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pooled = torch.matmul(weights, image_tokens.float())  # mean patch feature under each box
            visual = self.ocr_visual_proj(self.ocr_visual_norm(pooled.to(image_tokens.dtype)))
        results = []
        for batch_index, item in enumerate(info):
            mapping = indices[batch_index]
            keep = valid[batch_index].unsqueeze(-1)
            token_info = {
                "width": item["width"],
                "height": item["height"],
                "boxes": boxes[batch_index][mapping] * keep,
                "det": item["det_features"].to(ids.device, self.target_dtype)[mapping] * keep,
                "rec": item["rec_features"].to(ids.device, self.target_dtype)[mapping] * keep,
            }
            features, _ = self.semantic_ocr_embedding(
                [token_info],
                text_features[batch_index : batch_index + 1],
                (char_features[batch_index][mapping] * keep).unsqueeze(0),
            )
            results.append(features.squeeze(0))
        features = torch.stack(results)
        if visual is not None:
            index = indices.unsqueeze(-1).expand(-1, -1, visual.shape[-1])
            features = features + torch.gather(visual, 1, index) * valid.unsqueeze(-1).to(visual.dtype)
        return features

    def _build_consformer(self, config):
        return OCREncoder(
            self.vit5.config.num_heads,
            self.d_model,
            self.vit5.config.d_kv,
            self.vit5.config.d_ff,
            constituent=config.consformer_constituent,
            residual=config.consformer_residual,
            gate=config.consformer_gate,
            tree_transformer=config.consformer_tree_transformer,
            geometry=config.consformer_geometry,
        )

    def _ocr_text(self, ids, token_mask, word_map, char_ids, char_mask, info=None):
        """Text part of the OCR features [B, L, D]: LN(token or ConsFormer) + LN(characters), as in
        SemanticOCREmbedding but without box/det/rec (the TWC features of twc_text_only). The token
        embeddings enter as a constant, so TWC never moves the embedding table that the decoder shares
        as its output layer; it trains ConsFormer, the character stream and the two norms."""
        embeddings = self.get_input_embeddings()(ids).detach()
        if self.config.ablation_use_ocr:
            text, _ = self.ocr_encoder(ids, token_mask, embeddings=embeddings,
                                       **self._link_inputs(info, word_map, token_mask))
        else:
            text = embeddings
        chars = self._char_features(char_ids, char_mask)
        index = word_map.clamp(0, char_ids.shape[1] - 1).unsqueeze(-1).expand(-1, -1, chars.shape[-1])
        valid = ((word_map >= 0) & token_mask.bool()).unsqueeze(-1).to(chars.dtype)
        chars = torch.gather(chars, 1, index) * valid
        embedding = self.semantic_ocr_embedding
        return embedding.layer_norm_tok(text.to(self.target_dtype)) + embedding.layer_norm_char(chars.to(self.target_dtype))

    def encode(
        self,
        input_ids,
        pixel_values,
        ocr_info,
        ocr_input_ids,
        ocr_to_word_map,
        ocr_char_ids,
        ocr_char_mask,
        ocr_word_mask=None,
        attention_mask=None,
        ocr_attention_mask=None,
        image_features=None,
        ocr_rel_input_ids=None,
        ocr_rel_attention_mask=None,
        ocr_rel_to_word_map=None,
        ocr_rel_char_ids=None,
        ocr_rel_char_mask=None,
        aux=None,
    ):
        """Encode [question, image patches, original OCR, related OCR] once for loss and generation.

        As TWA: with TWC the related OCR copy follows the original one in the encoder input and
        shares its box/det/rec; each copy is encoded on its own (separate Consformer passes,
        shared weights), never as one concatenated sequence.
        image_features: precomputed (tokens, mask) to reuse CLIP.
        aux: dict filled with the auxiliary-loss inputs of the image and of the original OCR pass.

        Returns (encoder outputs, fused mask, start index of the OCR block).
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if ocr_attention_mask is None:
            ocr_attention_mask = (ocr_input_ids != self.config.pad_token_id).long()
        pieces = [self.get_input_embeddings()(input_ids)]
        masks = [attention_mask]
        image_tokens = None
        if self.config.ablation_use_image:
            image_tokens, image_mask = (image_features if image_features is not None
                                        else self._encode_image(pixel_values, input_ids, attention_mask, aux=aux))
            pieces.append(image_tokens)
            masks.append(image_mask)
        ocr_offset = sum(piece.shape[1] for piece in pieces)
        pieces.append(self._encode_ocr(ocr_info, ocr_input_ids, ocr_attention_mask, ocr_to_word_map,
                                       ocr_char_ids, ocr_char_mask, image_tokens=image_tokens, aux=aux))
        masks.append(ocr_attention_mask)
        if self.config.use_twc and not self.config.twc_two_views and (
                ocr_rel_input_ids is not None or self.training or self.config.twc_eval_related != "none"):
            if ocr_rel_input_ids is None:
                raise ValueError("TWC needs the related OCR copy (ocr_rel_* batch fields).")
            if ocr_rel_attention_mask is None:
                ocr_rel_attention_mask = (ocr_rel_input_ids != self.config.pad_token_id).long()
            pieces.append(self._encode_ocr(ocr_info, ocr_rel_input_ids, ocr_rel_attention_mask,
                                           ocr_rel_to_word_map, ocr_rel_char_ids, ocr_rel_char_mask,
                                           image_tokens=image_tokens))
            masks.append(ocr_rel_attention_mask)
        if any(piece.shape[-1] != self.d_model for piece in pieces):
            raise ValueError("All vision/OCR backbone output dimensions must match ViT5 d_model.")
        fused_mask = torch.cat(masks, dim=1)
        encoded = self.vit5.encoder(
            inputs_embeds=torch.cat(pieces, dim=1), attention_mask=fused_mask, return_dict=True
        )
        return encoded, fused_mask, ocr_offset

    @staticmethod
    def _pool_words(tokens, word_map, token_mask, slots):
        """Mean of the token features of each OCR word slot, L2-normalised: [B, slots, D]."""
        batch = tokens.shape[0]
        tokens = tokens.float()
        valid = ((word_map >= 0) & token_mask.bool()).unsqueeze(-1).float()
        index = word_map.clamp(min=0, max=slots - 1).unsqueeze(-1)
        sums = tokens.new_zeros(batch, slots, tokens.shape[-1]).scatter_add_(
            1, index.expand_as(tokens), tokens * valid
        )
        counts = tokens.new_zeros(batch, slots, 1).scatter_add_(1, index, valid)
        features = torch.nan_to_num(sums / counts.clamp_min(1), nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(features, dim=-1, eps=1e-6)

    def _ground_loss(self, relevance, ocr_info, answer_span):
        """Answer-region grounding of QA-ViT: -log of the share of the patch distribution
        pi = softmax_p(mean over layers and heads of logit lam_p) that falls on the patches touched by
        the boxes of the answer's OCR words (multiple-instance: any part of the region may take it).
        logit lam_p = logsumexp s_q - logsumexp s_v is the share of patch p's attention that the
        question takes, i.e. how much question signal the gated joint attention injects there; it
        does not pass through the gate, so this trains the question adapter while beta is still 0.
        Samples whose answer matches no OCR span are skipped. relevance [B, layers, H, 1 + P]."""
        logits = relevance[..., 1:].float().mean(dim=(1, 2))  # [B, P] (CLS dropped)
        grid = math.isqrt(logits.shape[-1])
        boxes = torch.stack([item["boxes"] for item in ocr_info]).to(logits.device)
        cover = self._patch_overlap(boxes, grid) * (answer_span >= 0).unsqueeze(-1).float()
        region = cover.sum(dim=1) > 0  # [B, P]
        rows = region.any(dim=-1)
        if not rows.any():
            return logits.sum() * 0.0, {}
        log_p = F.log_softmax(logits[rows], dim=-1)
        log_mass = torch.logsumexp(log_p.masked_fill(~region[rows], float("-inf")), dim=-1)
        statistics = {"ground_mass": log_mass.detach().exp().mean(),  # uniform map: = ground_area
                      "ground_area": region[rows].float().mean()}
        return -log_mass.mean(), statistics

    def _span_loss(self, break_probs, word_map, token_mask, answer_span):
        """Answer-span bracketing of ConsFormer (partial bracketing supervision, as unsupervised
        parsers trained with naturally occurring brackets): at the last layer the link a[k, k+1] of
        two tokens of the same answer span -> 1 and of a span edge -> 0; other links are free. The
        links only grow with depth (a_l = a_{l-1} + (1 - a_{l-1}) a_hat), so a span edge at 0 in the
        last layer is a constituent boundary in every layer and the span is one constituent at the
        top. break_probs [B, layers, h, L, L]; answer_span [B, W] span id of each word (-1 = none)."""
        link = break_probs[:, -1].float().diagonal(offset=1, dim1=-2, dim2=-1).mean(dim=1)  # [B, L-1]
        valid = (word_map >= 0) & token_mask.bool()
        span = torch.gather(answer_span, 1, word_map.clamp(0, answer_span.shape[1] - 1)).masked_fill(~valid, -1)
        left, right = span[:, :-1], span[:, 1:]
        pair = valid[:, :-1] & valid[:, 1:]
        if self.config.consformer_word_link:  # pieces of one word are linked to 1 already
            pair = pair & (word_map[:, :-1] != word_map[:, 1:])
        inside = pair & (left >= 0) & (left == right)
        edge = pair & (left != right) & ((left >= 0) | (right >= 0))
        keep = inside | edge
        if not keep.any():
            return link.sum() * 0.0, {}
        probability = link[keep].clamp(1e-6, 1 - 1e-6)
        target = inside[keep].float()
        # Written out: F.binary_cross_entropy is refused under autocast.
        loss = -(target * probability.log() + (1 - target) * (1 - probability).log()).mean()
        accuracy = ((probability.detach() > 0.5).float() == target).float().mean()
        return loss, {"span_acc": accuracy}

    @staticmethod
    def _twc_row_loss(logits, labels, keep):
        """SigLIP normalisation: BCE summed over the candidates of each anchor row, mean over the rows."""
        cells = F.binary_cross_entropy_with_logits(logits.float(), labels.clamp_min(0), reduction="none")
        rows = keep.any(dim=1)
        return (cells * keep).sum(dim=1)[rows].mean()

    def _twc(self, tokens, word_map, token_mask, o2r, r2o, orig_text=None, rel_text=None):
        """TWC as TWA (_forward_pretrain_cons_output + loss_accuracy): encoder output of every
        original word vs every related word of the batch, logits = exp(logit_scale) · cos,
        labels = create_batch_labels (block-diagonal, cross-sample 0, PAD rows/columns -1),
        symmetric sigmoid BCE over the cells != -1.
        tokens: [B, L, D] encoder output of [original OCR | related OCR]; word_map: [B, L] word
        slot of each token (original 0..W-1, related W..2W-1); o2r, r2o: [B, W, W] (r2o: rows =
        related words, columns = original words).

        twc_siglip: logits = t·cos + b and the loss is summed over the candidates of a row, averaged
        over the rows. Positives are ~1/(B·W) of the cells, so the mean over all cells without a bias
        is lowered mostly by pushing every pair down, and a lower temperature amplifies that.
        orig_text, rel_text: [B, W] batch-wide id of each word's text (-1 = PAD), with
        twc_text_labels — a pair with equal texts labelled 0 (the same string in another image, or an
          original word equal to another word's corrected copy) is a false negative: ignored;
        twc_text_only — only changed words (related text != original text) are anchors (rows): an
          unchanged word is the same input twice, so its positive is trivial. It stays a candidate.
        Returns (loss, top-1 accuracy of the anchors, mean positive / negative logit)."""
        batch, words = o2r.shape[:2]
        features = self._pool_words(tokens, word_map, token_mask, 2 * words)
        original = features[:, :words].reshape(batch * words, -1)
        related = features[:, words:].reshape(batch * words, -1)
        logits = self.twc_logit_scale.exp() * original @ related.t()  # [B·W, B·W]
        if self.config.twc_siglip:
            logits = logits + self.twc_bias
        o2r, r2o = torch.block_diag(*o2r.float()), torch.block_diag(*r2o.float())
        for labels in (o2r, r2o):
            ignored = labels.diagonal() == -1
            labels[ignored, :] = -1.0
            labels[:, ignored] = -1.0
        if orig_text is not None:
            orig_text, rel_text = orig_text.reshape(-1), rel_text.reshape(-1)
            if self.config.twc_text_labels:
                same = (orig_text[:, None] == rel_text[None, :]) & (orig_text[:, None] >= 0)
                o2r[same & (o2r == 0)] = -1.0
                r2o[same.t() & (r2o == 0)] = -1.0
            if self.config.twc_text_only:
                unchanged = orig_text == rel_text  # PAD (-1 == -1) is already ignored
                o2r[unchanged, :] = -1.0
                r2o[unchanged, :] = -1.0
        keep, keep_t = o2r != -1, r2o != -1
        if not (keep.any() and keep_t.any()):
            return logits.sum() * 0.0, logits.new_zeros(()), {}
        if self.config.twc_siglip:
            loss = 0.5 * (self._twc_row_loss(logits, o2r, keep) + self._twc_row_loss(logits.t(), r2o, keep_t))
        else:
            loss = 0.5 * (
                F.binary_cross_entropy_with_logits(logits[keep], o2r[keep])
                + F.binary_cross_entropy_with_logits(logits.t()[keep_t], r2o[keep_t])
            )
        with torch.no_grad():  # retrieval top-1 trong batch: từ gốc tìm đúng bản related của nó
            detached = logits.detach()
            rows = (o2r.diagonal() != -1).nonzero(as_tuple=True)[0]
            best = detached.masked_fill(~keep, -1e9)[rows].argmax(dim=1)
            accuracy = (o2r[rows, best] > 0.5).float().mean() if rows.numel() else logits.new_zeros(())
            statistics = {}
            positive, negative = keep & (o2r > 0.5), keep & (o2r == 0)
            if positive.any():
                statistics["twc_pos_logit"] = detached[positive].mean()
            if negative.any():
                statistics["twc_neg_logit"] = detached[negative].mean()
        return loss, accuracy, statistics

    def forward(
        self,
        input_ids,
        pixel_values,
        ocr_info,
        ocr_input_ids,
        ocr_to_word_map,
        ocr_char_ids,
        ocr_char_mask,
        ocr_word_mask=None,
        attention_mask=None,
        ocr_attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        ocr_rel_input_ids=None,
        ocr_rel_attention_mask=None,
        ocr_rel_to_word_map=None,
        ocr_rel_char_ids=None,
        ocr_rel_char_mask=None,
        twc_o2r=None,
        twc_r2o=None,
        ocr_words=None,
        twc_orig_text=None,
        twc_rel_text=None,
        answer_span=None,
    ):
        """answer_span [B, W] (training collator with qavit_ground / consformer_span): span id of the
        answer's OCR words, -1 elsewhere; adds ground_loss / span_loss to the outputs."""
        if ocr_attention_mask is None:
            ocr_attention_mask = (ocr_input_ids != self.config.pad_token_id).long()
        if ocr_rel_input_ids is not None and ocr_rel_attention_mask is None:
            ocr_rel_attention_mask = (ocr_rel_input_ids != self.config.pad_token_id).long()
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        two_views = self.config.use_twc and self.config.twc_two_views and ocr_rel_input_ids is not None
        aux = {} if answer_span is not None else None
        image_features = None
        if two_views and self.config.ablation_use_image:
            image_features = self._encode_image(pixel_values, input_ids, attention_mask, aux=aux)
        encoded, mask, ocr_offset = self.encode(
            input_ids,
            pixel_values,
            ocr_info,
            ocr_input_ids,
            ocr_to_word_map,
            ocr_char_ids,
            ocr_char_mask,
            ocr_word_mask,
            attention_mask,
            ocr_attention_mask,
            ocr_rel_input_ids=ocr_rel_input_ids,
            ocr_rel_attention_mask=ocr_rel_attention_mask,
            ocr_rel_to_word_map=ocr_rel_to_word_map,
            ocr_rel_char_ids=ocr_rel_char_ids,
            ocr_rel_char_mask=ocr_rel_char_mask,
            image_features=image_features,
            aux=aux,
        )
        result = {"encoder_outputs": encoded, "attention_mask": mask}
        if aux and "relevance" in aux:
            loss, statistics = self._ground_loss(aux["relevance"], ocr_info, answer_span.to(mask.device))
            result.update(ground_loss=loss, **statistics)
        if aux and "break_probs" in aux:
            loss, statistics = self._span_loss(aux["break_probs"], ocr_to_word_map, ocr_attention_mask,
                                               answer_span.to(mask.device))
            result.update(span_loss=loss, **statistics)
        if labels is not None or decoder_input_ids is not None:
            decoded = self.vit5(
                encoder_outputs=encoded,
                attention_mask=mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            result.update(loss=decoded.loss, logits=decoded.logits)
        if two_views:
            # Second view: the related copy takes the place of the original one, same answer.
            view, view_mask, _ = self.encode(
                input_ids, pixel_values, ocr_info, ocr_rel_input_ids, ocr_rel_to_word_map, ocr_rel_char_ids,
                ocr_rel_char_mask, None, attention_mask, ocr_rel_attention_mask, image_features=image_features,
            )
            if labels is not None:
                decoded = self.vit5(encoder_outputs=view, attention_mask=view_mask, labels=labels,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask, use_cache=False, return_dict=True)
                result.update(view_loss=decoded.loss, view_logits=decoded.logits)
        if self.config.use_twc and self.config.twc_contrastive and twc_o2r is not None:
            # TWA: split the encoder output of the OCR block into the original and related copies.
            words = ocr_char_ids.shape[1]
            length = ocr_input_ids.shape[1] + ocr_rel_input_ids.shape[1]
            word_map = torch.cat([ocr_to_word_map, torch.where(
                ocr_rel_to_word_map >= 0, ocr_rel_to_word_map + words, ocr_rel_to_word_map)], dim=1)
            if self.config.twc_text_only:
                # Text features of each copy (token or ConsFormer + characters) before box/det/rec: the
                # related copy shares those with its original word, so on the encoder output the pair
                # is found by position and appearance alone, without reading the text.
                tokens = torch.cat([
                    self._ocr_text(ocr_input_ids, ocr_attention_mask, ocr_to_word_map, ocr_char_ids, ocr_char_mask,
                                   ocr_info),
                    self._ocr_text(ocr_rel_input_ids, ocr_rel_attention_mask, ocr_rel_to_word_map,
                                   ocr_rel_char_ids, ocr_rel_char_mask, ocr_info),
                ], dim=1)
            elif two_views:
                tokens = torch.cat([
                    encoded.last_hidden_state[:, ocr_offset : ocr_offset + ocr_input_ids.shape[1]],
                    view.last_hidden_state[:, ocr_offset : ocr_offset + ocr_rel_input_ids.shape[1]],
                ], dim=1)
            else:
                tokens = encoded.last_hidden_state[:, ocr_offset : ocr_offset + length]
            texts = {} if twc_orig_text is None else {"orig_text": twc_orig_text, "rel_text": twc_rel_text}
            loss, accuracy, statistics = self._twc(
                tokens,
                word_map,
                torch.cat([ocr_attention_mask, ocr_rel_attention_mask], dim=1),
                twc_o2r,
                twc_r2o,
                **texts,
            )
            result.update(twc_loss=loss, twc_acc=accuracy, **statistics)
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        ocr_info=None,
        ocr_input_ids=None,
        ocr_to_word_map=None,
        ocr_char_ids=None,
        ocr_char_mask=None,
        ocr_word_mask=None,
        attention_mask=None,
        ocr_attention_mask=None,
        encoder_outputs=None,
        generation_config=None,
        **generation_kwargs,
    ):
        for name in ("twc_o2r", "twc_r2o", "twc_orig_text", "twc_rel_text", "ocr_words", "labels", "answer_span"):
            generation_kwargs.pop(name, None)
        related = {name: generation_kwargs.pop(name) for name in RELATED_FIELDS if name in generation_kwargs}
        if encoder_outputs is None:
            encoder_outputs, attention_mask, _ = self.encode(
                input_ids,
                pixel_values,
                ocr_info,
                ocr_input_ids,
                ocr_to_word_map,
                ocr_char_ids,
                ocr_char_mask,
                ocr_word_mask,
                attention_mask,
                ocr_attention_mask,
                **related,
            )
        return self.vit5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            generation_config=generation_config or self.generation_config,
            use_cache=False,
            **generation_kwargs,
        )
