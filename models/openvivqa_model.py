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
                    # Không nhận câu hỏi → CLIP thường, không adapter QA-ViT.
                    integration_point="late" if config.ablation_use_qaclip else "none",
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

        if config.ablation_use_ocr:
            # SceSpaVis: ConsFormer + ký tự + box/det/rec.
            with _seeded(seed + 2):
                self.ocr_encoder = OCREncoder(
                    self.vit5.config.num_heads,
                    self.d_model,
                    self.vit5.config.d_kv,
                    self.vit5.config.d_ff,
                )
                self.semantic_ocr_embedding = SemanticOCREmbedding(
                    SimpleNamespace(
                        d_model=self.d_model,
                        d_det=config.ocr_d_det,
                        d_rec=config.ocr_d_rec,
                        layer_norm_epsilon=norm_eps,
                    ),
                )
                self.char_embedding = nn.Embedding(config.char_num, self.d_model)
                self.char_position_embedding = nn.Embedding(config.char_max_num, self.d_model)
                self.ocr_char_layernorm = T5LayerNorm(self.d_model, eps=norm_eps)
            self.ocr_encoder.set_word_embed_proxy(lambda ids: self.get_input_embeddings()(ids))
        else:
            # Baseline: OCR text-only ở mức token (embedding ViT5 → RMSNorm → Linear).
            with _seeded(seed + 3):
                self.ocr_lite_text_ln = T5LayerNorm(self.d_model, eps=norm_eps)
                self.ocr_lite_text_proj = nn.Linear(self.d_model, self.d_model)
            with torch.no_grad():
                self.ocr_lite_text_proj.weight.copy_(torch.eye(self.d_model))
                self.ocr_lite_text_proj.bias.zero_()

        if config.use_twc:
            # Nhiệt độ học được τ của TWC (CLIP/TWA): exp(logit_scale) = 1/0.07 lúc đầu.
            self.twc_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

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

    def _encode_image(self, pixel_values, input_ids, attention_mask):
        instruction = instruction_mask = None
        if self.config.ablation_use_qaclip:
            instruction = self.vit5.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
            instruction_mask = attention_mask
        outputs = self.qa_clip(
            pixel_values=pixel_values,
            text_emb=instruction,
            text_mask=instruction_mask,
            output_attentions=False,
            return_dict=True,
        )
        tokens = outputs.last_hidden_state[:, 1:].to(self.target_dtype)
        mask = torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device)
        return tokens, mask

    def _char_features(self, ids, mask):
        positions = torch.arange(ids.shape[-1], device=ids.device)
        features = self.char_embedding(ids) + self.char_position_embedding(positions)
        return self.ocr_char_layernorm((features * mask.unsqueeze(-1)).mean(dim=-2))

    def _encode_ocr(self, info, ids, token_mask, word_map, char_ids, char_mask):
        if not self.config.ablation_use_ocr:
            tokens = self.get_input_embeddings()(ids)
            features = self.ocr_lite_text_proj(self.ocr_lite_text_ln(tokens))
            return features * token_mask.unsqueeze(-1).to(features.dtype), token_mask
        char_features = self._char_features(char_ids, char_mask)
        valid = (word_map >= 0) & token_mask.bool()
        indices = word_map.clamp(0, char_ids.shape[1] - 1)
        boxes = torch.stack([item["boxes"] for item in info]).to(ids.device, self.target_dtype)
        text_features, _ = self.ocr_encoder(ids, token_mask)
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
        return torch.stack(results), token_mask

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
    ):
        """Encode [question, image patches, OCR] once for loss and generation.

        Returns (encoder outputs, fused mask, start index of the OCR block).
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if ocr_attention_mask is None:
            ocr_attention_mask = (ocr_input_ids != self.config.pad_token_id).long()
        pieces = [self.get_input_embeddings()(input_ids)]
        masks = [attention_mask]
        if self.config.ablation_use_image:
            image_tokens, image_mask = self._encode_image(pixel_values, input_ids, attention_mask)
            pieces.append(image_tokens)
            masks.append(image_mask)
        ocr_offset = sum(piece.shape[1] for piece in pieces)
        ocr_features, ocr_mask = self._encode_ocr(
            ocr_info, ocr_input_ids, ocr_attention_mask, ocr_to_word_map, ocr_char_ids, ocr_char_mask
        )
        pieces.append(ocr_features)
        masks.append(ocr_mask)
        if any(piece.shape[-1] != self.d_model for piece in pieces):
            raise ValueError("All vision/OCR backbone output dimensions must match ViT5 d_model.")
        fused_mask = torch.cat(masks, dim=1)
        encoded = self.vit5.encoder(
            inputs_embeds=torch.cat(pieces, dim=1), attention_mask=fused_mask, return_dict=True
        )
        return encoded, fused_mask, ocr_offset

    def _twc(self, hidden, ocr_offset, word_map, token_mask, o2r, r2o, groups):
        """TWC (TWA §3.4–3.6): word features of the original and related OCR copies,
        cosine similarity with learnable temperature, sigmoid BCE on soft labels."""
        batch, words = o2r.shape[:2]
        tokens = hidden[:, ocr_offset : ocr_offset + word_map.shape[1]].float()
        valid = ((word_map >= 0) & token_mask.bool()).unsqueeze(-1).float()
        index = word_map.clamp(min=0).unsqueeze(-1)
        sums = tokens.new_zeros(batch, 2 * words, tokens.shape[-1]).scatter_add_(
            1, index.expand_as(tokens), tokens * valid
        )
        counts = tokens.new_zeros(batch, 2 * words, 1).scatter_add_(1, index, valid)
        features = torch.nan_to_num(sums / counts.clamp_min(1), nan=0.0, posinf=0.0, neginf=0.0)
        features = F.normalize(features, dim=-1, eps=1e-6)
        original = features[:, :words].reshape(batch * words, -1)
        related = features[:, words:].reshape(batch * words, -1)
        scale = self.twc_logit_scale.clamp(math.log(1 / 100), math.log(100)).exp()
        logits = scale * original @ related.t()  # [B·W, B·W], mọi cặp trong batch

        # Nhãn: block-diagonal theo mẫu (khác ảnh = âm); cặp khác mẫu nhưng cùng
        # (ảnh, chữ) kế thừa nhãn đường chéo của đối tác; hàng/cột PAD = -1.
        o2r, r2o = torch.block_diag(*o2r.float()), torch.block_diag(*r2o.float())
        group = groups.reshape(-1)
        sample = torch.arange(group.numel(), device=group.device) // words
        inherit = (
            (group.unsqueeze(1) == group.unsqueeze(0))
            & (group.unsqueeze(1) >= 0)
            & (sample.unsqueeze(1) != sample.unsqueeze(0))
        )
        o2r = torch.where(inherit, o2r.diagonal().unsqueeze(0).expand_as(o2r), o2r)
        r2o = torch.where(inherit, r2o.diagonal().unsqueeze(0).expand_as(r2o), r2o)
        retrieval_labels = o2r.clone()
        for labels in (o2r, r2o):
            ignored = labels.diagonal() == -1
            labels[ignored, :] = -1.0
            labels[:, ignored] = -1.0
        keep = o2r != -1
        if not keep.any():
            return logits.sum() * 0.0, logits.new_zeros(())
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(logits[keep], o2r[keep])
            + F.binary_cross_entropy_with_logits(logits.t()[keep], r2o[keep])
        )
        with torch.no_grad():  # retrieval top-1: từ gốc tìm đúng bản related của nó
            real = retrieval_labels.diagonal() != -1
            rows = real.nonzero(as_tuple=True)[0]
            best = logits.detach().masked_fill(~real.unsqueeze(0), -1e9)[rows].argmax(dim=1)
            accuracy = (retrieval_labels[rows, best] > 0.5).float().mean()
        return loss, accuracy

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
        twc_o2r=None,
        twc_r2o=None,
        twc_group_ids=None,
        ocr_words=None,
    ):
        if ocr_attention_mask is None:
            ocr_attention_mask = (ocr_input_ids != self.config.pad_token_id).long()
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
        )
        result = {"encoder_outputs": encoded, "attention_mask": mask}
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
        if self.config.use_twc and twc_o2r is not None:
            result["twc_loss"], result["twc_acc"] = self._twc(
                encoded.last_hidden_state, ocr_offset, ocr_to_word_map, ocr_attention_mask,
                twc_o2r, twc_r2o, twc_group_ids,
            )
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
        for name in ("twc_o2r", "twc_r2o", "twc_group_ids", "ocr_words", "labels"):
            generation_kwargs.pop(name, None)
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
            )
        return self.vit5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            generation_config=generation_config or self.generation_config,
            use_cache=False,
            **generation_kwargs,
        )
