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
                    # Không nhận câu hỏi → CLIP thường; clip_late_adapter giữ adapter có
                    # cổng ở 6 lớp cuối như code trước refactor (câu hỏi rỗng).
                    integration_point="late" if (config.ablation_use_qaclip or config.clip_late_adapter) else "none",
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
                        norm_type=config.ocr_feature_norm,
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

        if config.use_twc and config.twc_objective in ("lexical", "both"):
            # Nhiệt độ học được τ của TWC (CLIP/TWA): exp(logit_scale) = 1/0.07 lúc đầu.
            self.twc_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        if config.use_twc and config.twc_objective in ("answer", "both"):
            # Answer-grounded TWC: câu hỏi (đã qua encoder) → truy vấn; khởi tạo đồng nhất.
            with _seeded(seed + 5):
                self.twc_answer_query = nn.Linear(self.d_model, self.d_model)
            with torch.no_grad():
                self.twc_answer_query.weight.copy_(torch.eye(self.d_model))
                self.twc_answer_query.bias.zero_()
            self.twc_answer_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

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
        elif self.config.clip_late_adapter:
            # Empty instruction: the gated adapter runs on the visual attention only.
            batch = pixel_values.shape[0]
            instruction = pixel_values.new_zeros(batch, 0, self.qa_clip.config.instruction_dim)
            instruction_mask = torch.zeros(batch, 0, dtype=torch.long, device=pixel_values.device)
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
            features = features * token_mask.unsqueeze(-1).to(features.dtype)
            return features, token_mask, features
        char_features = self._char_features(char_ids, char_mask)
        valid = (word_map >= 0) & token_mask.bool()
        indices = word_map.clamp(0, char_ids.shape[1] - 1)
        boxes = torch.stack([item["boxes"] for item in info]).to(ids.device, self.target_dtype)
        text_features, _ = self.ocr_encoder(ids, token_mask)
        results, lexical = [], []
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
            features, text_only = self.semantic_ocr_embedding(
                [token_info],
                text_features[batch_index : batch_index + 1],
                (char_features[batch_index][mapping] * keep).unsqueeze(0),
            )
            results.append(features.squeeze(0))
            lexical.append(text_only.squeeze(0))  # RMSNorm(ConsFormer) + RMSNorm(chars)
        return torch.stack(results), token_mask, torch.stack(lexical)

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
        return_lexical=False,
    ):
        """Encode [question, image patches, OCR] once for loss and generation.

        Returns (encoder outputs, fused mask, start index of the OCR block)
        [+ token-level lexical OCR features when return_lexical].
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
        ocr_features, ocr_mask, lexical = self._encode_ocr(
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
        if return_lexical:
            return encoded, fused_mask, ocr_offset, lexical
        return encoded, fused_mask, ocr_offset

    @staticmethod
    def _soft_infonce(scores, labels, rows):
        """Softmax cross-entropy per row against the soft multi-positive targets.

        rows: the real (non-PAD) query rows. Columns labelled -1 (padding, ignored text
        collisions) leave the softmax; positives (1.0 / 0.9, incl. same-text inheritance)
        share the target mass; a row weighs max(label), so augmented pairs (0.9) count
        slightly less than exact ones, as in TWA. Extra columns (memory) are negatives (0).
        """
        scores, labels = scores[rows], labels[rows]
        valid = labels != -1
        positive = labels.clamp(min=0) * valid
        weight = positive.max(dim=1).values
        target = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = torch.log_softmax(scores.masked_fill(~valid, float("-inf")), dim=1)
        loss = -(target * log_prob.masked_fill(~valid, 0.0)).sum(dim=1)
        return (weight * loss).sum() / weight.sum().clamp_min(1e-12)

    @staticmethod
    def _text_collision(query_text, key_text):
        """[N, M] True where the (original, related) texts of a query and a candidate share
        a string. query_text: [N, 2], key_text: [M, 2] hashed texts, -1 = unknown."""
        hit = torch.zeros(query_text.shape[0], key_text.shape[0], dtype=torch.bool, device=query_text.device)
        for a in range(2):
            for b in range(2):
                q, k = query_text[:, a].unsqueeze(1), key_text[:, b].unsqueeze(0)
                hit |= (q == k) & (q >= 0) & (k >= 0)
        return hit

    def _twc_memory(self, dim, device):
        """FIFO cross-batch memory: a plain attribute, so it is neither saved in checkpoints
        nor optimised; it restarts empty after a resume."""
        memory = getattr(self, "_twc_memory_state", None)
        size = self.config.twc_queue_size
        if memory is None or memory["original"].device != device or memory["original"].shape != (size, dim):
            memory = {
                "original": torch.zeros(size, dim, device=device),
                "related": torch.zeros(size, dim, device=device),
                "text": torch.full((size, 2), -1, dtype=torch.long, device=device),
                "pointer": 0, "filled": 0, "forwards": 0,
            }
            self._twc_memory_state = memory
        return memory

    @torch.no_grad()
    def _twc_enqueue(self, memory, original, related, text):
        size, count = memory["original"].shape[0], original.shape[0]
        if count == 0:
            return
        if count >= size:
            original, related, text, count = original[-size:], related[-size:], text[-size:], size
        index = (memory["pointer"] + torch.arange(count, device=original.device)) % size
        # Out-of-place: this step's loss still holds the old bank for its backward pass.
        memory["original"] = memory["original"].index_copy(0, index, original)
        memory["related"] = memory["related"].index_copy(0, index, related)
        memory["text"] = memory["text"].index_copy(0, index, text)
        memory["pointer"] = (memory["pointer"] + count) % size
        memory["filled"] = min(size, memory["filled"] + count)

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

    def _twc_answer(self, hidden, question_mask, ocr_tokens, word_map, token_mask,
                    labels, text_ids, answer_ids):
        """Answer-grounded TWC: the question (fused encoder states, mean over its tokens)
        must retrieve the OCR words of its answer among every OCR word slot of the batch
        (original + corrected copies). Positives = slots whose text is an answer word;
        other images' slots with an answer word's text are ignored (false negatives).
        labels: [B, S] 1/0/-1, text_ids: [B, S], answer_ids: [B, A] (-1 = pad)."""
        batch, slots = labels.shape
        words = self._pool_words(ocr_tokens, word_map, token_mask, slots).reshape(batch * slots, -1)
        mask = question_mask.unsqueeze(-1).float()
        question = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        query = F.normalize(self.twc_answer_query(question.to(self.twc_answer_query.weight.dtype)).float(),
                            dim=-1, eps=1e-6)
        scale = self.twc_answer_logit_scale.clamp(math.log(1 / 100), math.log(100)).exp()
        scores = scale * query @ words.t()  # [B, B·S]

        flat = labels.reshape(-1).float()
        owner = torch.arange(batch, device=labels.device).repeat_interleave(slots)
        own = owner.unsqueeze(0) == torch.arange(batch, device=labels.device).unsqueeze(1)
        targets = torch.where(flat.unsqueeze(0) == -1, torch.full_like(scores, -1.0),
                              torch.zeros_like(scores))
        targets = torch.where(own, flat.unsqueeze(0).expand_as(targets), targets)
        flat_text = text_ids.reshape(-1)
        same = ((flat_text.view(1, -1, 1) == answer_ids.unsqueeze(1))
                & (answer_ids.unsqueeze(1) >= 0) & (flat_text.view(1, -1, 1) >= 0)).any(-1)
        targets = targets.masked_fill(same & ~own, -1.0)
        rows = (targets == 1).any(dim=1)
        if not rows.any():
            return scores.sum() * 0.0, scores.new_zeros(())
        loss = self._soft_infonce(scores, targets, rows)
        with torch.no_grad():
            best = scores.detach().masked_fill(targets == -1, -1e9)[rows].argmax(dim=1)
            accuracy = (targets[rows].gather(1, best.unsqueeze(1)) == 1).float().mean()
        return loss, accuracy

    def _twc(self, tokens, word_map, token_mask, o2r, r2o, groups, text_ids=None):
        """TWC (TWA §3.4–3.6): word features of the original and related OCR copies,
        cosine similarity with learnable temperature; softmax InfoNCE (paper) or
        sigmoid BCE (TWA code) on the soft labels. tokens: [B, L_ocr, D].
        text_ids: [B, 2W] hashed word texts (original half, related half), -1 = PAD.
        With twc_queue_size > 0 the softmax also sees the features of earlier batches."""
        batch, words = o2r.shape[:2]
        features = self._pool_words(tokens, word_map, token_mask, 2 * words)
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
        pair_text = None
        if text_ids is not None:
            pair_text = torch.stack(
                [text_ids[:, :words].reshape(-1), text_ids[:, words:].reshape(-1)], dim=1
            ).long()
            if self.config.twc_mask_same_text:
                collision = self._text_collision(pair_text, pair_text)
                o2r = torch.where(collision & (o2r == 0), torch.full_like(o2r, -1.0), o2r)
                r2o = torch.where(collision.t() & (r2o == 0), torch.full_like(r2o, -1.0), r2o)
        for labels in (o2r, r2o):
            ignored = labels.diagonal() == -1
            labels[ignored, :] = -1.0
            labels[:, ignored] = -1.0
        real = o2r.diagonal() != -1
        if not real.any():
            return logits.sum() * 0.0, logits.new_zeros(())
        if self.config.twc_loss == "infonce":
            scores_o, labels_o = logits[real], o2r[real]
            scores_r, labels_r = logits.t()[real], r2o[real]
            use_memory = self.training and self.config.twc_queue_size > 0 and pair_text is not None
            if use_memory:
                memory = self._twc_memory(original.shape[-1], original.device)
                memory["forwards"] += 1
                filled = memory["filled"]
                if memory["forwards"] > self.config.twc_queue_start and filled > 0:
                    extra = torch.zeros(int(real.sum()), filled, device=original.device)
                    if self.config.twc_mask_same_text:
                        extra = extra.masked_fill(
                            self._text_collision(pair_text[real], memory["text"][:filled]), -1.0
                        )
                    scores_o = torch.cat([scores_o, scale * original[real] @ memory["related"][:filled].t()], dim=1)
                    scores_r = torch.cat([scores_r, scale * related[real] @ memory["original"][:filled].t()], dim=1)
                    labels_o = torch.cat([labels_o, extra], dim=1)
                    labels_r = torch.cat([labels_r, extra], dim=1)
            every = torch.ones(scores_o.shape[0], dtype=torch.bool, device=scores_o.device)
            loss = 0.5 * (
                self._soft_infonce(scores_o, labels_o, every) + self._soft_infonce(scores_r, labels_r, every)
            )
            if use_memory:
                self._twc_enqueue(memory, original[real].detach(), related[real].detach(), pair_text[real])
        else:
            keep = o2r != -1
            loss = 0.5 * (
                F.binary_cross_entropy_with_logits(logits[keep], o2r[keep])
                + F.binary_cross_entropy_with_logits(logits.t()[keep], r2o[keep])
            )
        with torch.no_grad():  # retrieval top-1 trong batch: từ gốc tìm đúng bản related của nó
            rows = real.nonzero(as_tuple=True)[0]
            best = logits.detach().masked_fill(o2r == -1, -1e9)[rows].argmax(dim=1)
            accuracy = (o2r[rows, best] > 0.5).float().mean()
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
        twc_text_ids=None,
        twc_answer_labels=None,
        twc_answer_text_ids=None,
        ocr_words=None,
    ):
        if ocr_attention_mask is None:
            ocr_attention_mask = (ocr_input_ids != self.config.pad_token_id).long()
        encoded, mask, ocr_offset, lexical = self.encode(
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
            return_lexical=True,
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
        if self.config.use_twc and twc_answer_labels is not None and hasattr(self, "twc_answer_query"):
            questions = input_ids.shape[1]
            result["twc_answer_loss"], result["twc_answer_acc"] = self._twc_answer(
                encoded.last_hidden_state[:, :questions],
                attention_mask if attention_mask is not None else (input_ids != self.config.pad_token_id).long(),
                encoded.last_hidden_state[:, ocr_offset : ocr_offset + ocr_input_ids.shape[1]],
                ocr_to_word_map, ocr_attention_mask, twc_answer_labels, twc_text_ids, twc_answer_text_ids,
            )
        if self.config.use_twc and twc_o2r is not None and hasattr(self, "twc_logit_scale"):
            if self.config.twc_feature == "lexical":
                tokens = lexical
            else:
                tokens = encoded.last_hidden_state[:, ocr_offset : ocr_offset + ocr_input_ids.shape[1]]
            result["twc_loss"], result["twc_acc"] = self._twc(
                tokens, ocr_to_word_map, ocr_attention_mask, twc_o2r, twc_r2o, twc_group_ids,
                twc_text_ids,
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
        for name in ("twc_o2r", "twc_r2o", "twc_group_ids", "twc_text_ids", "twc_answer_labels",
                     "twc_answer_text_ids", "ocr_words", "labels"):
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
