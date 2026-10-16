"""ViT5 answer generation with question-guided vision and OCR features."""

from types import SimpleNamespace

import torch
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
from models.modules.ocr_image_fusion import JointAttentionLayer, OCRImageFusion
from models.modules.qa_clip import QACLIPEncoder


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
        self.vit5 = T5ForConditionalGeneration.from_pretrained(config.vit5_name)
        self.d_model = self.vit5.config.d_model
        for name in ("pad_token_id", "eos_token_id", "decoder_start_token_id"):
            value = getattr(config, name)
            if value is None:
                value = getattr(self.vit5.config, name)
            setattr(config, name, value)
            setattr(self.vit5.config, name, value)
        self.qa_clip = QACLIPEncoder.from_pretrained(
            config.clip_vision_name,
            instruction_dim=config.qa_clip_d_text or self.d_model,
            integration_point="late",
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
        heads = self.vit5.config.num_heads
        self.ocr_encoder = OCREncoder(
            heads,
            self.d_model,
            self.vit5.config.d_kv,
            self.vit5.config.d_ff,
        )
        self.ocr_encoder.set_word_embed_proxy(
            lambda ids: self.get_input_embeddings()(ids)
        )
        norm_eps = self.vit5.config.layer_norm_epsilon
        self.semantic_ocr_embedding = SemanticOCREmbedding(
            SimpleNamespace(
                d_model=self.d_model,
                d_det=config.ocr_d_det,
                d_rec=config.ocr_d_rec,
                layer_norm_epsilon=norm_eps,
            ),
        )
        self.image_ocr_fusion = OCRImageFusion(self.d_model, heads, norm_eps)
        self.char_embedding = nn.Embedding(config.char_num, self.d_model)
        self.char_position_embedding = nn.Embedding(config.char_max_num, self.d_model)
        self.ocr_char_layernorm = T5LayerNorm(self.d_model, eps=norm_eps)
        # The OCR ablation retains text/character/box input via a lightweight encoder.
        self.ocr_lite_text_ln = T5LayerNorm(self.d_model, eps=norm_eps)
        self.ocr_lite_text_proj = nn.Linear(self.d_model, self.d_model)
        self.ocr_lite_text_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.ocr_lite_box_proj = nn.Linear(4, self.d_model)
        self.ocr_lite_box_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        with torch.no_grad():
            self.ocr_lite_text_proj.weight.copy_(torch.eye(self.d_model))
            self.ocr_lite_text_proj.bias.zero_()
        self.generation_config = GenerationConfig(
            max_new_tokens=config.generation_max_new_tokens,
            num_beams=config.generation_num_beams,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            decoder_start_token_id=config.decoder_start_token_id,
            do_sample=False,
        )
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, JointAttentionLayer):
            module.reset_parameters()
        elif isinstance(module, nn.RMSNorm):
            module.reset_parameters()
        elif isinstance(module, SemanticOCREmbedding):
            # Explicit initialization for missing modules in baseline checkpoints.
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
                elif isinstance(layer, nn.RMSNorm):
                    layer.reset_parameters()
                elif isinstance(layer, T5LayerNorm):
                    layer.weight.data.fill_(1)

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

    def _encode_image(self, pixel_values, question_states, question_mask):
        batch_size = pixel_values.shape[0]
        if self.config.ablation_use_qaclip:
            instruction, instruction_mask = question_states, question_mask
        else:
            instruction, instruction_mask = question_states[:, :0], question_mask[:, :0]
        outputs = self.qa_clip(
            pixel_values=pixel_values,
            text_emb=instruction,
            text_mask=instruction_mask,
            output_attentions=False,
            return_dict=True,
        )
        tokens = outputs.last_hidden_state[:, 1:].to(self.target_dtype)
        mask = torch.ones(
            batch_size, tokens.shape[1], dtype=torch.long, device=tokens.device
        )
        return tokens, mask

    def _char_features(self, ids, mask):
        positions = torch.arange(ids.shape[-1], device=ids.device)
        features = self.char_embedding(ids) + self.char_position_embedding(positions)
        return self.ocr_char_layernorm((features * mask.unsqueeze(-1)).mean(dim=-2))

    def _encode_ocr(
        self,
        info,
        ids,
        token_mask,
        word_map,
        char_ids,
        char_mask,
        word_mask,
        image_tokens,
        image_mask,
    ):
        char_features = self._char_features(char_ids, char_mask)
        valid = (word_map >= 0) & token_mask.bool()
        indices = word_map.clamp(0, char_ids.shape[1] - 1)
        boxes = torch.stack([item["boxes"] for item in info]).to(
            ids.device, self.target_dtype
        )
        if not self.config.ablation_use_ocr:
            tokens = self.get_input_embeddings()(ids)
            features = tokens.new_zeros(*char_features.shape)
            counts = tokens.new_zeros(*char_features.shape[:2], 1)
            features.scatter_add_(
                1, indices.unsqueeze(-1).expand_as(tokens), tokens * valid.unsqueeze(-1)
            )
            counts.scatter_add_(
                1, indices.unsqueeze(-1), valid.unsqueeze(-1).to(tokens.dtype)
            )
            text = self.ocr_lite_text_ln(features / counts.clamp_min(1) + char_features)
            text = self.ocr_lite_text_proj(text)
            text = text + self.ocr_lite_text_ff(text)
            box_features = self.ocr_lite_box_proj(boxes)
            box_features = box_features + self.ocr_lite_box_ff(box_features)
            return (
                image_tokens,
                (text + box_features) * word_mask.unsqueeze(-1),
                word_mask,
            )
        visual = self.semantic_ocr_embedding.project_visual(info, word_mask)
        image_tokens, visual = self.image_ocr_fusion(
            image_tokens, visual, image_mask, word_mask
        )
        text_features, _ = self.ocr_encoder(ids, token_mask)
        features, _ = self.semantic_ocr_embedding(
            visual, text_features, char_features, word_map, token_mask
        )
        return image_tokens, features, token_mask

    def encode(
        self,
        input_ids,
        pixel_values,
        ocr_info,
        ocr_input_ids,
        ocr_to_word_map,
        ocr_char_ids,
        ocr_char_mask,
        ocr_word_mask,
        attention_mask=None,
        ocr_attention_mask=None,
    ):
        """Encode once; the same fused memory serves decoder loss and generation."""
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if ocr_attention_mask is None:
            ocr_attention_mask = (ocr_input_ids != self.config.pad_token_id).long()
        text_embeddings = self.get_input_embeddings()(input_ids)
        question = self.vit5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        image_tokens, image_mask = self._encode_image(
            pixel_values, question, attention_mask
        )
        image_tokens, ocr_features, ocr_mask = self._encode_ocr(
            ocr_info,
            ocr_input_ids,
            ocr_attention_mask,
            ocr_to_word_map,
            ocr_char_ids,
            ocr_char_mask,
            ocr_word_mask,
            image_tokens,
            image_mask,
        )
        summary = image_tokens.mean(dim=1, keepdim=True)
        sequences = [
            text_embeddings,
            image_tokens,
            ocr_features,
            summary,
        ]
        if any(sequence.shape[-1] != self.d_model for sequence in sequences):
            raise ValueError(
                "All vision/OCR backbone output dimensions must match ViT5 d_model."
            )
        fused_mask = torch.cat(
            [
                attention_mask,
                image_mask,
                ocr_mask,
                torch.ones(input_ids.shape[0], 1, device=device, dtype=torch.long),
            ],
            dim=1,
        )
        encoded = self.vit5.encoder(
            inputs_embeds=torch.cat(sequences, dim=1),
            attention_mask=fused_mask,
            return_dict=True,
        )
        return encoded, fused_mask

    def forward(
        self,
        input_ids,
        pixel_values,
        ocr_info,
        ocr_input_ids,
        ocr_to_word_map,
        ocr_char_ids,
        ocr_char_mask,
        ocr_word_mask,
        attention_mask=None,
        ocr_attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
    ):
        encoded, mask = self.encode(
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
        if encoder_outputs is None:
            encoder_outputs, attention_mask = self.encode(
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
