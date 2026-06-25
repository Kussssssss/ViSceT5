"""
models/openvivqa_model.py
Main OpenViVQA model — PreTrainedModel wrapping ViT5 + QACLIPEncoder +
OCR Consformer + VisualSearch.

Bug fixed vs notebook: _encode_ocr_features was accidentally dedented to module
scope in the notebook source; it is restored here as a proper class method.
"""

from configs.model_config import OpenViVQAConfig
from models.modules.qa_clip import QACLIPEncoder
from models.modules.ocr_consformer import OCREncoder
from models.modules.ocr_spatial import SemanticOCREmbedding
from models.modules.visual_search import VisualSearch

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from transformers import (
    PreTrainedModel,
    T5ForConditionalGeneration,
    CLIPImageProcessor,
    GenerationConfig,
)
from transformers.models.t5.modeling_t5 import T5LayerNorm

def _any_device_fallback(**kwargs):
    for _, v in kwargs.items():
        if torch.is_tensor(v): return v.device
        if isinstance(v, list) and len(v) and torch.is_tensor(v[0]): return v[0].device
    return torch.device("cpu")

def _char_embedding(char_embedding, char_position_embedding, _ocr_char, ocr_char_mask, mean=True):
    ocr_char_emb = char_embedding(_ocr_char)
    pos_idx = torch.arange(ocr_char_emb.size(-2), device=ocr_char_emb.device)
    pos_emb = char_position_embedding(pos_idx)

    _dim = ocr_char_emb.dim() - pos_emb.dim()
    if _dim == 2: pos_emb = pos_emb.unsqueeze(0).unsqueeze(0)
    elif _dim == 1: pos_emb = pos_emb.unsqueeze(0)
    else: raise RuntimeError("dim mismatch in _char_embedding")

    ocr_char_emb = ocr_char_emb + pos_emb
    mask = ocr_char_mask.to(dtype=ocr_char_emb.dtype).unsqueeze(-1)
    ocr_char_emb = ocr_char_emb * mask
    if mean: 
        denom = mask.sum(dim=-2).clamp_min(1.0)
        ocr_char_emb = ocr_char_emb.sum(dim=-2) / denom
    return ocr_char_emb

def _pad_or_crop_lastdim(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
    cur = x.size(1)
    if cur == target_len: return x
    if cur > target_len: return x[:, :target_len]
    pad_shape = list(x.shape)
    pad_shape[1] = target_len - cur
    pad = torch.full(pad_shape, pad_value, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

def _pad_or_crop_lastdim_int(x: torch.Tensor, target_len: int, pad_value: int = -1) -> torch.Tensor:
    cur = x.size(1)
    if cur == target_len: return x
    if cur > target_len: return x[:, :target_len]
    pad = torch.full((x.size(0), target_len - cur), pad_value, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

def _ensure_1_token(x: torch.Tensor, B: int, D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x is None: return torch.zeros(B, 1, D, device=device, dtype=dtype)
    if x.ndim == 2: x = x.unsqueeze(1)
    if x.size(1) == 0: return torch.zeros(B, 1, D, device=device, dtype=dtype)
    if x.size(1) == 1: return x.to(device=device, dtype=dtype)
    return x.mean(dim=1, keepdim=True).to(device=device, dtype=dtype)

def _normalize_boxes_auto(boxes: torch.Tensor, width: float, height: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    boxes = boxes.to(device=device, dtype=dtype)
    if boxes.numel() == 0: return boxes
    mx = float(boxes.detach().abs().max().item())
    w = max(float(width) if width else 1.0, 1.0)
    h = max(float(height) if height else 1.0, 1.0)
    if mx <= 1.5: return boxes.clamp(0.0, 1.0)
    norm = torch.tensor([w, h, w, h], device=device, dtype=dtype).clamp_min(1.0)
    out = boxes / norm
    return out.clamp(0.0, 1.0)

class T5PolluteHead(nn.Module):
    def __init__(self, input_size, hidden_size=768, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = T5LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, 1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dense(x)
        h = self.gelu(h)
        h = self.LayerNorm(h)
        return self.decoder(h).squeeze(-1)

# =====================================================================
# MÔ HÌNH CHÍNH: OPENVIVQA MODEL
# =====================================================================
class OpenViVQAModel(PreTrainedModel):
    config_class = OpenViVQAConfig
    base_model_prefix = "openvivqa"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit5 = T5ForConditionalGeneration.from_pretrained(config.vit5_name)
        self.d_model = int(self.vit5.config.d_model)

        pad_id = self.vit5.config.pad_token_id or 0
        eos_id = self.vit5.config.eos_token_id or 1
        dec_start_id = config.decoder_start_token_id if config.decoder_start_token_id is not None else pad_id

        for obj in (self.vit5.config, self.config):
            obj.pad_token_id = pad_id
            obj.eos_token_id = eos_id
            obj.decoder_start_token_id = dec_start_id

        self.target_dtype = self.vit5.get_input_embeddings().weight.dtype

        # Khởi tạo QACLIP
        d_text = getattr(config, "qa_clip_d_text", None) or self.d_model
        self.qa_clip = QACLIPEncoder.from_pretrained(
            getattr(config, "clip_vision_name", "openai/clip-vit-base-patch16"),
            instruction_dim=d_text,
            integration_point="late",
            freeze_clip=True,
        )

        # Khởi tạo Bộ tiền xử lý ảnh
        img_sz = int(getattr(config, "vs_target_size", getattr(self.qa_clip.config, "image_size", 224)))
        self.image_processor = CLIPImageProcessor(
            do_resize=bool(getattr(config, "do_resize", True)),
            do_center_crop=bool(getattr(config, "do_center_crop", False)),
            size={"width": img_sz, "height": img_sz},
            crop_size={"width": img_sz, "height": img_sz},
            image_mean=list(getattr(config, "image_mean", [0.48145466, 0.4578275, 0.40821073])),
            image_std=list(getattr(config, "image_std", [0.26862954, 0.26130258, 0.27577711])),
        )
        self.clip_hidden = int(getattr(self.qa_clip.config, "hidden_size", 768))

        # Khởi tạo Visual Search (Kính lúp)
        self.visual_search = VisualSearch(
            vit_processor=self.image_processor,
            model_config=self.config,
            vit_dim=self.d_model,
            device=getattr(self.config, "ocr_cuda_device", "cuda:0"),
            vs_local_dir=None,
            local_files_only=False,
        )
        self.visual_search.vit_processor = self.image_processor

        # Khởi tạo OCR Consformer (Bản Full)
        self.seq_max_ocr = int(getattr(config, "ocr_max_scene_text", 180))
        ns = type("NS", (object,), dict(
            d_model=self.d_model,
            cuda_device=getattr(config, "ocr_cuda_device", "cuda:0"),
            d_det=int(getattr(config, "ocr_d_det", 256)),
            d_rec=int(getattr(config, "ocr_d_rec", 256)),
            max_scene_text=int(getattr(config, "ocr_max_scene_text", 180)),
            num_attention_heads=int(getattr(self.vit5.config, "num_attention_heads", 8)),
            num_distances=int(getattr(config, "ocr_num_distances", 32)),
            max_2d_position_embeddings=int(getattr(config, "ocr_max_2d_position_embeddings", 1024)),
        ))()

        self.ocr_encoder = OCREncoder(
            int(getattr(self.vit5.config, "num_attention_heads", 8)),
            self.d_model,
            int(getattr(self.vit5.config, "d_kv", self.d_model // int(getattr(self.vit5.config, "num_attention_heads", 8)))),
            int(getattr(self.vit5.config, "d_ff", 4 * self.d_model)),
            word_embed=None,
        )
        self.ocr_encoder.set_word_embed_proxy(lambda ids: self.vit5.get_input_embeddings()(ids))
        self.semantic_ocr_embedding = SemanticOCREmbedding(ns)

        self.char_max_num = int(getattr(config, "char_max_num", 50))
        self.char_num = int(getattr(config, "char_num"))
        self.char_position_embedding = nn.Embedding(self.char_max_num, self.d_model)
        self.char_embedding = nn.Embedding(self.char_num, self.d_model)
        self.ocr_char_layernorm = T5LayerNorm(self.d_model, eps=1e-12)

        # Mạng Baseline cho OCR (Bản Lite - Dùng khi tắt OCR Module)
        self.ocr_lite_text_ln = T5LayerNorm(self.d_model, eps=1e-12)
        self.ocr_lite_text_proj = nn.Linear(self.d_model, self.d_model)
        self.ocr_lite_text_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model),
        )
        self.ocr_lite_box_proj = nn.Linear(4, self.d_model)
        self.ocr_lite_box_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model),
        )
        with torch.no_grad():
            self.ocr_lite_text_proj.weight.copy_(torch.eye(self.d_model))
            self.ocr_lite_text_proj.bias.zero_()

        self.pretrain = bool(getattr(self.config, "pretrain", True))
        self.pretrain_ablation_mode = str(
            getattr(self.config, "pretrain_ablation_mode", "full")
        ).lower().strip()
        self.use_twc = bool(getattr(self.config, "use_twc", True))

        self.pollute_head = T5PolluteHead(input_size=self.d_model, layer_norm_eps=1e-12).to(torch.float32)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.generation_config = GenerationConfig(
            max_new_tokens=int(getattr(config, "generation_max_new_tokens", 27)),
            num_beams=int(getattr(config, "generation_num_beams", 4)),
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            decoder_start_token_id=self.config.decoder_start_token_id,
            do_sample=False,
        )

        if hasattr(self.vit5, "tie_weights"): self.vit5.tie_weights()
        if hasattr(self.qa_clip, "init_qavit_comps"): self.qa_clip.init_qavit_comps()
        self.post_init()

    # --- Các hàm đồng bộ cơ bản ---
    def sync_tokenizer_ids(self, tokenizer, persist_dir: Optional[str] = None):
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = self.config.pad_token_id
        if tokenizer.eos_token_id is None: tokenizer.eos_token_id = self.config.eos_token_id
        if getattr(tokenizer, "decoder_start_token_id", None) is None:
            setattr(tokenizer, "decoder_start_token_id", self.config.decoder_start_token_id)
        for obj in (self.vit5.config, self.config, self.generation_config):
            obj.pad_token_id = tokenizer.pad_token_id
            obj.eos_token_id = tokenizer.eos_token_id
            obj.decoder_start_token_id = tokenizer.decoder_start_token_id
        if persist_dir:
            try: tokenizer.save_pretrained(persist_dir)
            except Exception: pass

    def get_input_embeddings(self): return self.vit5.get_input_embeddings()
    def set_input_embeddings(self, v): return self.vit5.set_input_embeddings(v)
    def get_output_embeddings(self): return self.vit5.get_output_embeddings()
    def tie_weights(self): 
        if hasattr(self.vit5, "tie_weights"): self.vit5.tie_weights()
    def get_encoder(self): return self.vit5.get_encoder()
    def get_decoder(self): return self.vit5.get_decoder()

    # --- ENCODE TEXT ---
    def _encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], device: torch.device):
        emb = self.vit5.get_input_embeddings()(input_ids).to(dtype=self.target_dtype)
        B, T, _ = emb.size()
        if attention_mask is not None:
            attn = attention_mask.long().to(device)
            if attn.size(1) != T:
                if attn.size(1) > T: attn = attn[:, :T]
                else:
                    pad = torch.zeros(B, T - attn.size(1), dtype=torch.long, device=device)
                    attn = torch.cat([attn, pad], dim=1)
        else:
            attn = torch.ones(B, T, dtype=torch.long, device=device)
        return emb, attn

    # --- ENCODE IMAGE & TÍNH ATTENTION ---
    def _encode_image(
        self, pixel_values: torch.Tensor, device: torch.device, 
        txt_emb: Optional[torch.Tensor] = None, txt_mask: Optional[torch.Tensor] = None, 
        fuse_with_text: bool = True, return_attn: bool = False
    ):
        B = pixel_values.size(0)

        if fuse_with_text and txt_emb is not None:
            text_emb = txt_emb.to(device=device, dtype=self.target_dtype)
            T = text_emb.size(1)
            if txt_mask is not None:
                if txt_mask.size(1) > T: text_mask = txt_mask[:, :T].long().to(device)
                elif txt_mask.size(1) < T:
                    pad = torch.zeros(B, T - txt_mask.size(1), dtype=torch.long, device=device)
                    text_mask = torch.cat([txt_mask.long().to(device), pad], dim=1)
                else: text_mask = txt_mask.long().to(device)
            else: text_mask = torch.ones(B, T, dtype=torch.long, device=device)

            qa_out = self.qa_clip(pixel_values=pixel_values, text_emb=text_emb, text_mask=text_mask, output_attentions=True, return_dict=True)
        else:
            # Baseline (Tắt QA): Không có câu hỏi định hướng
            D_txt = int(getattr(self.qa_clip.config, "instruction_dim", self.d_model))
            text_emb = torch.zeros(B, 0, D_txt, device=device, dtype=self.target_dtype)
            text_mask = torch.zeros(B, 0, dtype=torch.long, device=device)
            qa_out = self.qa_clip(pixel_values=pixel_values, text_emb=text_emb, text_mask=text_mask, output_attentions=True, return_dict=True)
            T = 0

        img_hs = qa_out.last_hidden_state
        img_tokens = img_hs[:, 1:, :].to(self.target_dtype)
        img_attn_mask = torch.ones(B, img_tokens.size(1), dtype=torch.long, device=device)
        out = {"img_tokens": img_tokens, "img_attn_mask": img_attn_mask}

        # Tính bản đồ nhiệt (Heatmap) cho Visual Search
        if qa_out.attentions is not None:
            last_attn = qa_out.attentions[-1]
            if T > 0:
                # Question-Guided Attention (Lấy tương tác giữa Image Patches và Text)
                patch_to_text = last_attn[:, :, 1:, :T]
                patch_scores = patch_to_text.mean(dim=[1, 3])
            else:
                # Visual Saliency Attention (Lấy tương tác thuần túy của token [CLS] với Image Patches)
                cls_to_patch = last_attn[:, :, 0, 1:]
                patch_scores = cls_to_patch.mean(dim=1)

            out["patch_scores"] = patch_scores.to(self.target_dtype)
            if return_attn: out["qa_attn_last"] = last_attn.detach().cpu()
        else:
            out["patch_scores"] = torch.ones(B, img_tokens.size(1), device=device, dtype=self.target_dtype)

        if return_attn: out["image_features_full"] = img_hs
        return out

    def _get_ocr_word_mask(
        self,
        info_i: Dict[str, Any],
        box_mask_i: Optional[torch.Tensor],
        n_word: int,
        device: torch.device,
    ) -> torch.Tensor:
        if box_mask_i is not None:
            word_mask = box_mask_i
        else:
            word_mask = info_i.get("word_mask_all", None)
            if word_mask is None:
                word_mask = torch.ones(n_word, device=device, dtype=torch.long)
    
        if not torch.is_tensor(word_mask):
            word_mask = torch.tensor(word_mask, device=device, dtype=torch.long)
        else:
            word_mask = word_mask.to(device=device, dtype=torch.long)
    
        if word_mask.size(0) != n_word:
            if word_mask.size(0) > n_word:
                word_mask = word_mask[:n_word]
            else:
                pad = torch.zeros(
                    n_word - word_mask.size(0),
                    device=device,
                    dtype=torch.long,
                )
                word_mask = torch.cat([word_mask, pad], dim=0)
    
        return word_mask
    
    # --- ENCODE OCR (BẢN FULL CONSFORMER) ---
    def _encode_ocr_features(
        self,
        ocr_info: List[Dict[str, Any]],
        ocr_token_ids: torch.Tensor,
        ocr_to_word_map: torch.Tensor,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor,
        token_mask: torch.Tensor,
        box_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        ocr_token_ids = ocr_token_ids.to(device)
        token_mask = token_mask.to(device)
        ocr_to_word_map = ocr_to_word_map.to(device)

        L_tok = ocr_token_ids.size(1)
        if ocr_to_word_map.size(1) != L_tok:
            ocr_to_word_map = _pad_or_crop_lastdim_int(ocr_to_word_map, L_tok, pad_value=-1)

        ocr_text_tok, _ = self.ocr_encoder(ocr_token_ids, token_mask)
        ocr_text_tok = ocr_text_tok.to(self.target_dtype)

        B, L_tok2, D = ocr_text_tok.size()
        if L_tok2 != L_tok:
            L_tok = L_tok2
            ocr_to_word_map = _pad_or_crop_lastdim_int(ocr_to_word_map, L_tok, pad_value=-1)
            token_mask = _pad_or_crop_lastdim_int(token_mask.long(), L_tok, pad_value=0).to(token_mask.dtype)

        N_word = int(char_ids.size(1))

        char_feat_word = _char_embedding(
            self.char_embedding,
            self.char_position_embedding,
            char_ids.to(device),
            char_mask.to(device),
            mean=True,
        )
        char_feat_word = self.ocr_char_layernorm(char_feat_word).to(self.target_dtype)

        ocr_box_feat_tok_list = []

        for i in range(B):
            map_i = ocr_to_word_map[i]
            tok_mask_i = token_mask[i]

            valid = (map_i >= 0) & (tok_mask_i > 0)
            map_clamped = map_i.clamp(min=0, max=max(N_word - 1, 0))

            char_feat_tok_i = char_feat_word[i][map_clamped] * valid.unsqueeze(-1)

            info_i = ocr_info[i]
            # --- XỬ LÝ BOXES ---
            boxes_word_all = info_i.get("boxes_word_all")
            if not torch.is_tensor(boxes_word_all):
                boxes_word_all = torch.tensor(boxes_word_all, device=device, dtype=self.target_dtype)
            else:
                boxes_word_all = boxes_word_all.to(device=device, dtype=self.target_dtype)

            if boxes_word_all.size(0) != N_word:
                if boxes_word_all.size(0) > N_word:
                    boxes_word_all = boxes_word_all[:N_word]
                else:
                    pad = torch.zeros(N_word - boxes_word_all.size(0), 4, device=device, dtype=self.target_dtype)
                    boxes_word_all = torch.cat([boxes_word_all, pad], dim=0)

            # Map boxes sang token level
            boxes_tok_i = boxes_word_all[map_clamped] * valid.unsqueeze(-1)

            # --- XỬ LÝ DET FEATURES ---
            det_word_all = info_i.get("det_features")
            if det_word_all is not None:
                if not torch.is_tensor(det_word_all):
                    det_word_all = torch.tensor(det_word_all, device=device, dtype=self.target_dtype)
                else:
                    det_word_all = det_word_all.to(device=device, dtype=self.target_dtype)

                if det_word_all.size(0) != N_word:
                    if det_word_all.size(0) > N_word:
                        det_word_all = det_word_all[:N_word]
                    else:
                        pad_d = torch.zeros(N_word - det_word_all.size(0), det_word_all.size(-1), device=device, dtype=self.target_dtype)
                        det_word_all = torch.cat([det_word_all, pad_d], dim=0)

                # Map det sang token level
                det_tok_i = det_word_all[map_clamped] * valid.unsqueeze(-1)
            else:
                det_tok_i = None

            # --- XỬ LÝ REC FEATURES ---
            rec_word_all = info_i.get("rec_features")
            if rec_word_all is not None:
                if not torch.is_tensor(rec_word_all):
                    rec_word_all = torch.tensor(rec_word_all, device=device, dtype=self.target_dtype)
                else:
                    rec_word_all = rec_word_all.to(device=device, dtype=self.target_dtype)

                if rec_word_all.size(0) != N_word:
                    if rec_word_all.size(0) > N_word:
                        rec_word_all = rec_word_all[:N_word]
                    else:
                        pad_r = torch.zeros(N_word - rec_word_all.size(0), rec_word_all.size(-1), device=device, dtype=self.target_dtype)
                        rec_word_all = torch.cat([rec_word_all, pad_r], dim=0)

                # Map rec sang token level
                rec_tok_i = rec_word_all[map_clamped] * valid.unsqueeze(-1)
            else:
                rec_tok_i = None

            # --- XỬ LÝ WORD MASK ---
            if box_mask is not None:
                word_mask_all = box_mask[i]
            else:
                word_mask_all = info_i.get("word_mask_all", None)
                if word_mask_all is None:
                    word_mask_all = torch.ones(boxes_word_all.size(0), device=device, dtype=torch.long)

            if not torch.is_tensor(word_mask_all):
                word_mask_all = torch.tensor(word_mask_all, device=device, dtype=torch.long)
            else:
                word_mask_all = word_mask_all.to(device=device)

            if word_mask_all.size(0) != N_word:
                if word_mask_all.size(0) > N_word:
                    word_mask_all = word_mask_all[:N_word]
                else:
                    padm = torch.zeros(N_word - word_mask_all.size(0), device=device, dtype=torch.long)
                    word_mask_all = torch.cat([word_mask_all, padm], dim=0)

            tok_mask_all_i = word_mask_all[map_clamped] * valid.long()

            # Bổ sung det và rec đã ở token level vào dictionary để đẩy qua hàm Semantic
            sal_info_tok = {
                "width": info_i["width"],
                "height": info_i["height"],
                "boxes": boxes_tok_i,
                "det": det_tok_i,
                "rec": rec_tok_i
            }

            sal_input_i, _ = self.semantic_ocr_embedding(
                [sal_info_tok],
                ocr_text_tok[i].unsqueeze(0),
                char_feat_tok_i.unsqueeze(0)
            )



            ocr_box_feat_tok_list.append(sal_input_i.squeeze(0))

        final_ocr_feat = torch.stack(ocr_box_feat_tok_list, dim=0).to(self.target_dtype)
        return final_ocr_feat

    # --- ENCODE OCR (BẢN LITE BASELINE) ---
    def _encode_ocr_baseline_features(
        self,
        ocr_info: List[Dict[str, Any]],
        ocr_token_ids: torch.Tensor,
        ocr_to_word_map: torch.Tensor,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor,
        token_mask: torch.Tensor,
        box_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        ocr_token_ids = ocr_token_ids.to(device)
        ocr_to_word_map = ocr_to_word_map.to(device)
        char_ids = char_ids.to(device)
        char_mask = char_mask.to(device)
        token_mask = token_mask.to(device)
    
        B, L_tok = ocr_token_ids.size()
        if ocr_to_word_map.size(1) != L_tok:
            ocr_to_word_map = _pad_or_crop_lastdim_int(
                ocr_to_word_map,
                L_tok,
                pad_value=-1,
            )
    
        N_word = int(char_ids.size(1))
        D = self.d_model
    
        tok_emb = self.vit5.get_input_embeddings()(ocr_token_ids).to(
            dtype=self.target_dtype
        )
    
        text_word_feat = torch.zeros(
            B,
            N_word,
            D,
            device=device,
            dtype=self.target_dtype,
        )
        word_counts = torch.zeros(
            B,
            N_word,
            device=device,
            dtype=self.target_dtype,
        )
    
        for b in range(B):
            map_b = ocr_to_word_map[b]
            tok_mask_b = token_mask[b]
            valid = (map_b >= 0) & (tok_mask_b > 0)
    
            if not valid.any() or N_word == 0:
                continue
    
            idx = map_b[valid].clamp(0, N_word - 1)
            src = tok_emb[b][valid]
    
            text_word_feat[b].scatter_add_(
                0,
                idx.unsqueeze(-1).expand_as(src),
                src,
            )
            word_counts[b].scatter_add_(
                0,
                idx,
                torch.ones_like(
                    idx,
                    dtype=self.target_dtype,
                    device=device,
                ),
            )
    
        text_word_feat = text_word_feat / word_counts.unsqueeze(-1).clamp_min(1e-6)
    
        char_feat_word = _char_embedding(
            self.char_embedding,
            self.char_position_embedding,
            char_ids,
            char_mask,
            mean=True,
        )
        char_feat_word = self.ocr_char_layernorm(char_feat_word).to(self.target_dtype)
    
        text_mix = self.ocr_lite_text_ln(
            (text_word_feat + char_feat_word).to(torch.float32)
        ).to(self.target_dtype)
    
        text_mix = self.ocr_lite_text_proj(
            text_mix.to(torch.float32)
        ).to(self.target_dtype)
    
        text_mix = (
            text_mix
            + self.ocr_lite_text_ff(text_mix.to(torch.float32)).to(self.target_dtype)
        )
    
        box_word = torch.zeros(
            B,
            N_word,
            4,
            device=device,
            dtype=self.target_dtype,
        )
        ocr_word_mask_list = []
    
        for b in range(B):
            info_b = ocr_info[b]
            w = float(info_b.get("width", 1.0) or 1.0)
            h = float(info_b.get("height", 1.0) or 1.0)
    
            boxes = info_b.get("boxes_word_all", None)
            if boxes is not None:
                if not torch.is_tensor(boxes):
                    boxes = torch.tensor(
                        boxes,
                        device=device,
                        dtype=self.target_dtype,
                    )
                else:
                    boxes = boxes.to(device=device, dtype=self.target_dtype)
    
                if boxes.dim() == 2 and boxes.size(-1) == 4:
                    if boxes.size(0) != N_word:
                        if boxes.size(0) > N_word:
                            boxes = boxes[:N_word]
                        else:
                            pad = torch.zeros(
                                N_word - boxes.size(0),
                                4,
                                device=device,
                                dtype=self.target_dtype,
                            )
                            boxes = torch.cat([boxes, pad], dim=0)
    
                    box_word[b] = _normalize_boxes_auto(
                        boxes,
                        w,
                        h,
                        device,
                        self.target_dtype,
                    )
    
            box_mask_b = box_mask[b] if box_mask is not None else None
            word_mask_b = self._get_ocr_word_mask(
                info_b,
                box_mask_b,
                N_word,
                device,
            )
            ocr_word_mask_list.append(word_mask_b)
    
        ocr_word_mask = torch.stack(ocr_word_mask_list, dim=0).long()
    
        box_emb = self.ocr_lite_box_proj(
            box_word.to(torch.float32)
        ).to(self.target_dtype)
    
        box_emb = (
            box_emb
            + self.ocr_lite_box_ff(box_emb.to(torch.float32)).to(self.target_dtype)
        )
    
        mask = ocr_word_mask.to(self.target_dtype).unsqueeze(-1)
    
        ocr_fused_feat = (text_mix + box_emb).to(self.target_dtype)
        ocr_fused_feat = ocr_fused_feat * mask
    
        return ocr_fused_feat, ocr_word_mask

    # =====================================================================
    # LUỒNG XỬ LÝ CHÍNH: HÀM FORWARD
    # =====================================================================
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pil_images: Optional[List] = None,
        ocr_info: Optional[List[Dict[str, Any]]] = None,
        ocr_mask_token: Optional[torch.LongTensor] = None,
        ocr_mask_box: Optional[torch.LongTensor] = None,
        mlm_input_ids: Optional[torch.LongTensor] = None,
        cmb_text_mask_label: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        twa_ocr_char: Optional[torch.LongTensor] = None,
        twa_ocr_char_mask: Optional[torch.FloatTensor] = None,
        twa_word_ids: Optional[torch.LongTensor] = None,
        ocr_to_word_map: Optional[torch.LongTensor] = None,
        twc_split_word_idx: Optional[torch.LongTensor] = None,
        tag_pollute: Optional[torch.LongTensor] = None,
        o2r_labels: Optional[torch.FloatTensor] = None,
        r2o_labels: Optional[torch.FloatTensor] = None,
        return_visual_search_debug: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        
        device = _any_device_fallback(input_ids=input_ids, pixel_values=pixel_values, mlm_input_ids=mlm_input_ids)
        assert pixel_values is not None, "pixel_values is required"

        # ĐỌC CÁC CÔNG TẮC ABLATION TỪ CONFIG
        use_qaclip = bool(getattr(self.config, "ablation_use_qaclip", True))
        use_vs = bool(getattr(self.config, "ablation_use_vs", True))
        use_ocr = bool(getattr(self.config, "ablation_use_ocr", True))

        if self.pretrain:
            assert mlm_input_ids is not None, "mlm_input_ids required in pretrain"
            assert cmb_text_mask_label is not None, "cmb_text_mask_label required in pretrain"
            q_ids_for_enc = mlm_input_ids.to(device)
            enc_attention_mask = attention_mask.to(device) if attention_mask is not None else None
            max_q_len_cfg = int(getattr(self.config, "text_max_input_length", q_ids_for_enc.size(1)))
            max_q_len = min(max_q_len_cfg, q_ids_for_enc.size(1))
            q_ids_for_clip = q_ids_for_enc[:, :max_q_len]
        else:
            assert input_ids is not None, "input_ids required in finetune"
            q_ids_for_enc = input_ids.to(device)
            enc_attention_mask = attention_mask.to(device) if attention_mask is not None else None
            q_ids_for_clip = q_ids_for_enc

        txt_emb_for_enc, txt_attn_mask_for_enc = self._encode_text(q_ids_for_enc, enc_attention_mask, device)

        # Encoded question for QACLIP
        txt_outputs = self.vit5.encoder(
            input_ids=q_ids_for_enc,
            attention_mask=txt_attn_mask_for_enc,
            return_dict=True
        )
        txt_hidden_states = txt_outputs.last_hidden_state.to(dtype=self.target_dtype)

        if self.pretrain:
            txt_emb_for_clip = txt_hidden_states[:, : q_ids_for_clip.size(1)]
            txt_attn_mask_for_clip = txt_attn_mask_for_enc[:, : q_ids_for_clip.size(1)]
        else:
            txt_emb_for_clip = txt_hidden_states
            txt_attn_mask_for_clip = txt_attn_mask_for_enc

        pixel_values_dev = pixel_values.to(device)
        B = pixel_values_dev.size(0)
        D = self.d_model

        # ----------------------------------------------------
        # 1. ABLATION MODULE: QACLIP
        # ----------------------------------------------------
        img_pack = self._encode_image(
            pixel_values=pixel_values_dev,
            device=device,
            txt_emb=txt_emb_for_clip,
            txt_mask=txt_attn_mask_for_clip,
            fuse_with_text=use_qaclip,  # Tắt True/False ở đây
            return_attn=return_visual_search_debug,
        )

        # ----------------------------------------------------
        # 2. ABLATION MODULE: VISUAL SEARCH
        # ----------------------------------------------------
        if use_vs:
            vs_out = self.visual_search(
                img_tokens=img_pack["img_tokens"],
                patch_scores=img_pack["patch_scores"],
                pixel_values=pixel_values_dev,
                return_debug=return_visual_search_debug,
                pil_images=pil_images,
            )
            attn_summary = _ensure_1_token(vs_out.get("attn_summary", None), B, D, device, self.target_dtype)
            crop_tokens = vs_out.get("crop_tokens", torch.zeros(B, 0, D, device=device, dtype=self.target_dtype))
            crop_mask = vs_out.get("crop_mask", torch.zeros(B, 0, device=device, dtype=torch.long))
        else:
            # Baseline (Global View): Ép box bao trọn toàn bộ ảnh
            W_img = float(self.visual_search.image_size.item())
            global_box = torch.tensor([[0.0, 0.0, W_img, W_img]], device=device, dtype=self.target_dtype)
            global_boxes = global_box.expand(B, 4)
            
            # ConvNeXt vẫn hoạt động trên toàn cảnh bức ảnh
            dummy_crop_tokens, _ = self.visual_search._extract_roi_features(pixel_values_dev, global_boxes)
            
            vs_out = {}
            crop_tokens = dummy_crop_tokens.to(self.target_dtype)
            crop_mask = torch.ones(B, dummy_crop_tokens.size(1), device=device, dtype=torch.long)
            attn_summary = img_pack["img_tokens"].mean(dim=1, keepdim=True)

        # ----------------------------------------------------
        # 3. ABLATION MODULE: OCR CONSFORMER
        # ----------------------------------------------------
        assert twa_word_ids is not None, "twa_word_ids required"
        assert ocr_to_word_map is not None, "ocr_to_word_map required"
        assert twa_ocr_char is not None, "twa_ocr_char required"
        assert twa_ocr_char_mask is not None, "twa_ocr_char_mask required"
        assert ocr_info is not None, "ocr_info required"

        word_ids_for_ocr = twa_word_ids.to(device)
        pad_id = self.vit5.config.pad_token_id
        token_mask_for_ocr = (word_ids_for_ocr != pad_id).long()
        ocr_map = ocr_to_word_map.to(device).long()
        L_tok = word_ids_for_ocr.size(1)

        if ocr_map.size(1) != L_tok: 
            ocr_map = _pad_or_crop_lastdim_int(ocr_map, L_tok, pad_value=-1)
        if token_mask_for_ocr.size(1) != L_tok: 
            token_mask_for_ocr = _pad_or_crop_lastdim_int(token_mask_for_ocr, L_tok, pad_value=0)

        char_ids_for_ocr = twa_ocr_char.to(device)
        char_mask_for_ocr = twa_ocr_char_mask.to(device)
        ocr_box_mask_for_ocr = ocr_mask_box.to(device).long() if ocr_mask_box is not None else None

        if use_ocr:
            ocr_fused_feat = self._encode_ocr_features(
                ocr_info,
                word_ids_for_ocr,
                ocr_map,
                char_ids_for_ocr,
                char_mask_for_ocr,
                token_mask_for_ocr,
                ocr_box_mask_for_ocr,
                device,
            )
        else:
            # Baseline: Dùng Linear thay vì Spatial/Group Attention
            # NOTE: baseline trả về word-level (N_word), không phải token-level (L_tok)
            ocr_fused_feat, mask_ocr = self._encode_ocr_baseline_features(
                ocr_info, word_ids_for_ocr, ocr_map, twa_ocr_char,
                twa_ocr_char_mask, token_mask_for_ocr, ocr_box_mask_for_ocr, device
            )
            # Cập nhật mask theo word-level từ baseline
            token_mask_for_ocr = mask_ocr

        # Chỉ pad/crop khi dùng full OCR Consformer (token-level)
        if use_ocr:
            L_tok = word_ids_for_ocr.size(1)
            if ocr_fused_feat.size(1) != L_tok:
                ocr_fused_feat = _pad_or_crop_lastdim(ocr_fused_feat, L_tok, pad_value=0.0)
            if token_mask_for_ocr.size(1) != L_tok:
                token_mask_for_ocr = _pad_or_crop_lastdim_int(token_mask_for_ocr, L_tok, pad_value=0)

        # attn_summary, crop_tokens, crop_mask were already assigned in the use_vs block above

        # CONCAT CHUỖI VÀO ENCODER: CHỈ CÓ 1 KHỐI ocr_fused_feat
        fused_seq = torch.cat(
            [
                txt_emb_for_enc,
                img_pack["img_tokens"],
                ocr_fused_feat,
                crop_tokens,
                attn_summary,
            ],
            dim=1,
        )

        mask_txt = txt_attn_mask_for_enc.long()
        mask_img = img_pack["img_attn_mask"].long()
        mask_ocr = token_mask_for_ocr.long()
        mask_crop = crop_mask.long()
        mask_attn = torch.ones(B, attn_summary.size(1), device=device, dtype=torch.long)

        # GHÉP ATTENTION MASK KHỚP HOÀN TOÀN VỚI FUSED_SEQ
        fused_mask = torch.cat(
            [
                mask_txt,
                mask_img,
                mask_ocr,
                mask_crop,
                mask_attn,
            ],
            dim=1,
        )

        if fused_mask.size(1) != fused_seq.size(1):
            if fused_mask.size(1) > fused_seq.size(1):
                fused_mask = fused_mask[:, : fused_seq.size(1)]
            else:
                pad = torch.zeros(fused_mask.size(0), fused_seq.size(1) - fused_mask.size(1), device=device, dtype=torch.long)
                fused_mask = torch.cat([fused_mask, pad], dim=1)

        if not return_visual_search_debug:
            for k in list(vs_out.keys()):
                if k not in ("attn_summary", "crop_tokens", "crop_mask"): del vs_out[k]
            for k in list(img_pack.keys()):
                if k not in ("img_tokens", "img_attn_mask", "patch_scores"): del img_pack[k]

        enc_out = self.vit5.encoder(inputs_embeds=fused_seq, attention_mask=fused_mask, return_dict=True)

        out_dict: Dict[str, Any] = {"encoder_outputs": enc_out, "attention_mask": fused_mask}

        if self.pretrain:
            mlm_labels = cmb_text_mask_label.to(device)
            mlm_labels = torch.where(mlm_labels == -1, torch.full_like(mlm_labels, -100), mlm_labels)

            outputs = self.vit5(
                encoder_outputs=enc_out,
                attention_mask=fused_mask,
                labels=mlm_labels,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            out_dict["textcls_scores"] = outputs.logits
            out_dict["mlm_loss"] = outputs.loss

            dec_last = outputs.decoder_hidden_states[-1]
            dec_first = dec_last[:, 0, :]
            out_dict["pollutecls_scores"] = self.pollute_head(dec_first.float())

            out_dict["contrastive_scores"] = None
            out_dict["o2r_block"] = None
            out_dict["r2o_block"] = None

            use_twc = bool(getattr(self.config, "use_twc", getattr(self, "use_twc", True)))

            # SỬA LỖI INDEXERROR: ĐỊNH VỊ CHÍNH XÁC PHÂN ĐOẠN OCR TRONG LAST HIDDEN STATE
            if use_twc and use_ocr and o2r_labels is not None:
                enc_hid = enc_out.last_hidden_state
                L_txt = txt_emb_for_enc.size(1)
                L_img = img_pack["img_tokens"].size(1)
                L_ocr = ocr_fused_feat.size(1)

                off_img = L_txt
                off_ocr = off_img + L_img # Vị trí bắt đầu chính xác của ocr_fused_feat

                # Cắt chuẩn xác tensor đại diện cho OCR (chiều dài là L_ocr khớp hoàn toàn với valid_map_mask)
                ocr_enc_out = enc_hid[:, off_ocr: off_ocr + L_ocr, :]

                valid_map_mask = (ocr_map >= 0) & (token_mask_for_ocr > 0)

                if valid_map_mask.any():
                    Bc, _, D0 = ocr_enc_out.size()
                    N_word = int(twa_ocr_char.size(1))

                    if N_word % 2 != 0:
                        raise RuntimeError(f"TWC requires OCR original + OCR augmented pairs, but N_word={N_word} is odd.")

                    word_vectors = torch.zeros(Bc, N_word, D0, device=device, dtype=ocr_enc_out.dtype)
                    word_counts = torch.zeros(Bc, N_word, device=device, dtype=ocr_enc_out.dtype)

                    for b in range(Bc):
                        vmask_b = valid_map_mask[b]
                        if not vmask_b.any(): continue

                        src = ocr_enc_out[b][vmask_b] # ĐÃ KHỚP SHAPE [196] với [196, 768]
                        idx = ocr_map[b][vmask_b].clamp(0, max(N_word - 1, 0))

                        word_vectors[b].scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
                        ones = torch.ones_like(idx, dtype=word_counts.dtype, device=device)
                        word_counts[b].scatter_add_(0, idx, ones)

                    word_counts = word_counts.unsqueeze(-1)
                    word_counts_inv = torch.where(word_counts > 0, 1.0 / word_counts, torch.zeros_like(word_counts))
                    word_vectors = word_vectors * word_counts_inv

                    half = N_word // 2
                    ocr_feat = word_vectors[:, :half, :]
                    rel_feat = word_vectors[:, half:, :]

                    def _safe_l2(x, dim=-1, eps=1e-6):
                        n = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True) + eps)
                        return x / n

                    ocr_feat = _safe_l2(ocr_feat, dim=-1)
                    rel_feat = _safe_l2(rel_feat, dim=-1)

                    Bn, W, Dn = ocr_feat.shape
                    ocr_flat = ocr_feat.reshape(Bn * W, Dn)
                    rel_flat = rel_feat.reshape(Bn * W, Dn)

                    logit_scale = self.logit_scale.clamp(min=np.log(1 / 100), max=np.log(100)).exp()

                    contrastive_scores = logit_scale * (ocr_flat @ rel_flat.t())
                    contrastive_scores = torch.nan_to_num(contrastive_scores, nan=0.0, posinf=1e4, neginf=-1e4)

                    out_dict["contrastive_scores"] = contrastive_scores

                    B_lab, _, _ = o2r_labels.shape
                    blocks = [o2r_labels[b].to(device) for b in range(B_lab)]
                    out_dict["o2r_block"] = torch.block_diag(*blocks)

                    if r2o_labels is not None:
                        blocks_r = [r2o_labels[b].to(device) for b in range(B_lab)]
                        out_dict["r2o_block"] = torch.block_diag(*blocks_r)
                    else:
                        out_dict["r2o_block"] = None

            if return_visual_search_debug:
                out_dict["vs_debug"] = vs_out
                out_dict["clip_input_ids"] = q_ids_for_clip.detach().cpu()

            return out_dict

        if labels is not None:
            outputs = self.vit5(
                encoder_outputs=enc_out,
                attention_mask=fused_mask,
                labels=labels.to(device),
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            out_dict["loss"] = outputs.loss
            out_dict["logits"] = outputs.logits
        else:
            out_dict["logits"] = None

        if return_visual_search_debug:
            out_dict["vs_debug"] = vs_out
            out_dict["clip_input_ids"] = q_ids_for_clip.detach().cpu()

        return out_dict

    # --- HÀM TẠO SINH ---
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        pil_images: Optional[List] = None,
        ocr_info: Optional[List[Dict[str, Any]]] = None,
        ocr_mask_token: Optional[torch.LongTensor] = None,
        ocr_mask_box: Optional[torch.LongTensor] = None,
        max_new_tokens: int = None,
        num_beams: int = None,
        **kwargs,
    ):
        gen_cfg = kwargs.pop("generation_config", None)
        if gen_cfg is None:
            gen_cfg = (
                GenerationConfig.from_dict(self.generation_config.to_dict())
                if getattr(self, "generation_config", None) is not None
                else GenerationConfig()
            )

        twa_keys = ["twa_ocr_char", "twa_ocr_char_mask", "twa_word_ids", "ocr_to_word_map", "twc_split_word_idx"]
        twa_kwargs = {k: kwargs.pop(k) for k in twa_keys if k in kwargs}
        forward_kwargs = {k: v for k, v in kwargs.items() if k not in ["max_new_tokens", "num_beams", "generation_config"]}

        orig_pretrain = self.pretrain
        self.pretrain = False
        try:
            enc = self.forward(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                labels=None, pil_images=pil_images, ocr_info=ocr_info,
                ocr_mask_token=ocr_mask_token, ocr_mask_box=ocr_mask_box,
                **twa_kwargs, **forward_kwargs,
            )
        finally:
            self.pretrain = orig_pretrain

        if max_new_tokens is not None: gen_cfg.max_new_tokens = max_new_tokens
        if num_beams is not None: gen_cfg.num_beams = num_beams

        return self.vit5.generate(
            encoder_outputs=enc["encoder_outputs"], attention_mask=enc["attention_mask"],
            use_cache=False, generation_config=gen_cfg, **kwargs,
        )