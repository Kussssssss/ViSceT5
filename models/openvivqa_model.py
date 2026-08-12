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
from models.modules.ocr_spatial import SemanticOCREmbedding, SpatialCirclePosition
from models.modules.visual_search import VisualSearch

import os
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
    ocr_char_emb = ocr_char_emb * ocr_char_mask.unsqueeze(-1)
    if mean:
        ocr_char_emb = ocr_char_emb.mean(dim=-2)
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
        # Tắt AVF = bỏ hẳn module → 27.87M tham số (chủ yếu ConvNeXt) không còn dùng tới.
        # Vẫn dựng ở trên để tiêu RNG đúng thứ tự (nó đứng TRƯỚC mọi module khác), rồi mới
        # gỡ đi: tiết kiệm ~111MB VRAM và một lượt .to(device), state_dict cũng sạch.
        if not bool(getattr(self.config, "ablation_use_vs", True)):
            del self.visual_search

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
        # SpatialCirclePosition KHÔNG được gọi trên bất kỳ đường forward nào (đã kiểm bằng
        # forward hook: 0 lần ở cả 6 cấu hình ablation) → 2.36M tham số chết, nằm trong
        # state_dict và chiếm state của optimizer.
        # Vẫn PHẢI khởi tạo nó ở ĐÚNG vị trí này vì nó rút RNG global; bỏ hẳn lời gọi sẽ
        # làm lệch trọng số khởi tạo của MỌI module dựng sau (char_*/ocr_lite/pollute_head/
        # init_qavit_comps) so với notebook. Nên: vẫn dựng để tiêu RNG, nhưng KHÔNG gán vào
        # self → không vào state_dict, không vào optimizer.
        _rng_only_spatial = SpatialCirclePosition(ns)
        del _rng_only_spatial

        self.char_max_num = int(getattr(config, "char_max_num", 50))
        self.char_num = int(getattr(config, "char_num"))
        self.char_position_embedding = nn.Embedding(self.char_max_num, self.d_model)
        self.char_embedding = nn.Embedding(self.char_num, self.d_model)
        self.ocr_char_layernorm = T5LayerNorm(self.d_model, eps=1e-12)

        # Mạng Baseline cho OCR (Bản Lite - Dùng khi tắt OCR Module)
        self.ocr_lite_text_ln = T5LayerNorm(self.d_model, eps=1e-12)
        self.ocr_lite_text_proj = nn.Linear(self.d_model, self.d_model)
        # Hai MLP _ff đã ngưng dùng từ 780c6f7 (baseline OCR chỉ còn Linear để đối xứng với
        # hai nhánh OFF còn lại) → 2.36M tham số chết. Cũng dựng-rồi-bỏ để giữ RNG.
        _rng_only_text_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model),
        )
        self.ocr_lite_box_proj = nn.Linear(4, self.d_model)
        _rng_only_box_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model),
        )
        del _rng_only_text_ff, _rng_only_box_ff
        with torch.no_grad():
            self.ocr_lite_text_proj.weight.copy_(torch.eye(self.d_model))
            self.ocr_lite_text_proj.bias.zero_()

        self.pretrain = bool(getattr(self.config, "pretrain", True))
        self.pretrain_ablation_mode = str(
            getattr(self.config, "pretrain_ablation_mode", "full")
        ).lower().strip()
        self.use_twc = bool(getattr(self.config, "use_twc", True))

        self.pollute_head = T5PolluteHead(input_size=self.d_model, layer_norm_eps=1e-12).to(torch.float32)

        # ITC (Image-Text Contrastive) heads — align image ↔ question in a shared space
        # (ALBEF "align-before-fuse"). PRETRAIN-ONLY: nhánh forward/loss ITC đều nằm dưới
        # `if self.pretrain` (xem forward), nên CHỈ dựng khi self.pretrain=True. Nhờ vậy
        # model FINETUNE không hề tạo ITC → state_dict + thứ tự rút RNG khởi tạo khớp
        # notebook (notebook không có ITC), tránh tham số chết. Finetune ép config.pretrain
        # =False TRƯỚC khi dựng model (xem training/finetune.py) nên nhánh này bị bỏ qua.
        self.itc_dim = 256
        if self.pretrain:
            self.itc_img_proj = nn.Linear(self.d_model, self.itc_dim)
            self.itc_txt_proj = nn.Linear(self.d_model, self.itc_dim)
            self.itc_logit_scale = nn.Parameter(torch.tensor(2.659))  # exp≈14.3 (CLIP-style)

        # ── TWC Projection Heads ────────────────────────────────────────────────
        # TWC (theo notebook gốc / TWA paper): similarity tính trực tiếp trên
        # word-feature đã L2-normalize từ encoder — KHÔNG dùng projection head riêng.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.generation_config = GenerationConfig(
            max_new_tokens=int(getattr(config, "generation_max_new_tokens", 27)),
            num_beams=int(getattr(config, "generation_num_beams", 4)),
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            decoder_start_token_id=self.config.decoder_start_token_id,
            do_sample=False,
        )

        # det/rec (SwinTextSpotter detection/recognition, 256-d) là ĐẶC TRƯNG OCR bình
        # thường (giống text/box) → chiếu Linear về d_model rồi cộng vào baseline OCR.
        # Đặt Ở CUỐI __init__ để KHÔNG xê dịch thứ tự rút RNG của các module phía trước.
        self.ocr_lite_det_proj = nn.Linear(int(getattr(config, "ocr_d_det", 256)), self.d_model)
        self.ocr_lite_rec_proj = nn.Linear(int(getattr(config, "ocr_d_rec", 256)), self.d_model)
        # LayerNorm SAU Linear — GIỐNG HỆT đường ON (SemanticOCREmbedding.layer_norm_det/rec)
        # và đúng công thức paper: x_OCR = x_semantic + LN(det·W) + LN(rec·W) + LN(box·W).
        # THIẾU LN thì det/rec cộng vào fused_seq với biên độ không kiểm soát → gradient lớn
        # → backward của QA-CLIP tràn → NaN (đo được: chỉ lỗi khi det/rec ON + qaclip ON).
        self.ocr_lite_det_ln = nn.LayerNorm(self.d_model)
        self.ocr_lite_rec_ln = nn.LayerNorm(self.d_model)

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
    def tie_weights(self, *args, **kwargs):
        # Accept/ignore extra kwargs (e.g. recompute_mapping) that newer
        # transformers pass from init_weights(); our tie logic just defers to ViT5.
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
        fuse_with_text: bool = True, return_attn: bool = False,
        need_attn_map: bool = True,
    ):
        # need_attn_map=False khi AVF tắt: patch_scores chỉ phục vụ việc chọn vùng crop.
        # Xin output_attentions buộc HF bỏ CLIPSdpaAttention để quay về attention thủ công
        # (nó tự cảnh báo điều này), tức là trả giá tốc độ + bộ nhớ cho một tensor
        # (B, 12, 197, 229) không ai dùng.
        want_attn = bool(need_attn_map or return_attn)
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
        else:
            # Baseline (Tắt QA): Không có câu hỏi định hướng
            D_txt = int(getattr(self.qa_clip.config, "instruction_dim", self.d_model))
            text_emb = torch.zeros(B, 0, D_txt, device=device, dtype=self.target_dtype)
            text_mask = torch.zeros(B, 0, dtype=torch.long, device=device)
            T = 0

        # PRETRAIN STABILITY: QA-CLIP vision is FROZEN (freeze_clip). Its custom
        # attention occasionally emits NaN in the forward (nondeterministic); the
        # fused_seq nan_to_num guard cleans the VALUE but the NaN still poisons the
        # BACKWARD through this module, corrupting grads for vit5/visual_search. Since
        # nothing here needs a gradient during pretrain, run it under no_grad and
        # sanitize the output → the NaN can never reach the trainable modules. Gated by
        # `_pretrain_stage` (set only in pretrain.py; stays set through the cloze forward
        # which flips `pretrain=False`), so FINETUNE is completely untouched.
        # no_grad+sanitize the vision ONLY when it is truly frozen in pretrain. If vision
        # is being UNFROZEN (_vision_trainable), let grads flow (ITC/unfreeze needs them);
        # the NaN root fix in MMCLIPAttention + the fused_seq guard keep it stable.
        _frozen_pt = (bool(getattr(self, "_pretrain_stage", False))
                      and bool(getattr(self.qa_clip.config, "freeze_clip", False))
                      and not bool(getattr(self, "_vision_trainable", False)))
        if _frozen_pt:
            with torch.no_grad():
                qa_out = self.qa_clip(pixel_values=pixel_values, text_emb=text_emb, text_mask=text_mask, output_attentions=want_attn, return_dict=True)
            img_hs = torch.nan_to_num(qa_out.last_hidden_state, nan=0.0, posinf=1e4, neginf=-1e4)
        else:
            qa_out = self.qa_clip(pixel_values=pixel_values, text_emb=text_emb, text_mask=text_mask, output_attentions=want_attn, return_dict=True)
            img_hs = qa_out.last_hidden_state
            if getattr(self, "_pretrain_stage", False) or bool(getattr(self.config, "clamp_vision", False)):
                # UNFROZEN pretrain: QA-CLIP's RANDOM-INIT instruction adapters can overflow
                # (~1e36 → inf). Sanitize + clamp WITHOUT detaching — grads still flow.
                # config.clamp_vision: weights PRETRAINED under this clamp expect it as part
                # of the forward. Scratch finetune (no flag) stays byte-identical to notebook.
                img_hs = torch.nan_to_num(img_hs, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

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

            patch_scores = patch_scores.to(self.target_dtype)
            if getattr(self, "_pretrain_stage", False) or bool(getattr(self.config, "clamp_vision", False)):
                # covers BOTH frozen and unfrozen pretrain — attention scores from the
                # overflowing adapters poison visual_search otherwise; clamp_vision: same
                # guard when finetuning/inferring FROM such weights.
                patch_scores = torch.nan_to_num(patch_scores, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            out["patch_scores"] = patch_scores
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

        # Notebook gốc / TWA paper: original và related token được encode CÙNG NHAU
        # với full attention (z_ocr = [z_T; z_W] qua transformer). KHÔNG chặn cross-half.
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
                        # KHỚP NOTEBOOK: nếu N_word gấp đôi det_word_all (do OCR augmentation
                        # ở finetune tạo nửa gốc + nửa nhiễu), COPY nửa gốc để nửa augmented
                        # dùng lại đặc trưng thị giác của OCR gốc tương ứng, thay vì zero-pad.
                        if N_word == 2 * det_word_all.size(0):
                            det_word_all = torch.cat([det_word_all, det_word_all], dim=0)
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
                        # KHỚP NOTEBOOK: copy nửa gốc khi N_word gấp đôi — xem chú thích det ở trên.
                        if N_word == 2 * rec_word_all.size(0):
                            rec_word_all = torch.cat([rec_word_all, rec_word_all], dim=0)
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

        # ── TOKEN-LEVEL (khop dung duong ON) ───────────────────────────────────
        # Baseline phai o CUNG cap va CUNG do dai voi _encode_ocr_features (L_tok), de
        # ablation chi doi PHAN XU LY (Constituent + GroupAttention + SpatialCircle) chu
        # khong doi luon do phan giai chuoi. Truoc day baseline gop sub-word -> word
        # (N_word) nen chuoi ngan hon ~1.8x, lam phep so sanh lan hai bien.
        tok_emb = self.vit5.get_input_embeddings()(ocr_token_ids).to(dtype=self.target_dtype)

        char_feat_word = _char_embedding(
            self.char_embedding, self.char_position_embedding, char_ids, char_mask, mean=True,
        )
        char_feat_word = self.ocr_char_layernorm(char_feat_word).to(self.target_dtype)

        _d_det = self.ocr_lite_det_proj.in_features
        _d_rec = self.ocr_lite_rec_proj.in_features
        char_tok = torch.zeros(B, L_tok, D, device=device, dtype=self.target_dtype)
        box_tok = torch.zeros(B, L_tok, 4, device=device, dtype=self.target_dtype)
        det_tok = torch.zeros(B, L_tok, _d_det, device=device, dtype=self.target_dtype)
        rec_tok = torch.zeros(B, L_tok, _d_rec, device=device, dtype=self.target_dtype)
        _has_det = _has_rec = False

        def _to_word(feat, dim):
            """Dua dac trung word-level cua 1 mau ve dung N_word (crop / copy nua goc khi
            OCR-aug nhan doi / zero-pad) — cung quy tac voi duong ON."""
            if feat is None:
                return None
            if not torch.is_tensor(feat):
                feat = torch.tensor(feat, device=device, dtype=self.target_dtype)
            else:
                feat = feat.to(device=device, dtype=self.target_dtype)
            if feat.dim() != 2 or feat.size(-1) != dim:
                return None
            n = feat.size(0)
            if n == N_word:
                return feat
            if n > N_word:
                return feat[:N_word]
            if N_word == 2 * n:
                return torch.cat([feat, feat], dim=0)
            pad = torch.zeros(N_word - n, dim, device=device, dtype=self.target_dtype)
            return torch.cat([feat, pad], dim=0)

        for b in range(B):
            map_b = ocr_to_word_map[b]
            valid = ((map_b >= 0) & (token_mask[b] > 0)).unsqueeze(-1)
            map_clamped = map_b.clamp(min=0, max=max(N_word - 1, 0))

            char_tok[b] = char_feat_word[b][map_clamped] * valid

            info_b = ocr_info[b]
            w = float(info_b.get("width", 1.0) or 1.0)
            h = float(info_b.get("height", 1.0) or 1.0)
            _bx = _to_word(info_b.get("boxes_word_all"), 4)
            if _bx is not None:
                _bx = _normalize_boxes_auto(_bx, w, h, device, self.target_dtype)
                box_tok[b] = _bx[map_clamped] * valid

            _dt = _to_word(info_b.get("det_features"), _d_det)
            if _dt is not None:
                det_tok[b] = _dt[map_clamped] * valid; _has_det = True
            _rc = _to_word(info_b.get("rec_features"), _d_rec)
            if _rc is not None:
                rec_tok[b] = _rc[map_clamped] * valid; _has_rec = True

        # Hop nhat TUYEN TINH — khac biet DUY NHAT so voi ON: khong Constituent/Group/Spatial
        # CHỈ Linear (+LayerNorm) — KHÔNG dùng ocr_lite_text_ff / ocr_lite_box_ff (MLP 2 lớp
        # GELU có residual). Lý do: hai nhánh OFF còn lại KHÔNG thêm lớp học được nào
        # (QA-ViT OFF = CLIP thuần; AVF OFF = cùng ConvNeXt, chỉ đổi box), nên baseline OCR
        # cũng phải là "đặc trưng đi qua Linear bình thường" thì ablation mới đồng bộ.
        # Hai module _ff giữ lại trong __init__ cho tương thích checkpoint, nhưng không dùng.
        text_mix = self.ocr_lite_text_ln((tok_emb + char_tok).to(torch.float32)).to(self.target_dtype)
        text_mix = self.ocr_lite_text_proj(text_mix.to(torch.float32)).to(self.target_dtype)

        box_emb = self.ocr_lite_box_proj(box_tok.to(torch.float32)).to(self.target_dtype)

        det_emb = self.ocr_lite_det_ln(
            self.ocr_lite_det_proj(det_tok.to(torch.float32))
        ).to(self.target_dtype) if _has_det else 0.0
        rec_emb = self.ocr_lite_rec_ln(
            self.ocr_lite_rec_proj(rec_tok.to(torch.float32))
        ).to(self.target_dtype) if _has_rec else 0.0

        ocr_tok_mask = token_mask.long()
        ocr_fused_feat = (text_mix + box_emb + det_emb + rec_emb).to(self.target_dtype)
        ocr_fused_feat = ocr_fused_feat * ocr_tok_mask.to(self.target_dtype).unsqueeze(-1)

        return ocr_fused_feat, ocr_tok_mask

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
        twc_group_ids: Optional[torch.LongTensor] = None,
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
            need_attn_map=use_vs,       # chỉ AVF cần patch_scores
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
            # TAT AVF = BO HAN MODULE: khong crop theo attention, va ConvNeXt KHONG chay.
            # Truoc day nhanh nay van cho ConvNeXt chay tren toan anh, tuc la THAY THE
            # module chu khong bo — nen "tat AVF" van them 49 token thi giac tu mot
            # backbone thu hai, va phep so sanh ON/OFF do luong nham (ON con MAT goc nhin
            # toan cuc so voi OFF). Gio OFF khong dong gop token nao, dung nhu QA-ViT OFF
            # cho text_emb rong.
            #
            # attn_summary VAN GIU: no la trung binh cong cua chinh img_tokens (CLIP),
            # khong lien quan crop hay ConvNeXt, va giong het nhau o ca hai nhanh — nen no
            # thuoc kien truc nen, khong thuoc AVF.
            vs_out = {}
            crop_tokens = torch.zeros(B, 0, D, device=device, dtype=self.target_dtype)
            crop_mask = torch.zeros(B, 0, device=device, dtype=torch.long)
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
            # Baseline: hợp nhất TUYẾN TÍNH thay cho Constituent/Group/SpatialCircle.
            # Cũng ở TOKEN-LEVEL (L_tok) như đường ON → ablation chỉ đổi phần xử lý,
            # KHÔNG đổi độ dài chuỗi (trước đây baseline trả word-level nên ngắn hơn ~1.8x).
            ocr_fused_feat, mask_ocr = self._encode_ocr_baseline_features(
                ocr_info, word_ids_for_ocr, ocr_map, twa_ocr_char,
                twa_ocr_char_mask, token_mask_for_ocr, ocr_box_mask_for_ocr, device
            )
            token_mask_for_ocr = mask_ocr

        # Cả hai đường giờ đều token-level → pad/crop về L_tok cho cả hai (thường là no-op).
        L_tok = word_ids_for_ocr.size(1)
        if ocr_fused_feat.size(1) != L_tok:
            ocr_fused_feat = _pad_or_crop_lastdim(ocr_fused_feat, L_tok, pad_value=0.0)
        if token_mask_for_ocr.size(1) != L_tok:
            token_mask_for_ocr = _pad_or_crop_lastdim_int(token_mask_for_ocr, L_tok, pad_value=0)

        # attn_summary, crop_tokens, crop_mask were already assigned in the use_vs block above

        # Per-component non-finite diagnostic (NaN AND inf) — names the exact culprit
        # feeding fused_seq + its magnitude, so the pretrain guard below is never a guess.
        # FWD_DIAG=1: mở kiểm tra này cho CẢ finetune (chỉ in log, không đổi tính toán) —
        # cần để biết thành phần nào của fused_seq sinh NaN khi enc_out báo NaN.
        if getattr(self, "_pretrain_stage", False) or os.environ.get("FWD_DIAG") == "1":
            # pixel_values / txt_hidden_states = ĐẦU VÀO THỰC của QA-CLIP (txt_emb ở dưới
            # là txt_emb_for_enc — tensor KHÁC, nên trước đây không lộ ra thủ phạm).
            if os.environ.get("FWD_DIAG") == "1":
                _bad_w = [n for n, p in self.qa_clip.named_parameters() if not torch.isfinite(p).all()]
                if _bad_w:
                    print(f"🚨 [FWD_DIAG] qa_clip có {len(_bad_w)} TRỌNG SỐ non-finite, vd: {_bad_w[:3]}")
            for _cn, _ct in (("pixel_values", pixel_values_dev),
                             ("txt_hidden_states(->qa_clip)", txt_emb_for_clip),
                             ("txt_emb", txt_emb_for_enc), ("img_tokens", img_pack["img_tokens"]),
                             ("ocr_fused_feat", ocr_fused_feat), ("crop_tokens", crop_tokens),
                             ("attn_summary", attn_summary)):
                if not torch.isfinite(_ct).all():
                    _f = _ct.detach().float()
                    _n_nan = int(torch.isnan(_f).sum()); _n_inf = int(torch.isinf(_f).sum())
                    _fin = _f[torch.isfinite(_f)]
                    _rng = (f"finite[min={_fin.min():.3g},max={_fin.max():.3g}]" if _fin.numel() else "all-nonfinite")
                    print(f"🚨 [FORWARD CHECK] {_cn} non-finite: nan={_n_nan} inf={_n_inf} {_rng}")

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

        # PRETRAIN-ONLY numerical guard (giữ nguyên như bản gốc): QA-CLIP vision có thể
        # phát giá trị non-finite lan vào enc_out. Gate bằng `_pretrain_stage` (chỉ set ở
        # pretrain.py) nên FINETUNE giữ nguyên hành vi notebook.
        if getattr(self, "_pretrain_stage", False) and not torch.isfinite(fused_seq).all():
            print("⚠️ [pretrain guard] non-finite in fused_seq (QA-CLIP vision) → nan_to_num")
            fused_seq = torch.nan_to_num(fused_seq, nan=0.0, posinf=1e4, neginf=-1e4)

        # Notebook gốc / TWA paper: dùng mask 1D chuẩn — original và related token
        # được encode cùng nhau với full attention (KHÔNG chặn cross-half).
        enc_out = self.vit5.encoder(inputs_embeds=fused_seq, attention_mask=fused_mask, return_dict=True)
        if torch.isnan(enc_out.last_hidden_state).any(): print("🚨 [FORWARD CHECK] enc_out.last_hidden_state has NaN")

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
            if torch.isnan(outputs.logits).any(): print("🚨 [FORWARD CHECK] vit5 output logits has NaN")
            out_dict["textcls_scores"] = outputs.logits
            # Encoder-head MLM is dropped in cloze mode: cmb_text_mask_label is all -100,
            # so the T5 loss averages over 0 valid tokens → NaN. mlm_loss is NOT used in
            # the cloze total, so sanitize it to an in-graph 0 (avoids the NaN detector /
            # any downstream propagation). Legacy MLM modes keep the real finite loss.
            _mlm_l = outputs.loss
            if _mlm_l is None or not torch.isfinite(_mlm_l):
                _mlm_l = enc_out.last_hidden_state.sum() * 0.0
            out_dict["mlm_loss"] = _mlm_l

            dec_last = outputs.decoder_hidden_states[-1]
            dec_first = dec_last[:, 0, :]
            out_dict["pollutecls_scores"] = self.pollute_head(dec_first.float())

            # ITC vectors: masked-mean pool image tokens + text side → project → L2.
            # (When vision is unfrozen, img_tokens carries grad → ITC shapes CLIP.)
            _im = img_pack["img_attn_mask"].unsqueeze(-1).to(img_pack["img_tokens"].dtype)
            _img_pooled = (img_pack["img_tokens"] * _im).sum(1) / _im.sum(1).clamp_min(1e-6)
            if getattr(self, "_itc_text_source", "question") == "ocr":
                # v4: text side = CLEAN OCR string (bag of static embeddings of the
                # original-half OCR word tokens). WHY: questions are generic templates
                # ("cửa hàng này tên gì?") with ~zero image-specific info — image↔question
                # mutual information is so low that vs a 4096-queue the optimizer's best
                # move was collapsing logit_scale → eval loss pinned at ln(4) (measured,
                # v3 full run, epoch 2.9). The OCR string UNIQUELY identifies the image,
                # so image↔OCR is learnable at any queue size AND trains the unfrozen
                # CLIP to encode scene text — the frozen-vision root cause. No leakage:
                # the image side never sees the OCR string (QA-CLIP = pixels + question
                # instruction); a bag-of-embeddings is appropriate for an unordered OCR
                # word set and costs one lookup (no extra forward). Pretrain-only attr
                # set by pretrain.py; finetune never reaches this block.
                _half_w = twa_ocr_char.size(1) // 2 if o2r_labels is not None else twa_ocr_char.size(1)
                _pad_tok = int(self.config.pad_token_id or 0)
                _tok_ok = ((ocr_map >= 0) & (ocr_map < _half_w)
                           & (word_ids_for_ocr != _pad_tok)).unsqueeze(-1)
                # Per-sample fallback: ảnh không có từ OCR sạch nào (mask rỗng) thì dùng
                # mọi token non-pad — KHÔNG BAO GIỜ pool ra vector 0 (vector 0 → mọi
                # text vec identical → softmax uniform → loss ≡ ln(N), grad ≡ 0).
                _cnt = _tok_ok.sum(1)  # (B, 1)
                if bool((_cnt == 0).any()):
                    _fb = (word_ids_for_ocr != _pad_tok).unsqueeze(-1)
                    _tok_ok = torch.where((_cnt == 0).view(-1, 1, 1), _fb, _tok_ok)
                _ocr_emb = self.vit5.get_input_embeddings()(word_ids_for_ocr).to(dtype=self.target_dtype)
                _tm = _tok_ok.to(_ocr_emb.dtype)
                _txt_pooled = (_ocr_emb * _tm).sum(1) / _tm.sum(1).clamp_min(1e-6)
                # diag (verify đọc): số token sạch được pool + norm vector text
                out_dict["itc_dbg_tokok"] = _tok_ok.detach().sum(1).squeeze(-1)
                out_dict["itc_dbg_txt_norm"] = _txt_pooled.detach().norm(dim=-1)
            else:
                _tm = txt_attn_mask_for_enc.unsqueeze(-1).to(txt_emb_for_enc.dtype)
                if getattr(self, "_itc_text_pool", "embed") == "encoder":
                    # v3: pool the CONTEXTUAL question encoding (vit5 encoder output —
                    # already computed above for QA-CLIP, zero extra cost). A real
                    # sentence vector instead of a bag of static embeddings; ITC then
                    # also shapes the text ENCODER (ALBEF-style align-before-fuse).
                    _txt_src = txt_hidden_states
                else:
                    _txt_src = txt_emb_for_enc
                _txt_pooled = (_txt_src * _tm).sum(1) / _tm.sum(1).clamp_min(1e-6)
            _iv = self.itc_img_proj(_img_pooled.float())
            _tv = self.itc_txt_proj(_txt_pooled.float())
            out_dict["itc_img_vec"] = _iv / _iv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out_dict["itc_txt_vec"] = _tv / _tv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out_dict["itc_logit_scale"] = self.itc_logit_scale

            out_dict["contrastive_scores"] = None
            out_dict["o2r_block"] = None
            out_dict["r2o_block"] = None

            use_twc = bool(getattr(self.config, "use_twc", getattr(self, "use_twc", True)))

            # ── TWC: Tách biệt OCR-gốc và OCR-related, encode qua projection heads riêng ──
            # TWA paper (Fig 3): OCR_original và OCR_related được so sánh trong không gian
            # contrastive riêng biệt. Projection heads ngăn model dùng thông tin shared
            # context để trivially match cặp (i,i) — phải học biểu diễn thực sự.
            if use_twc and use_ocr and o2r_labels is not None:
                enc_hid = enc_out.last_hidden_state
                L_txt = txt_emb_for_enc.size(1)
                L_img = img_pack["img_tokens"].size(1)
                L_ocr = ocr_fused_feat.size(1)

                off_img = L_txt
                off_ocr = off_img + L_img  # Vị trí bắt đầu chính xác của ocr_fused_feat

                # Cắt đúng tensor encoder output tương ứng với chuỗi OCR
                ocr_enc_out = enc_hid[:, off_ocr: off_ocr + L_ocr, :]

                valid_map_mask = (ocr_map >= 0) & (token_mask_for_ocr > 0)

                if valid_map_mask.any():
                    Bc, _, D0 = ocr_enc_out.size()
                    N_word = int(twa_ocr_char.size(1))

                    if N_word % 2 != 0:
                        raise RuntimeError(
                            f"TWC requires OCR original + OCR augmented pairs, "
                            f"but N_word={N_word} is odd."
                        )

                    half = N_word // 2  # first half = original, second half = related

                    # Pool token-level encoder outputs về word-level
                    word_vectors = torch.zeros(Bc, N_word, D0, device=device, dtype=ocr_enc_out.dtype)
                    word_counts  = torch.zeros(Bc, N_word, device=device, dtype=ocr_enc_out.dtype)

                    for b in range(Bc):
                        vmask_b = valid_map_mask[b]
                        if not vmask_b.any():
                            continue
                        src = ocr_enc_out[b][vmask_b]
                        idx = ocr_map[b][vmask_b].clamp(0, max(N_word - 1, 0))
                        word_vectors[b].scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
                        ones = torch.ones_like(idx, dtype=word_counts.dtype, device=device)
                        word_counts[b].scatter_add_(0, idx, ones)

                    word_counts  = word_counts.unsqueeze(-1).clamp_min(1e-9)
                    word_vectors = word_vectors / word_counts
                    # Guard: encoder can produce inf/NaN (e.g. from weight overflow).
                    # Must clean HERE before the division creates NaN-gradient paths.
                    word_vectors = torch.nan_to_num(word_vectors, nan=0.0, posinf=0.0, neginf=0.0)

                    # Tách thành 2 nhánh: original và related
                    ocr_word_feat = word_vectors[:, :half, :]   # [B, half, D]
                    rel_word_feat = word_vectors[:, half:, :]   # [B, half, D]

                    # Notebook gốc / TWA paper: L2-normalize word-feature trực tiếp
                    # từ encoder rồi tính cosine similarity (KHÔNG qua projection head).
                    # nan_to_num BEFORE norm: nếu x có inf, norm(x)=inf → backward
                    # dL/dx = dL/dz*(1/inf) = 0*inf = NaN (IEEE 754). Làm sạch x trước
                    # để chặn đường gradient NaN này.
                    def _safe_l2(x, dim=-1, eps=1e-6):
                        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                        n = x.norm(dim=dim, keepdim=True).clamp_min(eps)
                        return x / n

                    ocr_feat = _safe_l2(ocr_word_feat, dim=-1)  # [B, half, D]
                    rel_feat = _safe_l2(rel_word_feat, dim=-1)  # [B, half, D]

                    Bn, W, Dn = ocr_feat.shape
                    ocr_flat = ocr_feat.reshape(Bn * W, Dn)  # [B*half, D]
                    rel_flat = rel_feat.reshape(Bn * W, Dn)  # [B*half, D]

                    # Logit scale (learnable temperature τ, theo CLIP/TWA Eq.4), clamp
                    # để tránh divergence.
                    logit_scale = self.logit_scale.clamp(
                        min=np.log(1 / 100), max=np.log(100)
                    ).exp()

                    # [B*half, B*half] — ma trận similarity giữa tất cả
                    # original tokens và tất cả related tokens trong batch
                    contrastive_scores = logit_scale * (ocr_flat @ rel_flat.t())
                    contrastive_scores = torch.nan_to_num(
                        contrastive_scores, nan=0.0, posinf=1e4, neginf=-1e4
                    )

                    out_dict["contrastive_scores"] = contrastive_scores

                    # Label matrix: [B, half, half] → block_diag [B*half, B*half]
                    B_lab, _, _ = o2r_labels.shape
                    blocks = [o2r_labels[b].to(device) for b in range(B_lab)]
                    out_dict["o2r_block"] = torch.block_diag(*blocks)

                    if r2o_labels is not None:
                        blocks_r = [r2o_labels[b].to(device) for b in range(B_lab)]
                        out_dict["r2o_block"] = torch.block_diag(*blocks_r)
                    else:
                        out_dict["r2o_block"] = None

                    # ── Cross-sample SAME-IMAGE SAME-TEXT inheritance ──────────────
                    # TWA §3.4: OCR tokens with the same text + their augmented forms
                    # become positive/semi-positive samples "in pairs". Because we let
                    # augmentation vary per sample (not TWA's fixed seed), the same
                    # image's "cash" may be KEEP (1.0) in one sample and NOISE (0.9) in
                    # another. block_diag would wrongly mark these cross-sample pairs as
                    # negative. So for cross-sample pairs sharing (OCR image, text)
                    # — identified by twc_group_ids — re-inherit the partner's diagonal
                    # label, exactly like the within-sample same-text rule in the collator.
                    if twc_group_ids is not None:
                        _g = twc_group_ids.to(device).reshape(-1)          # [M], -1 = PAD/ignore
                        _o2rb = out_dict["o2r_block"]
                        _M = _o2rb.size(0)
                        if _g.numel() == _M:
                            _samp = torch.arange(_M, device=device) // half
                            _inherit = (
                                (_g.unsqueeze(1) == _g.unsqueeze(0))       # same (image, text) group
                                & (_g.unsqueeze(1) >= 0)                   # exclude PAD
                                & (_samp.unsqueeze(1) != _samp.unsqueeze(0))  # cross-sample only
                            )
                            if _inherit.any():
                                # o2r_block[p, q] <- o2r_block[q, q]  (partner's diagonal)
                                _diag_o = torch.diagonal(_o2rb).clone()
                                out_dict["o2r_block"] = torch.where(
                                    _inherit, _diag_o.unsqueeze(0).expand(_M, _M), _o2rb)
                                if out_dict["r2o_block"] is not None:
                                    _r2ob = out_dict["r2o_block"]
                                    _diag_r = torch.diagonal(_r2ob).clone()
                                    out_dict["r2o_block"] = torch.where(
                                        _inherit, _diag_r.unsqueeze(0).expand(_M, _M), _r2ob)

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