"""
models/modules/qa_clip.py
QACLIPEncoder — CLIP with instruction-guided late-fusion encoder.
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import (
    CLIPEncoderLayer, CLIPAttention, CLIPMLP, CLIPVisionEmbeddings, CLIPPreTrainedModel
)

def FeedForward(in_dim, out_dim, inner_dim=None):
    if inner_dim is None:
        inner_dim = out_dim
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, out_dim, bias=False),
    )

class MMCLIPAttention(CLIPAttention):
    def __init__(self, config):
        super().__init__(config)
        self.instruction_out_proj = torch.nn.Linear(self.out_proj.in_features, self.out_proj.out_features)
        self.instruction_proj_gate = nn.Parameter(torch.Tensor([0.]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        kv_states: torch.Tensor = None,
        kv_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # TRUE RESIDUAL GATING. Thiết kế cũ nối [question; visual] vào MỘT self-attention
        # rồi `out_proj(attn)` — nhánh out_proj này KHÔNG bị gate mà attn đã trộn value của
        # question, nên gate=0 KHÔNG đưa về CLIP thuần (đo được: đổi đặc trưng ảnh 7.3% dù
        # β=0). Trên bài OCR-chi-phối, nhiễu bắt buộc đó làm +qaclip THẤP hơn baseline.
        #
        # Sửa: tách hẳn hai đường, toàn bộ ảnh hưởng câu hỏi nằm SAU gate.
        #   base  = out_proj(SelfAttn(visual, visual))            # ĐÚNG CLIP thuần
        #   delta = instruction_out_proj(CrossAttn(visual→question))
        #   out   = base + tanh(β)·delta
        # β=0 ⇒ out = base = CLIP thuần từng số ⇒ thêm qaclip KHÔNG BAO GIỜ tệ hơn baseline;
        # model chỉ mở gate ở nơi câu hỏi thật sự giúp.
        if kv_states is None:
            raise ValueError("kv_states required")
        bsz, vis_len, embed_dim = hidden_states.size()
        mm_len = int(kv_states.shape[1])
        H, Dh = self.num_heads, self.head_dim
        ps = (bsz * H, -1, Dh)

        # visual query dùng chung (giữ scaling của CLIP)
        q = self._shape(self.q_proj(hidden_states) * self.scale, vis_len, bsz).view(*ps)

        # ---- base: self-attention CHỈ trên visual = đúng một lớp CLIP ----
        kv = self._shape(self.k_proj(hidden_states), -1, bsz).view(*ps)
        vv = self._shape(self.v_proj(hidden_states), -1, bsz).view(*ps)
        base_w = torch.bmm(q.float(), kv.float().transpose(1, 2))
        base_w = base_w - base_w.amax(dim=-1, keepdim=True)
        base_w = F.softmax(base_w, dim=-1)
        base_ctx = torch.bmm(F.dropout(base_w, p=self.dropout, training=self.training), vv.float())
        base_ctx = base_ctx.to(hidden_states.dtype).view(bsz, H, vis_len, Dh).transpose(1, 2).reshape(bsz, vis_len, embed_dim)
        base_out = self.out_proj(base_ctx)

        if mm_len == 0:
            # qaclip TẮT → đúng CLIP thuần, không có nhánh câu hỏi.
            attn_ret = base_w.view(bsz, H, vis_len, vis_len) if output_attentions else None
            return base_out, attn_ret

        # ---- delta: cross-attention visual → question (hoàn toàn sau gate) ----
        kq = self._shape(self.k_proj(kv_states), -1, bsz).view(*ps)
        vq = self._shape(self.v_proj(kv_states), -1, bsz).view(*ps)
        cross_w = torch.bmm(q.float(), kq.float().transpose(1, 2))   # (B*H, vis, mm)
        if kv_masks is not None:
            m = kv_masks.to(device=hidden_states.device)
            if m.size(1) != mm_len:
                m = m[:, :mm_len] if m.size(1) > mm_len else torch.cat(
                    [m, torch.zeros(bsz, mm_len - m.size(1), device=m.device, dtype=m.dtype)], dim=1)
            km = m.to(torch.bool)[:, None, None, :].expand(bsz, H, vis_len, mm_len).reshape(bsz * H, vis_len, mm_len)
            # -1e9 (không phải finfo.min) để trừ-max không tràn -inf → không sinh NaN
            # kể cả khi một hàng bị mask hết (question luôn có ≥1 token thật nên hiếm).
            cross_w = cross_w.masked_fill(~km, -1e9)
        cross_w = cross_w - cross_w.amax(dim=-1, keepdim=True)
        cross_w = F.softmax(cross_w, dim=-1)
        cross_w = torch.nan_to_num(cross_w, nan=0.0)
        cross_ctx = torch.bmm(F.dropout(cross_w, p=self.dropout, training=self.training), vq.float())
        cross_ctx = cross_ctx.to(hidden_states.dtype).view(bsz, H, vis_len, Dh).transpose(1, 2).reshape(bsz, vis_len, embed_dim)
        delta = self.instruction_out_proj(cross_ctx)

        out = base_out + torch.tanh(self.instruction_proj_gate) * delta
        # Trả về attention question-guided (visual × question) để làm patch_scores cho AVF.
        attn_ret = cross_w.view(bsz, H, vis_len, mm_len) if output_attentions else None
        return out, attn_ret

class MMCLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MMCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.instruct_dim_reduce = FeedForward(config.instruction_dim, config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        instruct_states: torch.Tensor = None,
        instruct_masks: torch.Tensor = None,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            kv_states=self.instruct_dim_reduce(instruct_states) if (self.instruct_dim_reduce and instruct_states is not None) else instruct_states,
            kv_masks=instruct_masks
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class MultiPathAlignModule(nn.Module):
    """Features combination module at the final stage of ViT (LLaVA-HR / MRA).
    Combines fast (ViT) and slow (ConvNeXt) features before feeding to projector/LLM.
    """
    def __init__(self, fast_vision_dim: int, slow_vision_dim: int, zero_init: bool = True):
        super().__init__()
        self.fast_vision_dim = int(fast_vision_dim)
        self.slow_vision_dim = int(slow_vision_dim)
        self.fast_proj = nn.Linear(self.fast_vision_dim, self.fast_vision_dim)
        self.slow_proj = nn.Linear(self.slow_vision_dim, self.fast_vision_dim)
        self.init_weights(zero_init=zero_init)

    def init_weights(self, zero_init: bool = True):
        if zero_init:
            nn.init.eye_(self.fast_proj.weight)
            nn.init.zeros_(self.fast_proj.bias)
            nn.init.zeros_(self.slow_proj.weight)
            nn.init.zeros_(self.slow_proj.bias)
        else:
            nn.init.xavier_uniform_(self.fast_proj.weight)
            nn.init.zeros_(self.fast_proj.bias)
            nn.init.xavier_uniform_(self.slow_proj.weight)
            nn.init.zeros_(self.slow_proj.bias)

    def forward(self, fast_feat: torch.Tensor, slow_feat: torch.Tensor) -> torch.Tensor:
        # Align device and dtype
        slow_feat = slow_feat.to(device=fast_feat.device, dtype=fast_feat.dtype)

        # 4D input handling: [B, C, H, W] or [B, H, W, C]
        if slow_feat.ndim == 4:
            if slow_feat.shape[1] == self.slow_vision_dim:
                b, c, h, w = slow_feat.shape
                slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)
            elif slow_feat.shape[-1] == self.slow_vision_dim:
                b, h, w, c = slow_feat.shape
                slow_feat = slow_feat.view(b, -1, c)
            else:
                b, c, h, w = slow_feat.shape
                slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)
        elif slow_feat.ndim == 3 and slow_feat.shape[-1] != self.slow_vision_dim and slow_feat.shape[1] == self.slow_vision_dim:
            slow_feat = slow_feat.transpose(1, 2)

        # Spatial alignment between slow_feat and fast_feat sequence lengths
        if slow_feat.shape[1] < fast_feat.shape[1]:
            b, l, c = slow_feat.shape
            src_size = int(math.isqrt(l))
            dst_size = int(math.isqrt(fast_feat.shape[1]))
            slow_feat = slow_feat.transpose(1, 2).view(b, c, src_size, src_size)
            slow_feat = F.interpolate(slow_feat.float(), size=(dst_size, dst_size), mode='bilinear',
                                      align_corners=True).to(dtype=fast_feat.dtype)
            slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)
        elif slow_feat.shape[1] > fast_feat.shape[1]:
            b, l, c = slow_feat.shape
            src_size = int(math.isqrt(l))
            dst_size = int(math.isqrt(fast_feat.shape[1]))
            slow_feat = slow_feat.transpose(1, 2).view(b, c, src_size, src_size)
            if src_size % dst_size == 0:
                stride = src_size // dst_size
                slow_feat = F.avg_pool2d(slow_feat, stride, stride)
            else:
                slow_feat = F.adaptive_avg_pool2d(slow_feat, (dst_size, dst_size))
            slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)

        return self.fast_proj(fast_feat) + self.slow_proj(slow_feat)


class S2FStitchAlignModuleV2(nn.Module):
    """Mixture-of-Resolution Adapter (MR-Adapter, Luo et al., ICLR 2025 / LLaVA-HR).
    Official implementation from authors' repository (S2FStitchAlignModuleV2).

    Slow-to-Fast (S2F) alignment: injects fine-grained high-resolution ConvNeXt (slow branch)
    representations into low-resolution ViT (fast branch) tokens at stage boundaries:
        F'_vl = F_vl + fast_proj(GELU(fast_conv(F_vl))) + slow_feat_align * gate.tanh()
    where:
        - fast_conv is 7x7 depthwise conv (ConvNeXt-style receptive field)
        - fast_proj is 1x1 conv
        - slow_conv is 1x1 conv
        - slow_proj is 1x1 conv
        - gate is dynamic MLP mapping pooled [fast; slow_align] -> d (channel-wise gating bounded in [-1, 1])
        - zero_init: zero-initializes projection layers for exact identity mapping at step 0 (ReZero-style)
    """
    def __init__(
        self,
        fast_vision_dim: Optional[int] = None,
        slow_vision_dim: Optional[int] = None,
        zero_init: bool = True,
        d_vit: Optional[int] = None,
        d_cnn: Optional[int] = None,
        grid: int = 14,
        kernel_size: Optional[int] = None,
    ):
        super().__init__()
        fast_dim = int(fast_vision_dim if fast_vision_dim is not None else d_vit)
        slow_dim = int(slow_vision_dim if slow_vision_dim is not None else d_cnn)
        self.fast_vision_dim = fast_dim
        self.slow_vision_dim = slow_dim
        self.grid = int(grid)

        # Receptive field scaling for ViT grid:
        # - Grid 14x14 (ViT 224): 3x3 kernel (receptive field 3/14 ~ 21.4%) matches LLaVA-HR's 7x7 on 32x32 (7/32 ~ 21.9%).
        # - Grid 21x21 / 24x24 (ViT 336/384): 3x3 or 5x5 kernel (3/21 ~ 14.3%, 5/24 ~ 20.8%).
        # - Grid >= 28 (e.g. 32x32 in LLaVA-HR 1024): 7x7 kernel (~21.9%).
        # groups=fast_dim ensures DEPTHWISE conv: each channel dimension has its own independent filter,
        # preventing channels from mixing in spatial convolution. Channel mixing is handled by 1x1 fast_proj.
        if kernel_size is not None:
            k = int(kernel_size)
        elif self.grid <= 16:
            k = 3
        elif self.grid < 28:
            k = 3
        else:
            k = 7
        padding = k // 2
        self.kernel_size = k
        self.padding = padding

        # Slow (high-res CNN) branch: 1x1 conv -> GELU -> 1x1 conv
        self.slow_conv = nn.Conv2d(slow_dim, slow_dim, 1)
        self.slow_proj = nn.Conv2d(slow_dim, fast_dim, 1)

        # Fast (low-res ViT) branch: Depthwise Conv (groups=fast_dim) + 1x1 pointwise conv
        self.fast_conv = nn.Conv2d(fast_dim, fast_dim, k, padding=padding, groups=fast_dim)
        self.fast_proj = nn.Conv2d(fast_dim, fast_dim, 1)

        # Dynamic Channel-wise Gating: MLP mapping pooled [fast; slow] to feature dimension d
        self.gate = nn.Sequential(
            nn.Linear(fast_dim * 2, fast_dim // 2),
            nn.GELU(),
            nn.Linear(fast_dim // 2, fast_dim)
        )

        # Weight initialization matching author's repo
        nn.init.xavier_uniform_(self.slow_conv.weight)
        nn.init.xavier_uniform_(self.fast_conv.weight)
        nn.init.zeros_(self.slow_conv.bias)
        nn.init.zeros_(self.fast_conv.bias)
        if zero_init:
            nn.init.zeros_(self.slow_proj.weight)
            nn.init.zeros_(self.fast_proj.weight)
        else:
            nn.init.xavier_uniform_(self.slow_proj.weight)
            nn.init.xavier_uniform_(self.fast_proj.weight)
        nn.init.zeros_(self.slow_proj.bias)
        nn.init.zeros_(self.fast_proj.bias)

    def src2dst_align(self, src_feat: torch.Tensor, dst_feat: torch.Tensor) -> Tuple[torch.Tensor, int]:
        dst_size = int(math.isqrt(dst_feat.shape[1]))
        if src_feat.shape[1] == dst_feat.shape[1]:
            return src_feat, dst_size
        b, l, c = src_feat.shape
        src_size = int(math.isqrt(l))
        src_feat = src_feat.transpose(1, 2).view(b, c, src_size, src_size)
        if src_size < dst_size:
            # upsample
            src_feat = F.interpolate(
                src_feat.float(), size=(dst_size, dst_size), mode='bilinear', align_corners=True
            ).to(dtype=src_feat.dtype)
        elif src_size > dst_size:
            # pooling
            if src_size % dst_size == 0:
                stride = src_size // dst_size
                src_feat = F.avg_pool2d(src_feat, stride, stride)
            else:
                src_feat = F.adaptive_avg_pool2d(src_feat, (dst_size, dst_size))
        src_feat = src_feat.view(b, c, -1).transpose(1, 2)
        return src_feat, dst_size

    def forward(self, fast_feat: torch.Tensor, slow_feat: torch.Tensor) -> torch.Tensor:
        # Support both 4D [B, C, H, W] and 3D [B, L, C] input for slow_feat
        if slow_feat.ndim == 3:
            b, l, c = slow_feat.shape
            src_size = int(math.isqrt(l))
            slow_feat = slow_feat.transpose(1, 2).view(b, c, src_size, src_size)

        b, c, h, w = slow_feat.shape
        _, _, d = fast_feat.shape

        # High-res branch processing
        slow_feat = self.slow_proj(F.gelu(self.slow_conv(slow_feat)))
        slow_feat = slow_feat.view(b, d, -1).transpose(1, 2)
        slow_feat_align, dst_size = self.src2dst_align(slow_feat, fast_feat)

        # Low-res branch processing (7x7 depthwise + 1x1 pointwise with residual)
        fast_feat_2d = fast_feat.transpose(1, 2).view(b, d, dst_size, dst_size)
        fast_feat_2d = fast_feat_2d + self.fast_proj(F.gelu(self.fast_conv(fast_feat_2d)))
        fast_feat = fast_feat_2d.view(b, d, dst_size * dst_size).transpose(1, 2)

        # Dynamic soft channel-wise gating across feature dimension d
        pooled = torch.cat([fast_feat, slow_feat_align], dim=-1).mean(dim=1)  # [B, 2*d]
        gate = self.gate(pooled).unsqueeze(1)  # [B, 1, d]

        # Fusion: channel-wise modulated residual injection
        fast_feat = fast_feat + slow_feat_align * gate.tanh()
        return fast_feat


MRAdapter = S2FStitchAlignModuleV2


class InstructCLIPEncoder(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        modules_list = []
        for layer_id in range(config.num_hidden_layers):
            if config.integration_point == 'late':
                layer = CLIPEncoderLayer if layer_id < (config.num_hidden_layers // 2) else MMCLIPEncoderLayer
            else:
                raise ValueError("unsupported integration_point")
            modules_list.append(layer(config))
        self.layers = nn.ModuleList(modules_list)
        self.gradient_checkpointing = False
        # MRA (Mixture-of-Resolution). Mac dinh tat; OpenViVQAModel gan vao khi bat.
        self.mra_adapters = None          # nn.ModuleList cac MRAdapter
        self.mra_layer_ids = []           # index tang ViT sau do se bom (vd [5,8,11])
        self._mra_hi = None               # list dac trung ConvNeXt [B,N,d_cnn], dat moi forward

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        instruct_states: torch.Tensor = None,
        instruct_masks: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for _li, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward
                if isinstance(encoder_layer, CLIPEncoderLayer):
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        instruct_states=instruct_states,
                        instruct_masks=instruct_masks,
                    )
            else:
                if isinstance(encoder_layer, CLIPEncoderLayer):
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        output_attentions=output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        output_attentions=output_attentions,
                        instruct_states=instruct_states,
                        instruct_masks=instruct_masks,
                    )
            hidden_states = layer_outputs[0]
            # --- MRA: bom high-res sau tang nay neu duoc cau hinh ---
            if (self.mra_adapters is not None and self._mra_hi is not None
                    and _li in self.mra_layer_ids):
                _k = self.mra_layer_ids.index(_li)
                # ĐÚNG paper: MỘT F_vh (đặc trưng CUỐI của CNN) dùng chung cho mọi stage bơm.
                _hi = self._mra_hi                           # [B, N, d_final]
                _cls = hidden_states[:, :1, :]               # token CLS giu nguyen
                _pat = hidden_states[:, 1:, :]               # [B, N, d_vit] = 14x14
                _pat = self.mra_adapters[_k](_pat, _hi.to(_pat.dtype))
                hidden_states = torch.cat([_cls, _pat], dim=1)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = InstructCLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        instruct_states: torch.Tensor = None,
        instruct_masks: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            instruct_states=instruct_states,
            instruct_masks=instruct_masks,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def convert_timm_vit_to_clip_state_dict(timm_sd: dict, prefix: str = "vision_model.") -> dict:
    """Converts a timm ViT state dict (e.g. vit_base_patch16_clip_384.laion2b_ft_in1k)
    into a Hugging Face CLIPVisionTransformer state dict.

    Mappings:
      - cls_token [1, 1, 768] -> embeddings.class_embedding [768]
      - patch_embed.proj.weight [768, 3, 16, 16] -> embeddings.patch_embedding.weight
      - pos_embed [1, 577, 768] -> embeddings.position_embedding.weight [577, 768]
      - norm_pre.weight/bias -> pre_layrnorm.weight/bias
      - norm.weight/bias -> post_layernorm.weight/bias
      - blocks.{i}.norm1 -> encoder.layers.{i}.layer_norm1
      - blocks.{i}.norm2 -> encoder.layers.{i}.layer_norm2
      - blocks.{i}.mlp.fc1/fc2 -> encoder.layers.{i}.mlp.fc1/fc2
      - blocks.{i}.attn.proj -> encoder.layers.{i}.self_attn.out_proj
      - blocks.{i}.attn.qkv [2304, 768] -> self_attn.q_proj, k_proj, v_proj [768, 768] each
    """
    clip_sd = {}
    if 'cls_token' in timm_sd:
        clip_sd[f'{prefix}embeddings.class_embedding'] = timm_sd['cls_token'].squeeze()
    if 'patch_embed.proj.weight' in timm_sd:
        clip_sd[f'{prefix}embeddings.patch_embedding.weight'] = timm_sd['patch_embed.proj.weight']
    if 'pos_embed' in timm_sd:
        clip_sd[f'{prefix}embeddings.position_embedding.weight'] = timm_sd['pos_embed'].squeeze(0)
    if 'norm_pre.weight' in timm_sd:
        clip_sd[f'{prefix}pre_layrnorm.weight'] = timm_sd['norm_pre.weight']
    if 'norm_pre.bias' in timm_sd:
        clip_sd[f'{prefix}pre_layrnorm.bias'] = timm_sd['norm_pre.bias']
    if 'norm.weight' in timm_sd:
        clip_sd[f'{prefix}post_layernorm.weight'] = timm_sd['norm.weight']
    if 'norm.bias' in timm_sd:
        clip_sd[f'{prefix}post_layernorm.bias'] = timm_sd['norm.bias']

    num_blocks = 0
    while f'blocks.{num_blocks}.norm1.weight' in timm_sd or f'blocks.{num_blocks}.attn.qkv.weight' in timm_sd:
        num_blocks += 1

    for i in range(num_blocks):
        prefix_timm = f'blocks.{i}.'
        prefix_clip = f'{prefix}encoder.layers.{i}.'
        if f'{prefix_timm}norm1.weight' in timm_sd:
            clip_sd[f'{prefix_clip}layer_norm1.weight'] = timm_sd[f'{prefix_timm}norm1.weight']
        if f'{prefix_timm}norm1.bias' in timm_sd:
            clip_sd[f'{prefix_clip}layer_norm1.bias'] = timm_sd[f'{prefix_timm}norm1.bias']
        if f'{prefix_timm}norm2.weight' in timm_sd:
            clip_sd[f'{prefix_clip}layer_norm2.weight'] = timm_sd[f'{prefix_timm}norm2.weight']
        if f'{prefix_timm}norm2.bias' in timm_sd:
            clip_sd[f'{prefix_clip}layer_norm2.bias'] = timm_sd[f'{prefix_timm}norm2.bias']
        if f'{prefix_timm}mlp.fc1.weight' in timm_sd:
            clip_sd[f'{prefix_clip}mlp.fc1.weight'] = timm_sd[f'{prefix_timm}mlp.fc1.weight']
        if f'{prefix_timm}mlp.fc1.bias' in timm_sd:
            clip_sd[f'{prefix_clip}mlp.fc1.bias'] = timm_sd[f'{prefix_timm}mlp.fc1.bias']
        if f'{prefix_timm}mlp.fc2.weight' in timm_sd:
            clip_sd[f'{prefix_clip}mlp.fc2.weight'] = timm_sd[f'{prefix_timm}mlp.fc2.weight']
        if f'{prefix_timm}mlp.fc2.bias' in timm_sd:
            clip_sd[f'{prefix_clip}mlp.fc2.bias'] = timm_sd[f'{prefix_timm}mlp.fc2.bias']
        if f'{prefix_timm}attn.proj.weight' in timm_sd:
            clip_sd[f'{prefix_clip}self_attn.out_proj.weight'] = timm_sd[f'{prefix_timm}attn.proj.weight']
        if f'{prefix_timm}attn.proj.bias' in timm_sd:
            clip_sd[f'{prefix_clip}self_attn.out_proj.bias'] = timm_sd[f'{prefix_timm}attn.proj.bias']
        if f'{prefix_timm}attn.qkv.weight' in timm_sd:
            qkv_w = timm_sd[f'{prefix_timm}attn.qkv.weight']
            dim = qkv_w.shape[1]
            clip_sd[f'{prefix_clip}self_attn.q_proj.weight'] = qkv_w[:dim, :]
            clip_sd[f'{prefix_clip}self_attn.k_proj.weight'] = qkv_w[dim:2*dim, :]
            clip_sd[f'{prefix_clip}self_attn.v_proj.weight'] = qkv_w[2*dim:3*dim, :]
        if f'{prefix_timm}attn.qkv.bias' in timm_sd:
            qkv_b = timm_sd[f'{prefix_timm}attn.qkv.bias']
            dim = qkv_b.shape[0] // 3
            clip_sd[f'{prefix_clip}self_attn.q_proj.bias'] = qkv_b[:dim]
            clip_sd[f'{prefix_clip}self_attn.k_proj.bias'] = qkv_b[dim:2*dim]
            clip_sd[f'{prefix_clip}self_attn.v_proj.bias'] = qkv_b[2*dim:3*dim]

    return clip_sd


class QACLIPEncoder(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig, instruction_dim: int = 768, freeze_clip: bool = False, image_size: Optional[int] = None):
        super().__init__(config)
        self.config.instruction_dim = int(getattr(self.config, "instruction_dim", instruction_dim))
        self.config.integration_point = getattr(self.config, "integration_point", "late")
        self.config.freeze_clip = bool(getattr(self.config, "freeze_clip", freeze_clip))
        self.vision_model = CLIPVisionTransformer(self.config)
        tgt_sz = image_size if image_size is not None else getattr(self.config, "image_size", None)
        if tgt_sz is not None and int(tgt_sz) != int(self.vision_model.embeddings.image_size):
            self.interpolate_position_embedding(int(tgt_sz))
        self._apply_freeze()
        self.post_init()

    def interpolate_position_embedding(self, target_image_size: int):
        """Interpolate 2D position embeddings in CLIPVisionEmbeddings to match target_image_size (e.g. 224 -> 336).
        Preserves CLS token (index 0) and applies 2D bicubic interpolation to the patch tokens.
        """
        embeddings = self.vision_model.embeddings
        patch_size = int(embeddings.patch_size)
        old_grid = int(embeddings.image_size) // patch_size
        new_grid = int(target_image_size) // patch_size

        if old_grid == new_grid and embeddings.num_positions == (new_grid * new_grid + 1):
            return

        new_num_patches = new_grid * new_grid
        new_num_positions = new_num_patches + 1

        old_weight = embeddings.position_embedding.weight.data
        embed_dim = old_weight.shape[1]

        # If already matching target number of positions, just update config/metadata
        if old_weight.shape[0] == new_num_positions:
            embeddings.image_size = target_image_size
            embeddings.num_positions = new_num_positions
            embeddings.num_patches = new_num_patches
            self.vision_model.config.image_size = target_image_size
            self.config.image_size = target_image_size
            return

        old_num_patches = old_weight.shape[0] - 1
        actual_old_grid = int(math.isqrt(old_num_patches))

        cls_pos = old_weight[:1, :].unsqueeze(0)
        patch_pos = old_weight[1:, :].unsqueeze(0)

        patch_pos = patch_pos.transpose(1, 2).reshape(1, embed_dim, actual_old_grid, actual_old_grid)

        new_patch_pos = F.interpolate(
            patch_pos.float(),
            size=(new_grid, new_grid),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=old_weight.dtype)

        new_patch_pos = new_patch_pos.reshape(1, embed_dim, new_num_patches).transpose(1, 2)
        new_pos = torch.cat([cls_pos, new_patch_pos], dim=1).squeeze(0)

        new_position_embedding = nn.Embedding(new_num_positions, embed_dim)
        new_position_embedding.weight.data.copy_(new_pos)

        if bool(getattr(self.config, "freeze_clip", False)):
            new_position_embedding.weight.requires_grad = False

        embeddings.position_embedding = new_position_embedding
        embeddings.register_buffer("position_ids", torch.arange(new_num_positions).expand((1, -1)), persistent=False)
        embeddings.num_positions = new_num_positions
        embeddings.num_patches = new_num_patches
        embeddings.image_size = target_image_size
        self.vision_model.config.image_size = target_image_size
        self.config.image_size = target_image_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        instruction_dim = kwargs.pop("instruction_dim", None)
        integration_point = kwargs.pop("integration_point", None)
        freeze_clip = kwargs.pop("freeze_clip", None)
        target_image_size = kwargs.pop("image_size", None)

        pretrained_str = str(pretrained_model_name_or_path)
        is_timm = ("timm/" in pretrained_str) or ("vit_base_patch16_clip_384" in pretrained_str)

        if is_timm:
            import os
            from huggingface_hub import hf_hub_download
            import safetensors.torch

            target_sz = int(target_image_size) if target_image_size is not None else 384
            config = CLIPVisionConfig(
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                image_size=target_sz,
                patch_size=16,
                hidden_act="quick_gelu",
            )
            ins_dim = int(instruction_dim) if instruction_dim is not None else 768
            frz = bool(freeze_clip) if freeze_clip is not None else False
            config.instruction_dim = ins_dim
            config.integration_point = integration_point if integration_point is not None else "late"
            config.freeze_clip = frz

            model = cls(config, instruction_dim=ins_dim, freeze_clip=frz, image_size=target_sz)

            if os.path.isfile(pretrained_str):
                weights_path = pretrained_str
            else:
                try:
                    weights_path = hf_hub_download(repo_id=pretrained_str, filename="model.safetensors")
                except Exception:
                    weights_path = hf_hub_download(repo_id=pretrained_str, filename="pytorch_model.bin")

            if weights_path.endswith(".safetensors"):
                timm_sd = safetensors.torch.load_file(weights_path)
            else:
                timm_sd = torch.load(weights_path, map_location="cpu")

            converted_sd = convert_timm_vit_to_clip_state_dict(timm_sd, prefix="vision_model.")
            model.load_state_dict(converted_sd, strict=False)

            if target_sz != int(model.vision_model.embeddings.image_size):
                model.interpolate_position_embedding(target_sz)
            model._apply_freeze()
            return model

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if instruction_dim is not None:
            model.config.instruction_dim = int(instruction_dim)
        if integration_point is not None:
            model.config.integration_point = integration_point
        if freeze_clip is not None:
            model.config.freeze_clip = bool(freeze_clip)
        if target_image_size is not None and int(target_image_size) != int(model.vision_model.embeddings.image_size):
            model.interpolate_position_embedding(int(target_image_size))
        model._apply_freeze()
        return model

    def _apply_freeze(self):
        if not bool(getattr(self.config, "freeze_clip", False)):
            return
        for _, p in self.named_parameters():
            p.requires_grad = False
        for n, p in self.named_parameters():
            if ('instruct' in n) or ('instruction' in n):
                p.requires_grad = True

    def init_qavit_comps(self):
        with torch.no_grad():
            for layer in self.vision_model.encoder.layers:
                if isinstance(layer, MMCLIPEncoderLayer):
                    # BẮT BUỘC: instruct_dim_reduce dùng nn.Linear(bias=False). HF
                    # `from_pretrained` (_fast_init) cấp phát bằng torch.empty() rồi chỉ
                    # điền weight CÓ trong checkpoint; weight thiếu để cho _init_weights lo.
                    # Nhưng CLIPPreTrainedModel._init_weights với nn.Linear thường CHỈ zero
                    # bias NẾU có bias → các Linear bias=False này KHÔNG được khởi tạo, giữ
                    # nguyên BỘ NHỚ RÁC và có thể chứa NaN (đo được:
                    # layers.10.instruct_dim_reduce.3.weight = NaN ngay step 0 → img_tokens
                    # NaN 100% → loss NaN). Rác nên lỗi KHÔNG tất định và seed không cứu được.
                    idr = getattr(layer, "instruct_dim_reduce", None)
                    if idr is not None:
                        for sub in idr.modules():
                            if isinstance(sub, (nn.Linear, nn.LayerNorm)):
                                sub.reset_parameters()
                    sa = layer.self_attn
                    if hasattr(sa, "instruction_out_proj") and sa.instruction_out_proj is not None:
                        sa.instruction_out_proj.load_state_dict(sa.out_proj.state_dict())
                    # ReZero gate mặc định = 0 (paper QA-ViT). Trên bài OCR-chi-phối, gate
                    # gần như KHÔNG mở trong 5 epoch (đo được: |tanh(β)|≤0.011 mọi layer) →
                    # nhánh fusion inert → +qaclip ≈ baseline. Env QAVIT_GATE_INIT (>0) ép
                    # nhánh hoạt động NGAY từ đầu để kiểm xem nó có giá trị hay không.
                    # Mặc định 0.0 = giữ nguyên hành vi paper.
                    import os as _os
                    try:
                        _g0 = float(_os.environ.get("QAVIT_GATE_INIT", "0") or "0")
                    except ValueError:
                        _g0 = 0.0
                    if _g0 != 0.0 and hasattr(sa, "instruction_proj_gate"):
                        sa.instruction_proj_gate.fill_(_g0)

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_emb: Optional[torch.FloatTensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
            instruct_states=text_emb,
            instruct_masks=text_mask,
        )
