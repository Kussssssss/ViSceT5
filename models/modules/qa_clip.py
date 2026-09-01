"""
models/modules/qa_clip.py
QACLIPEncoder — CLIP with instruction-guided late-fusion encoder.
"""

from typing import Optional, Tuple, Union
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
            kv_states=self.instruct_dim_reduce(instruct_states) if self.instruct_dim_reduce else instruct_states,
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
        for encoder_layer in self.layers:
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
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class FlexibleCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch_size, embed_dim, grid_h, grid_w]
        grid_h, grid_w = patch_embeds.shape[2], patch_embeds.shape[3]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        num_positions = embeddings.shape[1]
        orig_positions = self.position_embedding.weight.shape[0]

        if num_positions == orig_positions:
            pos_emb = self.position_embedding(self.position_ids)
        else:
            # 2D Bicubic Positional Embedding Interpolation (PreSTU / ViT standard)
            orig_grid = int(round((orig_positions - 1) ** 0.5))  # typically 14 for 224x224
            cls_pos = self.position_embedding.weight[:1, :].unsqueeze(0)  # (1, 1, D)
            patch_pos = self.position_embedding.weight[1:, :].unsqueeze(0)  # (1, orig_grid*orig_grid, D)
            D = patch_pos.shape[-1]
            patch_pos = patch_pos.transpose(1, 2).reshape(1, D, orig_grid, orig_grid)
            patch_pos_interp = F.interpolate(
                patch_pos.float(),
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False,
            ).to(patch_pos.dtype)
            patch_pos_interp = patch_pos_interp.flatten(2).transpose(1, 2)  # (1, grid_h*grid_w, D)
            pos_emb = torch.cat([cls_pos, patch_pos_interp], dim=1)

        embeddings = embeddings + pos_emb
        return embeddings

class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = FlexibleCLIPVisionEmbeddings(config)
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

class QACLIPEncoder(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig, instruction_dim: int = 768, freeze_clip: bool = False):
        super().__init__(config)
        self.config.instruction_dim = int(getattr(self.config, "instruction_dim", instruction_dim))
        self.config.integration_point = getattr(self.config, "integration_point", "late")
        self.config.freeze_clip = bool(getattr(self.config, "freeze_clip", freeze_clip))
        self.vision_model = CLIPVisionTransformer(self.config)
        self._apply_freeze()
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        instruction_dim = kwargs.pop("instruction_dim", None)
        integration_point = kwargs.pop("integration_point", None)
        freeze_clip = kwargs.pop("freeze_clip", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if instruction_dim is not None:
            model.config.instruction_dim = int(instruction_dim)
        if integration_point is not None:
            model.config.integration_point = integration_point
        if freeze_clip is not None:
            model.config.freeze_clip = bool(freeze_clip)
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
