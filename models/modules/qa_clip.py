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
        if kv_states is None:
            raise ValueError("kv_states required")

        bsz = hidden_states.size(0)
        mm_len = kv_states.shape[1]

        hidden_states = torch.cat([kv_states, hidden_states], dim=1)
        bsz, tgt_len, embed_dim = hidden_states.size()

        device = hidden_states.device
        dtype_mask = torch.long

        if kv_masks is None:
            kv_masks = torch.ones(bsz, mm_len, dtype=dtype_mask, device=device)
        else:
            if kv_masks.size(1) != mm_len:
                if kv_masks.size(1) > mm_len:
                    kv_masks = kv_masks[:, :mm_len]
                else:
                    pad = torch.zeros(bsz, mm_len - kv_masks.size(1), dtype=dtype_mask, device=device)
                    kv_masks = torch.cat([kv_masks.to(dtype_mask), pad], dim=1)
            else:
                kv_masks = kv_masks.to(device=device, dtype=dtype_mask)

        vis_len = tgt_len - mm_len
        if attention_mask is None:
            vis_masks = torch.ones(bsz, vis_len, dtype=dtype_mask, device=device)
        else:
            vis_masks = attention_mask.to(device=device, dtype=dtype_mask)

        full_key_mask = torch.cat([kv_masks, vis_masks], dim=1)

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = attn_weights.to(torch.float32)

        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        if full_key_mask is not None:
            if full_key_mask.size(1) != src_len:
                 if full_key_mask.size(1) < src_len:
                     pad = torch.ones(bsz, src_len - full_key_mask.size(1), dtype=dtype_mask, device=device)
                     full_key_mask = torch.cat([full_key_mask, pad], dim=1)
                 else:
                     full_key_mask = full_key_mask[:, :src_len]

            key_padding_mask = full_key_mask[:, None, None, :].to(torch.bool)
            attn_weights = attn_weights.masked_fill(~key_padding_mask, torch.finfo(attn_weights.dtype).min)

        attn_weights = attn_weights[:, :, mm_len:, :]
        q_len = attn_weights.size(2)

        attn_weights = attn_weights.view(bsz * self.num_heads, q_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, q_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, embed_dim)

        if attention_mask is not None:
            vis_masks_f = vis_masks[:, :, None].to(attn_output.dtype)
            attn_output = attn_output * vis_masks_f

        gate_val = self.instruction_proj_gate.tanh()
        attn_output = self.out_proj(attn_output) + (self.instruction_out_proj(attn_output) * gate_val)

        return attn_output, attn_weights_reshaped

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
                    sa = layer.self_attn
                    if hasattr(sa, "instruction_out_proj") and sa.instruction_out_proj is not None:
                        sa.instruction_out_proj.load_state_dict(sa.out_proj.state_dict())

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
