"""Question-aware CLIP, adapted from amazon-science/QA-ViT.

Upstream: models/clip_qavit.py at 96faee6ef8bb0157fe366d8315f175e8c4dbe6b8.
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
Upstream is licensed under Apache-2.0; see docs/QA_VIT_LICENSE.txt.

Keep upstream parameter names for checkpoint compatibility. ViSceT5 additionally
masks question padding, supports visual-query attention inspection, and
accumulates attention in FP32 for mixed precision training.

gated_joint=True (config.gated_joint): the gate covers every question path. Upstream runs one
softmax over [visual, question] keys and passes it through the frozen, ungated out_proj, so at
beta = 0 the visual tokens already mix question values (measured: CLIP features change by 7%).
With s_v, s_q the scores of a visual query on the visual / question keys, A_v, A_q their softmaxes
and C_v = A_v V_v, C_q = A_q V_q, the joint softmax splits exactly as
    joint = (1 - lam) C_v + lam C_q = C_v + lam (C_q - C_v),
    lam = Z_q / (Z_v + Z_q) = sigmoid(logsumexp s_q - logsumexp s_v),
i.e. plain CLIP attention plus a question term weighted by lam, the share of the query's attention
that the question takes. The layer is
    out = out_proj(C_v) + tanh(beta) W' (lam (C_q - C_v)),
with W' = instruction_out_proj.weight (no bias; initialised as out_proj.weight):
  beta = 0      -> exactly the frozen CLIP layer (bit-identical), so QA-ViT starts as plain CLIP;
  tanh(beta)=1, W' = W_o -> exactly the joint attention of QA-ViT (paper Eq. 3) through out_proj;
  dL/dbeta = <W' lam (C_q - C_v), dL/dout> != 0 at beta = 0, so the gate can open;
  lam keeps QA-ViT's per-patch selectivity: a patch that does not match the question gets little
  question signal (a separate cross-attention softmax would give every patch a unit mixture).
The relevance logit logit(lam) = logsumexp s_q - logsumexp s_v does not pass through the gate; it
is what the answer-region grounding loss supervises (output_relevance).
"""

from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import (
    CLIPAttention,
    CLIPEncoderLayer,
    CLIPMLP,
    CLIPPreTrainedModel,
    CLIPVisionEmbeddings,
)


class FeedForward(nn.Sequential):
    """Upstream instruction adapter: LayerNorm, bias-free Linear, GELU, Linear."""

    def __init__(self, in_dim, out_dim, inner_dim=None):
        inner_dim = out_dim if inner_dim is None else inner_dim
        super().__init__(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, out_dim, bias=False),
        )


def _valid_tokens(mask, batch_size, length, device, name):
    """Accept a two-dimensional 1/0 padding mask; never silently resize it."""
    if mask is None:
        return torch.ones(batch_size, length, device=device, dtype=torch.bool)
    if mask.shape != (batch_size, length):
        raise ValueError(
            f"{name} must have shape {(batch_size, length)}, got {tuple(mask.shape)}"
        )
    return mask.to(device=device, dtype=torch.bool)


class MMCLIPAttention(CLIPAttention):
    """Visual queries attend to [visual, question] keys, following paper Eq. (3).

    Returned attention columns contain all visual tokens (including CLS), then
    instruction tokens. Rows contain only visual queries.
    """

    def __init__(self, config):
        super().__init__(config)
        self.instruction_out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.instruction_proj_gate = nn.Parameter(torch.zeros(1))
        self.gated_joint = bool(getattr(config, "gated_joint", False))

    def _gated_joint(self, hidden_states, kv_states, kv_masks, output_attentions, output_relevance):
        """out = CLIPAttention(x) + tanh(beta) W' (lam (C_q - C_v)); see the module docstring."""
        batch_size, length, embed_dim = hidden_states.shape
        # The frozen CLIP layer itself, so beta = 0 is bit-identical to plain CLIP.
        base = CLIPAttention.forward(self, hidden_states=hidden_states, output_attentions=False)[0]
        if kv_states.shape[1] == 0:
            return (base, None, None) if output_relevance else (base, None)
        question = _valid_tokens(kv_masks, batch_size, kv_states.shape[1], hidden_states.device, "kv_masks")
        query = self._shape(self.q_proj(hidden_states) * self.scale, length, batch_size)
        key_v = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_v = self._shape(self.v_proj(hidden_states), -1, batch_size)
        key_q = self._shape(self.k_proj(kv_states), -1, batch_size)
        value_q = self._shape(self.v_proj(kv_states), -1, batch_size)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            query = query.float()
            scores_v = torch.matmul(query, key_v.float().transpose(-1, -2))
            # -1e4, not finfo.min: a finite fill keeps logsumexp, the head mean and the softmax finite.
            scores_q = torch.matmul(query, key_q.float().transpose(-1, -2)).masked_fill(
                ~question[:, None, None, :], -1e4)
            relevance = torch.logsumexp(scores_q, dim=-1) - torch.logsumexp(scores_v, dim=-1)  # logit(lam)
            share = torch.sigmoid(relevance).unsqueeze(-1)
            weights_v = F.softmax(scores_v, dim=-1)
            weights_q = F.softmax(scores_q, dim=-1)
            context_v = torch.matmul(F.dropout(weights_v, p=self.dropout, training=self.training), value_v.float())
            context_q = torch.matmul(F.dropout(weights_q, p=self.dropout, training=self.training), value_q.float())
            delta = share * (context_q - context_v)  # joint-softmax context minus plain CLIP context
        delta = delta.transpose(1, 2).reshape(batch_size, length, embed_dim).to(base.dtype)
        output = base + self.instruction_proj_gate.tanh() * F.linear(delta, self.instruction_out_proj.weight)
        weights = None
        if output_attentions:  # the joint attention this layer interpolates to: [(1-lam) A_v, lam A_q]
            weights = torch.cat(((1 - share) * weights_v, share * weights_q), dim=-1)
        return (output, weights, relevance) if output_relevance else (output, weights)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
        kv_states=None,
        kv_masks=None,
        output_relevance=False,
    ):
        """Returns (output, attention weights or None) [+ relevance logits [B, H, N] if output_relevance]."""
        if self.gated_joint:
            if kv_states is None:
                raise ValueError("Projected instruction states are required.")
            if attention_mask is not None or causal_attention_mask is not None:
                raise ValueError("The gated joint attention takes no visual attention mask.")
            return self._gated_joint(hidden_states, kv_states, kv_masks, output_attentions, output_relevance)
        if output_relevance:
            raise ValueError("output_relevance needs gated_joint.")
        if kv_states is None:
            raise ValueError(
                "Projected instruction states are required; use an empty sequence to disable them."
            )
        if causal_attention_mask is not None:
            raise ValueError(
                "Question-aware vision attention does not support a causal mask."
            )

        batch_size, visual_length, embed_dim = hidden_states.shape
        instruction_length = kv_states.shape[1]
        visual_mask = _valid_tokens(
            attention_mask,
            batch_size,
            visual_length,
            hidden_states.device,
            "attention_mask",
        )
        instruction_mask = _valid_tokens(
            kv_masks, batch_size, instruction_length, hidden_states.device, "kv_masks"
        )
        key_mask = torch.cat((visual_mask, instruction_mask), dim=1)[:, None, None, :]
        joint_states = torch.cat((hidden_states, kv_states), dim=1)

        # Upstream projects all queries, then discards the instruction rows.
        # Project only visual queries: the retained attention is mathematically identical.
        queries = self._shape(
            self.q_proj(hidden_states) * self.scale, visual_length, batch_size
        )
        keys = self._shape(self.k_proj(joint_states), -1, batch_size)
        values = self._shape(self.v_proj(joint_states), -1, batch_size)

        # Disable autocast here: casting inputs alone does not force an FP32 matmul.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2))
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1).masked_fill(~key_mask, 0)
            probabilities = F.dropout(weights, p=self.dropout, training=self.training)
            attended = torch.matmul(probabilities, values.float())

        attended = attended.transpose(1, 2).reshape(
            batch_size, visual_length, embed_dim
        )
        attended = attended.to(hidden_states.dtype)
        output = self.out_proj(attended)
        output = output + self.instruction_proj_gate.tanh() * self.instruction_out_proj(
            attended
        )
        output = output * visual_mask.unsqueeze(-1).to(output.dtype)
        return output, weights if output_attentions else None


class MMCLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MMCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.instruct_dim_reduce = FeedForward(config.instruction_dim, self.embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
        instruct_states=None,
        instruct_masks=None,
        output_relevance=False,
    ):
        result = self.self_attn(
            hidden_states=self.layer_norm1(hidden_states),
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            kv_states=self.instruct_dim_reduce(instruct_states),
            kv_masks=instruct_masks,
            output_relevance=output_relevance,
        )
        attended, weights = result[0], result[1]
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        outputs = (hidden_states, weights) if output_attentions else (hidden_states,)
        return outputs + (result[2],) if output_relevance else outputs


def _instruction_layers(integration_point, num_layers):
    """The six integration schedules implemented by QA-ViT."""
    schedules = {
        # "none": CLIP thường, không adapter nào (dùng khi CLIP không nhận câu hỏi).
        "none": lambda i: False,
        "all": lambda i: True,
        "early": lambda i: i < num_layers // 2,
        "late": lambda i: i >= num_layers // 2,
        "late2": lambda i: i >= 3 * num_layers // 4,
        "late3": lambda i: i >= num_layers // 4,
        "sparse": lambda i: i % 2 == 0,
    }
    if integration_point not in schedules:
        raise ValueError(
            f"Unsupported integration_point {integration_point!r}; choose from {tuple(schedules)}"
        )
    return {i for i in range(num_layers) if schedules[integration_point](i)}


class InstructCLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        instruction_layers = _instruction_layers(
            config.integration_point, config.num_hidden_layers
        )
        self.layers = nn.ModuleList(
            (MMCLIPEncoderLayer if i in instruction_layers else CLIPEncoderLayer)(
                config
            )
            for i in range(config.num_hidden_layers)
        )
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        instruct_states=None,
        instruct_masks=None,
        output_relevance=False,
    ):
        """output_relevance: also return the relevance logits [B, H, N] of every question layer
        (gated_joint), as a tuple after the usual outputs."""
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        relevances = ()
        hidden_states = inputs_embeds

        for layer in self.layers:
            if output_hidden_states:
                encoder_states += (hidden_states,)
            # Pass instruction tensors as positional inputs, including for reentrant
            # checkpointing. Bind each layer directly, avoiding late-bound closures.
            arguments = (
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions,
            )
            question_layer = isinstance(layer, MMCLIPEncoderLayer)
            if question_layer:
                arguments += (instruct_states, instruct_masks, output_relevance)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(layer.__call__, *arguments)
            else:
                outputs = layer(*arguments)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions += (outputs[1],)
            if output_relevance and question_layer:
                relevances += (outputs[-1],)

        if output_hidden_states:
            encoder_states += (hidden_states,)
        if not return_dict:
            outputs = tuple(
                value
                for value in (hidden_states, encoder_states, all_attentions)
                if value is not None
            )
        else:
            outputs = BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=encoder_states,
                attentions=all_attentions,
            )
        return (outputs, relevances) if output_relevance else outputs


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = InstructCLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        instruct_states=None,
        instruct_masks=None,
        output_relevance=False,
    ):
        if pixel_values is None:
            raise ValueError("pixel_values are required")
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        hidden_states = self.pre_layrnorm(self.embeddings(pixel_values))
        if instruct_states is None:
            instruct_states = hidden_states.new_empty(
                hidden_states.shape[0], 0, self.config.instruction_dim
            )
        if (
            instruct_states.ndim != 3
            or instruct_states.shape[0] != hidden_states.shape[0]
            or instruct_states.shape[2] != self.config.instruction_dim
        ):
            raise ValueError(
                f"Instruction states must have shape [B, L, {self.config.instruction_dim}], "
                f"got {tuple(instruct_states.shape)}"
            )
        instruct_states = instruct_states.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        instruct_masks = _valid_tokens(
            instruct_masks,
            hidden_states.shape[0],
            instruct_states.shape[1],
            hidden_states.device,
            "text_mask",
        )
        outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            instruct_states=instruct_states,
            instruct_masks=instruct_masks,
            output_relevance=output_relevance,
        )
        if output_relevance:
            outputs, relevances = outputs
        last_hidden_state = outputs[0]
        pooled_output = self.post_layernorm(last_hidden_state[:, 0])
        if not return_dict:
            result = (last_hidden_state, pooled_output) + outputs[1:]
        else:
            result = BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return (result, relevances) if output_relevance else result


class QACLIPEncoder(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: CLIPVisionConfig,
        instruction_dim: Optional[int] = None,
        freeze_clip: Optional[bool] = None,
        integration_point: Optional[str] = None,
        gated_joint: Optional[bool] = None,
    ):
        super().__init__(config)
        config.gated_joint = bool(
            getattr(config, "gated_joint", False) if gated_joint is None else gated_joint
        )
        # Resolve overrides before constructing modules, including non-768 inputs.
        config.instruction_dim = int(
            getattr(config, "instruction_dim", 768)
            if instruction_dim is None
            else instruction_dim
        )
        config.integration_point = (
            getattr(config, "integration_point", "late")
            if integration_point is None
            else integration_point
        )
        config.freeze_clip = bool(
            getattr(config, "freeze_clip", False)
            if freeze_clip is None
            else freeze_clip
        )
        if config.instruction_dim <= 0:
            raise ValueError("instruction_dim must be positive")
        self.vision_model = CLIPVisionTransformer(config)
        self.post_init()
        self._apply_freeze()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FeedForward):
            # Hugging Face skips constructor initialization when loading a backbone.
            # Its CLIP initializer does not initialize arbitrary Linear weights.
            for layer in module:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        elif isinstance(module, MMCLIPAttention):
            # In memory-efficient loading, the backbone may still be on meta.
            # Copy after its checkpoint tensors have been materialized instead.
            if not any(
                parameter.is_meta
                for projection in (module.out_proj, module.instruction_out_proj)
                for parameter in projection.parameters()
            ):
                module.instruction_out_proj.load_state_dict(
                    module.out_proj.state_dict()
                )
            module.instruction_proj_gate.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Copy loaded CLIP projections only where the checkpoint lacks adapter weights."""
        return_loading_info = kwargs.pop("output_loading_info", False)
        if kwargs.get("low_cpu_mem_usage") and kwargs.get("device_map") is None:
            # Transformers 4.45 compares storage pointers to detect tied weights
            # without a device map. Equal-shaped meta tensors falsely look tied,
            # hiding missing adapter weights. Explicit CPU placement uses actual
            # parameter identity and retains the memory-efficient loading path.
            kwargs["device_map"] = {"": "cpu"}
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            output_loading_info=True,
            **kwargs,
        )
        missing = set(loading_info["missing_keys"])
        with torch.no_grad():
            for index, layer in enumerate(model.vision_model.encoder.layers):
                if isinstance(layer, MMCLIPEncoderLayer):
                    prefix = f"vision_model.encoder.layers.{index}.self_attn.instruction_out_proj."
                    for name in ("weight", "bias"):
                        if prefix + name in missing:
                            getattr(layer.self_attn.instruction_out_proj, name).copy_(
                                getattr(layer.self_attn.out_proj, name)
                            )
        model._apply_freeze()
        return (model, loading_info) if return_loading_info else model

    def _apply_freeze(self):
        self.requires_grad_(not self.config.freeze_clip)
        if self.config.freeze_clip:
            for layer in self.vision_model.encoder.layers:
                if isinstance(layer, MMCLIPEncoderLayer):
                    layer.instruct_dim_reduce.requires_grad_(True)
                    layer.self_attn.instruction_out_proj.requires_grad_(True)
                    layer.self_attn.instruction_proj_gate.requires_grad_(True)
                    if layer.self_attn.gated_joint:  # W' acts on a difference of contexts: no bias
                        layer.self_attn.instruction_out_proj.bias.requires_grad_(False)

    def init_qavit_comps(self):
        """Explicitly reset adapter projections from CLIP; never call on a trained checkpoint."""
        with torch.no_grad():
            for layer in self.vision_model.encoder.layers:
                if isinstance(layer, MMCLIPEncoderLayer):
                    layer.self_attn.instruction_out_proj.load_state_dict(
                        layer.self_attn.out_proj.state_dict()
                    )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Non-reentrant checkpointing supports frozen vision inputs and trainable
        # adapters without requiring dummy input gradients.
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values=None,
        text_emb=None,
        text_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_relevance=False,
    ):
        """output_relevance (gated_joint): return (outputs, relevance logits of each question layer)."""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            instruct_states=text_emb,
            instruct_masks=text_mask,
            output_relevance=output_relevance,
        )
