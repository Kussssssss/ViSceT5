"""ViT5 whose answer distribution mixes generation with a copy from the OCR tokens.

p(w) = g * p_vocab(w) + (1 - g) * sum_{j: id_j = w} p_copy(j)     (pointer-generator, See et al. 2017)

The copy scores read the encoder output of the OCR tokens, the features TWC aligns, and the two OCR
copies add up on the same token id, so a word can be copied in its corrected spelling.

Continuing a phrase: when the previous answer token equals OCR token j-1, token j gets the bonus
copy_continue + copy_link_gain * link[j], with link[j] the ConsFormer probability that tokens j-1 and
j belong to one constituent. Without ConsFormer only copy_continue applies.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class CopyT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.copy_query = nn.Linear(config.d_model, config.d_model)
        self.copy_key = nn.Linear(config.d_model, config.d_model)
        self.copy_gate = nn.Linear(config.d_model, 1)
        self.copy_continue = nn.Parameter(torch.tensor(1.0))
        self.copy_link_gain = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def reset_copy_parameters(self):
        """Called after from_pretrained, which skips the init of the parameters missing from ViT5.
        Query and key start as identity, g starts at sigmoid(1) ~ 0.73 (mostly generation)."""
        for layer in (self.copy_query, self.copy_key):
            layer.weight.copy_(torch.eye(layer.weight.shape[0], dtype=layer.weight.dtype))
            layer.bias.zero_()
        self.copy_gate.weight.zero_()
        self.copy_gate.bias.fill_(1.0)
        self.copy_continue.fill_(1.0)
        self.copy_link_gain.fill_(1.0)

    def prepare_inputs_for_generation(self, input_ids, copy_ids=None, copy_prev=None, copy_links=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        inputs.update(copy_ids=copy_ids, copy_prev=copy_prev, copy_links=copy_links)
        return inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        copy_ids=None,
        copy_prev=None,
        copy_links=None,
    ):
        """copy_ids: [B, S] token id of each OCR position of the encoder output, -1 elsewhere;
        copy_prev: [B, S] id of the previous token of the same OCR copy, -1 if none;
        copy_links: [B, S] ConsFormer link to the previous token, optional.
        Without copy_ids this is plain ViT5 (scripts/visualize_attention.py).

        logits are log probabilities. With labels they stop at the last labelled step; without
        labels only the last step is returned, which is all generation reads."""
        if copy_ids is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
                decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds, labels=labels, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        outputs = super().forward(
            attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs,
            past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=True, return_dict=True,
        )
        if labels is not None:
            # 32 x 56 x vocabulary would be ~250 MB: answers are padded far beyond their length.
            steps = slice(0, int((labels != -100).any(dim=0).nonzero().max()) + 1)
            labels = labels[:, steps]
        else:
            steps = slice(-1, None)
        hidden = outputs.decoder_hidden_states[-1][:, steps].float()
        previous = decoder_input_ids[:, steps]
        memory = outputs.encoder_last_hidden_state.float()

        scores = self.copy_query(hidden) @ self.copy_key(memory).transpose(1, 2) / math.sqrt(hidden.shape[-1])
        bonus = self.copy_continue
        if copy_links is not None:
            bonus = bonus + self.copy_link_gain * copy_links.float().unsqueeze(1)
        scores = scores + (previous.unsqueeze(2) == copy_prev.unsqueeze(1)).float() * bonus
        source = copy_ids >= 0
        copy_probs = scores.masked_fill(~source.unsqueeze(1), -1e4).softmax(dim=-1)

        gate = torch.sigmoid(self.copy_gate(hidden))
        mixed = gate * outputs.logits[:, steps].float().softmax(dim=-1)
        index = copy_ids.clamp_min(0).unsqueeze(1).expand(-1, hidden.shape[1], -1)
        mixed = mixed.scatter_add(-1, index, (1 - gate) * copy_probs * source.unsqueeze(1).float())
        logits = mixed.clamp_min(1e-12).log()

        loss = None
        if labels is not None:
            loss = F.nll_loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        )
