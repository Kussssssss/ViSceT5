"""
models/modules/ocr_consformer.py
OCREncoder (Consformer): GroupAttention stacks.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class ScaledDotProductAttention_Encoder(nn.Module):
    def __init__(self, head: int, d_model: int, d_kv: int):
        super().__init__()
        self.d_model = d_model
        self.d_q = d_kv
        self.d_kv = d_kv
        self.head = head
        self.fc_q = nn.Linear(d_model, head * d_kv)
        self.fc_k = nn.Linear(d_model, head * d_kv)
        self.fc_v = nn.Linear(d_model, head * d_kv)

    def forward(self, queries, keys, values, group_prob, attention_mask):
        B, Nq = queries.shape[:2]
        Nk = keys.shape[1]
        q = self.fc_q(queries).view(B, Nq, self.head, self.d_q).permute(0, 2, 1, 3)
        k = self.fc_k(keys)  .view(B, Nk, self.head, self.d_kv).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(B, Nk, self.head, self.d_kv).permute(0, 2, 1, 3)
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        scores = torch.matmul(qf, kf) / math.sqrt(self.d_kv)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e4)
        att_f = F.softmax(scores, dim=-1)
        if isinstance(group_prob, torch.Tensor):
            att_f = att_f * group_prob.to(att_f.dtype)
        else:
            att_f = att_f * float(group_prob)
        out_f = torch.matmul(att_f, v.to(att_f.dtype))
        out = out_f.to(queries.dtype).permute(0, 2, 1, 3).reshape(B, -1, self.d_model)
        return out

def build_neighbor_mask(mask_1d: torch.Tensor) -> torch.Tensor:
    valid = mask_1d.bool()
    B, L = valid.shape
    device = valid.device
    adj = valid[:, :, None] & valid[:, None, :]
    band = (
        torch.diag(torch.ones(L-1, dtype=torch.bool, device=device),  1) |
        torch.diag(torch.ones(L-1, dtype=torch.bool, device=device), -1)
    )
    nb_mask = adj & band.unsqueeze(0)
    return nb_mask

class GroupAttention(nn.Module):
    def __init__(self, head, d_model):
        super().__init__()
        self.h = head
        self.d_k = d_model // head
        self.linear_key   = nn.Linear(self.d_k, self.d_k)
        self.linear_query = nn.Linear(self.d_k, self.d_k)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context, nb_mask, prior):
        B, L = context.size()[:2]
        device = context.device
        dtype_in = context.dtype
        ctx = self.norm(context).view(B, L, self.h, self.d_k).transpose(1, 2)
        q = self.linear_query(ctx).to(torch.float32)
        k = self.linear_key  (ctx).to(torch.float32)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~nb_mask[:, None, :, :], -1e4)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-4)
        if isinstance(prior, torch.Tensor):
            prior_f = prior.to(device=device, dtype=torch.float32)
        else:
            prior_f = torch.tensor(float(prior), device=device, dtype=torch.float32)
        neibor_attn = prior_f + (1.0 - prior_f) * neibor_attn
        tri = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=0).unsqueeze(0).unsqueeze(0)
        tri_f = tri.to(torch.float32)
        t = torch.log(neibor_attn + 1e-6).matmul(tri_f)
        g_attn = tri_f.matmul(t).exp()
        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(~eye, 1e-4)
        return g_attn.to(dtype_in), neibor_attn.to(dtype_in)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class OCREncoderLayer(nn.Module):
    def __init__(self, head, d_model, d_kv, d_ff, dropout=0.1):
        super().__init__()
        self.group_attn   = GroupAttention(head, d_model)
        self.self_attn    = ScaledDotProductAttention_Encoder(head, d_model, d_kv)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer     = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask_1d, nb_mask_2d, group_prob):
        group_prob, break_prob = self.group_attn(x, nb_mask_2d, group_prob)
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, group_prob, mask_1d))
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob


class OCREncoder(nn.Module):
    def __init__(self, head, d_model, d_kv, d_ff, word_embed=None, num_layers: int = 3):
        super().__init__()
        self.head = head
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self._word_embed_proxy = None
        self._legacy_embed = word_embed if isinstance(word_embed, nn.Module) else None
        self.layers = clones(OCREncoderLayer(head, d_model, d_kv, d_ff), num_layers)
        self.norm = nn.LayerNorm(d_model)

    def set_word_embed_proxy(self, proxy_callable):
        self._word_embed_proxy = proxy_callable

    def forward(self, inputs: torch.LongTensor, mask_1d: torch.LongTensor):
        if self._word_embed_proxy is not None:
            x = self._word_embed_proxy(inputs)
        elif self._legacy_embed is not None:
            x = self._legacy_embed(inputs)
        else:
            raise RuntimeError("OCREncoder: No word embedding proxy set.")
        dtype_in = x.dtype
        nb_mask_2d = build_neighbor_mask(mask_1d)
        break_probs = []
        group_prob = 0.0
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask_1d, nb_mask_2d, group_prob)
            break_probs.append(break_prob)
        x = self.norm(x).to(dtype_in)
        break_probs = torch.stack(break_probs, dim=1)
        return x, break_probs