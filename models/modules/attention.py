"""
models/modules/attention.py
LayerNorm, FC, MLP, AoA, SAoA, GAoA, SGAoA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModel,
    ViTModel,
    ViTImageProcessor,
)

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU()
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0.1, use_relu=True):
        super().__init__()
        self.fc = FC(in_size, mid_size, dropout_r, use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class AoA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        aoa_dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.aoa = nn.Sequential(
            nn.Linear(2 * inner_dim, 2 * dim),
            nn.GLU(),
            nn.Dropout(aoa_dropout)
        )

    def forward(self, x, context = None):
        h = self.heads
        q_ = self.to_q(x)
        context = default(context, x)
        kv = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q_, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)
        attn_out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(attn_out, 'b h n d -> b n (h d)', h = h)
        out = self.aoa(torch.cat((out, q_), dim = -1))
        return out

class SAoA(nn.Module):

    def __init__(self, feat_dim, heads=8):
        super(SAoA, self).__init__()
        self.AoA = AoA(dim=feat_dim, heads=heads)
        self.norm1 = LayerNorm(feat_dim)
        self.norm2 = LayerNorm(feat_dim)
        self.ffn = MLP(in_size= feat_dim,
                       mid_size= feat_dim,
                       out_size= feat_dim,
                       dropout_r= 0.1,
                       use_relu=True)
    def forward(self, x):
        x = self.norm1(x +
            self.AoA(x)
        )
        x = self.norm2(x +
            self.ffn(x)
        )
        return x

class GAoA(nn.Module):
    def __init__(self, feat_dim, heads):
        super().__init__()
        self.AoA = AoA(feat_dim, heads)
        self.norm1 = LayerNorm(feat_dim)
        self.norm2 = LayerNorm(feat_dim)
        self.ffn = MLP(feat_dim, feat_dim, feat_dim)

    def forward(self, x, context):
        x = self.norm1(x + self.AoA(x, context=context))
        x = self.norm2(x + self.ffn(x))
        return x

class SGAoA(nn.Module):
    def __init__(self, feat_dim, heads=8):
        super(SGAoA, self).__init__()
        self.AoA1 = AoA(dim=feat_dim, heads=heads)
        self.norm1 = LayerNorm(feat_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.AoA2 = AoA(dim=feat_dim, heads=heads)
        self.norm2 = LayerNorm(feat_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(feat_dim)
        self.dropout3 = nn.Dropout(0.1)
        self.ffn = MLP(in_size= feat_dim,
                       mid_size= feat_dim,
                       out_size= feat_dim,
                       dropout_r= 0.1,
                       use_relu=True)
    def forward(self, x, context):
        x = self.norm1(x + self.dropout1(
            self.AoA1(x)
        ))
        x = self.norm2(x + self.dropout2(
            self.AoA2(x, context = context)
        ))
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x