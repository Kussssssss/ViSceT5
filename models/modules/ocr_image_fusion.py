"""Joint image-patch/OCR-region attention before OCR subword expansion."""

import torch
from torch import nn


class JointAttentionLayer(nn.Module):
    """Pre-RMSNorm residual self-attention, without an FFN or dropout."""

    def __init__(self, d_model, num_heads, eps):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def reset_parameters(self):
        # HF skips constructor init during warm-start. Initialize both packed QKV
        # and out_proj explicitly for newly added layers in older checkpoints.
        self.attention.out_proj.reset_parameters()
        self.attention._reset_parameters()
        self.norm.reset_parameters()

    def forward(self, features, valid_mask):
        normalized = self.norm(features.to(self.norm.weight.dtype)).to(features.dtype)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        return (features + attended).masked_fill(~valid_mask.unsqueeze(-1), 0)


class OCRImageFusion(nn.Module):
    """Three independent attention layers over [image patches, OCR entries]."""

    def __init__(self, d_model, num_heads, eps):
        super().__init__()
        self.layers = nn.ModuleList(
            JointAttentionLayer(d_model, num_heads, eps) for _ in range(3)
        )

    def forward(self, image_features, ocr_features, image_mask, word_mask):
        if image_features.shape[-1] != ocr_features.shape[-1]:
            raise ValueError("QA-ViT and OCR feature dimensions must match for fusion.")
        image_length = image_features.shape[1]
        valid = torch.cat((image_mask, word_mask), dim=1).bool()
        features = torch.cat((image_features, ocr_features), dim=1)
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        for layer in self.layers:
            features = layer(features, valid)
        return features[:, :image_length], features[:, image_length:]
