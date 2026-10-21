"""Joint image-patch/OCR-region attention before OCR subword expansion."""

import torch
from torch import nn


def relative_geometry(query_boxes, key_boxes):
    """Ordered pair features: dx, dy, log-size ratios, IoU and query coverage."""
    # Coordinates stay normalized. FP32 and finite size clamps also cover padded
    # or zero-area regions, before their attention keys are masked out.
    query = query_boxes.float().unsqueeze(-2)
    key = key_boxes.float().unsqueeze(-3)
    query_size = (query[..., 2:] - query[..., :2]).clamp_min(1e-6)
    key_size = (key[..., 2:] - key[..., :2]).clamp_min(1e-6)
    delta = (key[..., :2] + key[..., 2:] - query[..., :2] - query[..., 2:]) * 0.5
    log_ratio = (key_size.log() - query_size.log()).clamp(-5, 5)
    intersection_size = (
        torch.minimum(query[..., 2:], key[..., 2:])
        - torch.maximum(query[..., :2], key[..., :2])
    ).clamp_min(0)
    intersection = intersection_size.prod(dim=-1)
    query_area = query_size.prod(dim=-1)
    key_area = key_size.prod(dim=-1)
    iou = intersection / (query_area + key_area - intersection).clamp_min(1e-12)
    coverage = intersection / query_area.clamp_min(1e-12)
    return torch.cat(
        (delta, log_ratio, iou.unsqueeze(-1), coverage.unsqueeze(-1)), dim=-1
    )


class GeometryBias(nn.Module):
    """Small per-head geometry MLP; zero output preserves the baseline initially."""

    def __init__(self, num_heads):
        super().__init__()
        self.input = nn.Linear(6, 32)
        self.output = nn.Linear(32, num_heads)
        self.reset_parameters()

    def reset_parameters(self):
        self.input.reset_parameters()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, geometry):
        return self.output(torch.relu(self.input(geometry.to(self.input.weight.dtype))))


class JointAttentionLayer(nn.Module):
    """Pre-RMSNorm residual self-attention, without an FFN or dropout."""

    def __init__(self, d_model, num_heads, eps):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.geometry_bias = GeometryBias(num_heads)

    def reset_parameters(self):
        # HF skips constructor init during warm-start. Initialize both packed QKV
        # and out_proj explicitly for newly added layers in older checkpoints.
        self.attention.out_proj.reset_parameters()
        self.attention._reset_parameters()
        self.norm.reset_parameters()
        self.geometry_bias.reset_parameters()

    def forward(self, features, valid_mask, image_to_ocr, ocr_to_image):
        normalized = self.norm(features.to(self.norm.weight.dtype)).to(features.dtype)
        batch_size, length, _ = features.shape
        image_length = image_to_ocr.shape[1]
        heads = self.attention.num_heads
        bias = normalized.new_zeros(batch_size, heads, length, length)
        bias[:, :, :image_length, image_length:] = (
            self.geometry_bias(image_to_ocr).permute(0, 3, 1, 2).to(normalized.dtype)
        )
        bias[:, :, image_length:, :image_length] = (
            self.geometry_bias(ocr_to_image).permute(0, 3, 1, 2).to(normalized.dtype)
        )
        # One additive mask for geometry and padding, with matching dtype. Padded
        # query outputs are zeroed below; valid image keys avoid all-masked rows.
        bias = bias.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=bias.reshape(batch_size * heads, length, length),
            need_weights=False,
        )
        return (features + attended).masked_fill(~valid_mask.unsqueeze(-1), 0)


class OCRImageFusion(nn.Module):
    """Three joint-attention layers with cross-modal spatial biases."""

    def __init__(self, d_model, num_heads, eps, image_grid_size):
        super().__init__()
        rows, columns = image_grid_size
        if rows < 1 or columns < 1:
            raise ValueError("The image patch grid must have positive dimensions.")
        row, column = torch.meshgrid(
            torch.arange(rows, dtype=torch.float32),
            torch.arange(columns, dtype=torch.float32),
            indexing="ij",
        )
        patch_boxes = torch.stack(
            (column / columns, row / rows, (column + 1) / columns, (row + 1) / rows),
            dim=-1,
        ).reshape(-1, 4)
        # Recreated from CLIP's grid, never loaded as trained checkpoint weights.
        self.register_buffer("patch_boxes", patch_boxes, persistent=False)
        self.layers = nn.ModuleList(
            JointAttentionLayer(d_model, num_heads, eps) for _ in range(3)
        )

    def forward(self, image_features, ocr_features, image_mask, word_mask, ocr_boxes):
        if image_features.shape[-1] != ocr_features.shape[-1]:
            raise ValueError("QA-ViT and OCR feature dimensions must match for fusion.")
        image_length = image_features.shape[1]
        if image_length != self.patch_boxes.shape[0]:
            raise ValueError("Image token count does not match the CLIP patch grid.")
        if ocr_boxes.shape != (*word_mask.shape, 4):
            raise ValueError("OCR boxes must align with the word-level padding mask.")
        # Clean padding before geometry arithmetic, including arbitrary padded
        # coordinates; geometry never expands regions into subwords.
        boxes = ocr_boxes.masked_fill(~word_mask.bool().unsqueeze(-1), 0)
        patches = self.patch_boxes.float().unsqueeze(0).expand(boxes.shape[0], -1, -1)
        image_to_ocr = relative_geometry(patches, boxes)
        ocr_to_image = relative_geometry(boxes, patches)
        valid = torch.cat((image_mask, word_mask), dim=1).bool()
        features = torch.cat((image_features, ocr_features), dim=1)
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        for layer in self.layers:
            features = layer(features, valid, image_to_ocr, ocr_to_image)
        return features[:, :image_length], features[:, image_length:]
