"""
models/modules/ocr_spatial.py
SpatialCirclePosition, SemanticOCREmbedding.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        h = config.num_attention_heads
        d_k = d_model // h
        d_v = d_model // h
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / math.sqrt(self.d_k)
        if attention_mask is not None:
            att = att + attention_mask
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out, att


class SpatialCirclePosition(ScaledDotProductAttention_Embedding):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.dist_embedding = nn.Embedding(config.num_distances, config.num_attention_heads)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.register_buffer("eps", torch.tensor(1e-6, dtype=torch.float32), persistent=False)

    def _target_dd(self):
        w = self.fc_q.weight if hasattr(self, "fc_q") else self.layer_norm.weight
        return w.device, w.dtype

    def calculate_distances(self, patch_x, patch_y, device, dtype):
        dx = (patch_x.unsqueeze(1) - patch_x.unsqueeze(2)).to(device=device, dtype=dtype)
        dy = (patch_y.unsqueeze(1) - patch_y.unsqueeze(2)).to(device=device, dtype=dtype)
        eps = self.eps.to(device=device, dtype=dtype)
        return torch.sqrt(dx * dx + dy * dy + eps)

    def patch(self, ocr_boxes: torch.Tensor, image_sizes: torch.Tensor, device, dtype):
        ocr_boxes = ocr_boxes.to(device=device, dtype=dtype)
        image_sizes = image_sizes.to(device=device, dtype=dtype)
        B = ocr_boxes.size(0)
        size_per_area = image_sizes[:, :2] / 11.0
        lower_bounds = torch.arange(0, 11, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        higher_bounds = lower_bounds + 1
        width_lower_bounds  = lower_bounds  * size_per_area[:, 0:1]
        width_higher_bounds = higher_bounds * size_per_area[:, 0:1]
        height_lower_bounds  = lower_bounds  * size_per_area[:, 1:2]
        height_higher_bounds = higher_bounds * size_per_area[:, 1:2]
        x0, y0, x1, y1 = ocr_boxes.unbind(-1)
        ocr_x_centroid = (x0 + x1) * 0.5
        ocr_y_centroid = (y0 + y1) * 0.5
        sel_x_mask = (width_lower_bounds.unsqueeze(1) <= ocr_x_centroid.unsqueeze(-1)) & \
                     (ocr_x_centroid.unsqueeze(-1) <= width_higher_bounds.unsqueeze(1))
        sel_y_mask = (height_lower_bounds.unsqueeze(1) <= ocr_y_centroid.unsqueeze(-1)) & \
                     (ocr_y_centroid.unsqueeze(-1) <= height_higher_bounds.unsqueeze(1))
        selected_x_centroid = sel_x_mask.to(dtype).argmax(dim=-1)
        selected_y_centroid = sel_y_mask.to(dtype).argmax(dim=-1)
        return selected_x_centroid, selected_y_centroid

    def forward(self, features, info, mask=None):
        target_device, target_dtype = self._target_dd()
        features = features.to(device=target_device, dtype=target_dtype)
        features = self.layer_norm(features)

        image_sizes, boxes = [], []
        for item in info:
            size = torch.tensor(
                [item["width"], item["height"], item["width"], item["height"]],
                device=target_device, dtype=target_dtype
            )
            image_sizes.append(size)
            box = item["boxes"]
            if not torch.is_tensor(box):
                box = torch.tensor(box, device=target_device, dtype=target_dtype)
            else:
                box = box.to(device=target_device, dtype=target_dtype)
            box = box * size
            boxes.append(box)

        image_sizes = torch.stack(image_sizes, dim=0).to(device=target_device, dtype=target_dtype)
        boxes       = torch.stack(boxes,       dim=0).to(device=target_device, dtype=target_dtype)

        bs, nq, _ = boxes.shape
        patch_x, patch_y = self.patch(boxes, image_sizes, device=target_device, dtype=target_dtype)
        dist = self.calculate_distances(patch_x, patch_y, device=target_device, dtype=torch.float32) * 2.0
        max_idx = self.dist_embedding.num_embeddings - 1
        dist_idx = torch.clamp(dist, 0, float(max_idx) + 1e-5).to(dtype=torch.long)
        dist_embed = self.dist_embedding(dist_idx)
        dist_embed = dist_embed.permute(0, 3, 1, 2).to(dtype=target_dtype, device=target_device)

        q = self.fc_q(features).view(bs, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(features).view(bs, nq, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(features).view(bs, nq, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / math.sqrt(self.d_k)
        att = att + dist_embed
        if mask is not None:
            mask4 = mask.to(device=target_device).bool()[:, None, None, :]
            att = att.masked_fill(~mask4, -1e4)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(bs, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out, att

class SemanticOCREmbedding(nn.Module):
    def __init__(self, ns):
        super().__init__()
        self.linear_boxes = nn.Linear(4, ns.d_model)
        self.layer_norm_bboxes = nn.LayerNorm(ns.d_model)

        # Thêm Linear và LayerNorm cho det và rec features
        self.linear_det = nn.Linear(ns.d_det, ns.d_model)
        self.layer_norm_det = nn.LayerNorm(ns.d_model)

        self.linear_rec = nn.Linear(ns.d_rec, ns.d_model)
        self.layer_norm_rec = nn.LayerNorm(ns.d_model)

        # Chỉ dùng LayerNorm cho text và char (đầu vào vốn đã là d_model)
        self.layer_norm_tok = nn.LayerNorm(ns.d_model)
        self.layer_norm_char = nn.LayerNorm(ns.d_model)

    def forward(self, ocr_info, ocr_tok_features, ocr_char_features):
        W = self.linear_boxes.weight
        target_device = W.device
        target_dtype  = W.dtype

        boxes, dets, recs = [], [], []
        for info in ocr_info:
            # Boxes
            b = info["boxes"]
            if not torch.is_tensor(b): b = torch.tensor(b, device=target_device, dtype=target_dtype)
            else: b = b.to(device=target_device, dtype=target_dtype)
            boxes.append(b)

            # Det features
            d = info.get("det")
            if d is not None:
                if not torch.is_tensor(d): d = torch.tensor(d, device=target_device, dtype=target_dtype)
                else: d = d.to(device=target_device, dtype=target_dtype)
                dets.append(d)

            # Rec features
            r = info.get("rec")
            if r is not None:
                if not torch.is_tensor(r): r = torch.tensor(r, device=target_device, dtype=target_dtype)
                else: r = r.to(device=target_device, dtype=target_dtype)
                recs.append(r)

        ocr_boxes = torch.stack(boxes, dim=0)
        ocr_box_features = self.layer_norm_bboxes(self.linear_boxes(ocr_boxes))

        # Xử lý Det và Rec features (nếu có truyền vào)
        if len(dets) > 0:
            ocr_dets = torch.stack(dets, dim=0)
            ocr_det_features = self.layer_norm_det(self.linear_det(ocr_dets))
        else:
            ocr_det_features = 0.0

        if len(recs) > 0:
            ocr_recs = torch.stack(recs, dim=0)
            ocr_rec_features = self.layer_norm_rec(self.linear_rec(ocr_recs))
        else:
            ocr_rec_features = 0.0

        # Chuẩn hóa dtype & device cho text và char
        ocr_tok_features = ocr_tok_features.to(device=target_device, dtype=target_dtype)
        ocr_char_features = ocr_char_features.to(device=target_device, dtype=target_dtype)

        # Cho tok và char đi qua LayerNorm riêng
        ocr_tok_norm = self.layer_norm_tok(ocr_tok_features)
        ocr_char_norm = self.layer_norm_char(ocr_char_features)

        # Cộng các features
        combined_text_features = ocr_tok_norm + ocr_char_norm

        # Tổng hợp toàn bộ (Boxes + Det + Rec + Text + Char)
        final_features = ocr_box_features + ocr_det_features + ocr_rec_features + combined_text_features

        return final_features, combined_text_features
