"""GT OCR (VinText labels) loading + IoU matching to SwinTextSpotter detections.

VinText ships ground-truth scene text in `labels/gt_{id}.txt`, one line per word:
    x1,y1,x2,y2,x3,y3,x4,y4,transcription
coordinates in PIXELS (quadrilateral), `###` = illegible/do-not-care.

For PreSTU SplitOCR we want the TARGET to be the clean GT text, while the INPUT
OCR evidence (det/rec features) comes from the noisy SwinTextSpotter. The two
sources are NOT index-aligned, so we match each GT box to the best-overlapping
spotter detection by IoU. EVJVQA has no GT labels -> callers fall back to spotter.
"""
from typing import List, Tuple, Optional
import os
import torch


def _quad_to_xyxy(coords: List[float]) -> Tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def load_vintext_gt(label_path: Optional[str], width: float, height: float
                    ) -> Tuple[List[str], torch.Tensor]:
    """Đọc GT VinText -> (texts, boxes_norm_xyxy [N,4] in [0,1]), BỎ nhãn illegible '###'.
    Trả ([], zeros(0,4)) nếu thiếu file / sai định dạng."""
    if not label_path or not os.path.exists(label_path) or width <= 0 or height <= 0:
        return [], torch.zeros((0, 4), dtype=torch.float)
    texts: List[str] = []
    boxes: List[List[float]] = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for ln in f.read().splitlines():
                parts = ln.split(",")
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(x) for x in parts[:8]]
                except ValueError:
                    continue
                txt = ",".join(parts[8:]).strip()
                if txt == "" or txt == "###" or set(txt) == {"#"}:
                    continue
                x0, y0, x1, y1 = _quad_to_xyxy(coords)
                boxes.append([x0 / width, y0 / height, x1 / width, y1 / height])
                texts.append(txt)
    except Exception:
        return [], torch.zeros((0, 4), dtype=torch.float)
    if not boxes:
        return [], torch.zeros((0, 4), dtype=torch.float)
    b = torch.tensor(boxes, dtype=torch.float).clamp(0.0, 1.0)
    return texts, b


def _iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU giữa mọi cặp box [Na,4] x [Nb,4] (xyxy chuẩn hoá)."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.size(0), b.size(0)))
    x0 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y0 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x1 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y1 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).clamp(min=0)[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp(min=0)[None, :]
    union = (area_a + area_b - inter).clamp(min=1e-9)
    return inter / union


def match_gt_to_spotter(gt_boxes: torch.Tensor, sp_boxes: torch.Tensor,
                        iou_thr: float = 0.5) -> List[int]:
    """Với mỗi GT box, trả index spotter khớp tốt nhất (IoU>=thr), hoặc -1 nếu không có.
    Greedy: mỗi spotter chỉ gán cho 1 GT (IoU cao nhất) để tránh trùng."""
    n_gt = gt_boxes.size(0)
    if n_gt == 0 or sp_boxes.size(0) == 0:
        return [-1] * n_gt
    iou = _iou_matrix(gt_boxes, sp_boxes)  # [n_gt, n_sp]
    match = [-1] * n_gt
    used = set()
    # Ưu tiên các cặp IoU cao trước (greedy toàn cục).
    pairs = []
    for i in range(n_gt):
        for j in range(sp_boxes.size(0)):
            v = float(iou[i, j])
            if v >= iou_thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    for v, i, j in pairs:
        if match[i] == -1 and j not in used:
            match[i] = j
            used.add(j)
    return match
