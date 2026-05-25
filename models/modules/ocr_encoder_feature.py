"""
models/modules/ocr_encoder_feature.py
Vision_Encode_Ocr_Feature — loads raw OCR .npy features.
"""

import os
import re
import numpy as np
import torch
from torch import nn
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

class Vision_Encode_Ocr_Feature(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        ocr_cfg = config['ocr_embedding']
        self.sort_type = ocr_cfg.get('sort_type', 'top-left bottom-right')
        self.scene_text_threshold = float(ocr_cfg.get('threshold', 0.2))
        self.max_scene_text = int(ocr_cfg.get('max_scene_text', 143))
        self.d_det = int(ocr_cfg['d_det'])
        self.d_rec = int(ocr_cfg['d_rec'])
        self.remove_accents_rate = ocr_cfg.get('remove_accents_rate', 0.0)
        self.use_word_seg = ocr_cfg.get('use_word_seg', False)
        self.wh_cache: Dict[str, Tuple[float, float]] = {}

    def forward(self, images: List[str], ocr_paths: Optional[List[str]] = None):
        if ocr_paths is None:
            ocr_info = [self.load_ocr_features(img, None) for img in images]
        else:
            ocr_info = [self.load_ocr_features(img, ocr) for img, ocr in zip(images, ocr_paths)]
        return ocr_info

    @staticmethod
    def _safe_np_load(path: str) -> Dict[str, Any]:
        if not os.path.exists(path): return {}
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception:
            return {}

        if isinstance(arr, np.lib.npyio.NpzFile):
            return {k: arr[k] for k in arr.files}

        if isinstance(arr, np.ndarray):
            if arr.dtype == object and arr.ndim == 0:
                obj = arr.item()
                if isinstance(obj, dict): return obj
            elif arr.ndim == 1:
                return {'data': arr.tolist()}

        return {}

    def _get_image_size(self, image_path: str) -> Tuple[float, float]:
        if image_path in self.wh_cache:
            return self.wh_cache[image_path]
        try:
            with Image.open(image_path) as im:
                w, h = float(im.size[0]), float(im.size[1])
                self.wh_cache[image_path] = (w, h)
                return w, h
        except Exception:
            return 1.0, 1.0

    @staticmethod
    def _reading_order(boxes_px: torch.Tensor, texts: List[str]) -> Tuple[List[str], List[int]]:
        if boxes_px.numel() == 0: return [], []
        cx = (boxes_px[:, 0] + boxes_px[:, 2]) * 0.5
        cy = (boxes_px[:, 1] + boxes_px[:, 3]) * 0.5
        h  = (boxes_px[:, 3] - boxes_px[:, 1]).clamp(min=1.0)
        line_key = torch.round(cy / (h.median() * 0.8)).long()
        order = torch.argsort(line_key * 1_000_000 + cx).tolist()
        sorted_texts = [texts[i] for i in order]
        return sorted_texts, order

    @staticmethod
    def _valid_text(s: str) -> bool:
        if s is None: return False
        s = str(s).strip()
        if len(s) == 0: return False
        return True

    def load_ocr_features(self, image_path: str, ocr_path: Optional[str]) -> Dict[str, Any]:
        empty_ret = {
            "det_features": torch.zeros(0, self.d_det, dtype=torch.float32),
            "rec_features": torch.zeros(0, self.d_rec, dtype=torch.float32),
            "boxes":        torch.zeros(0, 4,          dtype=torch.float32),
            "texts":        [],
            "height":       0.0,
            "width":        0.0,
        }

        feat = self._safe_np_load(ocr_path) if ocr_path else {}

        W = float(feat.get("width", 0.0))
        H = float(feat.get("height", 0.0))

        if W <= 0 or H <= 0:
            W, H = self._get_image_size(image_path)

        empty_ret["width"] = W
        empty_ret["height"] = H

        if not feat: return empty_ret

        def to_t(x, shape_fallback):
            if x is None or len(x) == 0: return torch.zeros(shape_fallback, dtype=torch.float32)
            if isinstance(x, torch.Tensor): return x.float()
            try:
                return torch.from_numpy(np.array(x)).float()
            except:
                return torch.zeros(shape_fallback, dtype=torch.float32)

        det = to_t(feat.get("det_features"), (0, self.d_det))
        rec = to_t(feat.get("rec_features"), (0, self.d_rec))
        boxes = to_t(feat.get("boxes"), (0, 4))

        scores = feat.get("scores", [])
        if len(scores) > 0:
            scores = to_t(scores, (0,))
        else:
            scores = torch.ones(boxes.size(0), dtype=torch.float32)

        texts: List[str] = list(feat.get("texts", []))

        keep_indices = []
        for i in range(len(scores)):
            if scores[i] >= self.scene_text_threshold and self._valid_text(texts[i]):
                keep_indices.append(i)

        if not keep_indices:
            return empty_ret

        det = det[keep_indices]
        rec = rec[keep_indices]
        boxes = boxes[keep_indices]
        texts = [texts[i] for i in keep_indices]

        if boxes.numel() > 0 and boxes.max() > 1.5:
            scale = torch.tensor([W, H, W, H], dtype=torch.float32)
            boxes_norm = boxes / scale
        else:
            boxes_norm = boxes

        boxes_norm = boxes_norm.clamp(0.0, 1.0)

        boxes_px = boxes_norm * torch.tensor([W, H, W, H], dtype=torch.float32)
        texts, order = self._reading_order(boxes_px, texts)

        det = det[order]
        rec = rec[order]
        boxes_norm = boxes_norm[order]

        return {
            "det_features": det.contiguous(),
            "rec_features": rec.contiguous(),
            "boxes":        boxes_norm.contiguous(),
            "texts":        texts,
            "height":       H,
            "width":        W,
        }