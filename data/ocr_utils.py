"""
data/ocr_utils.py
OCR utility functions: load_any(), centroid_reading_order_concat().
"""

import os, re, glob, numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import torch

def load_any(path):
    try:
        arr = np.load(path, allow_pickle=True, mmap_mode=None)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return 'error', None

    if isinstance(arr, np.lib.npyio.NpzFile):
        return 'npz', arr
    if isinstance(arr, np.ndarray):
        if arr.dtype == object and arr.shape == ():
            try:
                obj = arr.item()
                if isinstance(obj, dict):
                    return 'dict', obj
            except Exception:
                pass
        if arr.dtype.names is not None:
            return 'struct', arr
        return 'ndarray', arr
    return 'unknown', arr

def _extract_texts_exact(kind, obj):
    if kind == 'npz' and "texts" in obj.files:
        return obj["texts"].tolist()
    if kind == 'dict' and "texts" in obj:
        v = obj["texts"]
        return v.tolist() if isinstance(v, np.ndarray) else list(v)
    if kind == 'struct' and obj.dtype.names and "texts" in obj.dtype.names:
        return obj["texts"].tolist()
    return None

def _extract_boxes_and_size(kind, obj):
    boxes = None; W = H = None
    if kind == 'npz':
        if "boxes" in obj.files: boxes = obj["boxes"]
        W = obj.get("width", obj.get("weight", None))
        H = obj.get("height", None)
    elif kind == 'dict':
        if "boxes" in obj: boxes = obj["boxes"]
        W = obj.get("width", obj.get("weight", None))
        H = obj.get("height", None)
    elif kind == 'struct':
        if obj.dtype.names and "boxes" in obj.dtype.names: boxes = obj["boxes"]
    if boxes is not None and not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)
    if boxes is not None:
        boxes = boxes.astype(np.float32)
    W = float(W) if W is not None else None
    H = float(H) if H is not None else None
    return boxes, W, H

def centroid_reading_order_concat(boxes, texts, image_size=None):
    assert boxes.shape[0] == len(texts), "Số boxes phải khớp số texts"
    b = torch.tensor(boxes, dtype=torch.float32)

    if image_size is not None:
        W, H = float(image_size[0]), float(image_size[1])
        if torch.quantile(b.abs().view(-1), 0.95) <= 1.0 + 1e-6:
            size_vec = torch.tensor([W, H, W, H], dtype=torch.float32)
            b = b * size_vec

    N = b.size(0)
    if N <= 1:
        return " ".join([t.strip() for t in texts if str(t).strip() != ""]), []

    cx = (b[:, 0] + b[:, 2]) * 0.5
    cy = (b[:, 1] + b[:, 3]) * 0.5
    y_span = (b[:, 3] - b[:, 1]).clamp(min=1.0)

    line_key = torch.round(cy / (y_span.median() * 0.8)).long()
    order = torch.argsort(line_key * 1_000_000 + cx).tolist()
    texts_sorted = [texts[i] for i in order]

    joined = " ".join([str(t).strip() for t in texts_sorted if str(t).strip() != ""])
    return joined, order

def summarize_sample_from_df(row, show_image=True):
    img_path = row['image_path']
    ocr_path = row['ocr_path']
    fname = row.get('image_filename', os.path.basename(img_path))

    print(f"\n{'='*40}")
    print(f"FILE: {fname}")
    print(f"Dataset Source: {row.get('dataset', 'Unknown')}")
    print(f"{'='*40}")

    if not ocr_path or not os.path.exists(ocr_path):
        print(f"⚠️ OCR file not found: {ocr_path}")
        if show_image and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                plt.imshow(img); plt.axis("off"); plt.show()
            except: pass
        return

    kind, obj = load_any(ocr_path)
    if kind == 'error': return

    texts = _extract_texts_exact(kind, obj)
    if not texts:
        print("(Không tìm thấy 'texts' trong file OCR)")
        return

    boxes, W, H = _extract_boxes_and_size(kind, obj)
    if boxes is None or boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != len(texts):
        print("(Lỗi boxes: thiếu hoặc không khớp số lượng texts)")
        return

    if (W is None or H is None) and os.path.exists(img_path):
        try:
            with Image.open(img_path) as im:
                Wi, Hi = im.size
            if W is None: W = float(Wi)
            if H is None: H = float(Hi)
        except Exception:
            pass

    joined, order = centroid_reading_order_concat(
        boxes=boxes,
        texts=list(texts),
        image_size=(W, H) if (W and H) else None
    )

    print(f"OCR Content (Sorted):\n{joined if joined else '(Empty)'}")

    if show_image:
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"{fname} ({len(texts)} text boxes)")
                plt.show()
            except Exception as e:
                print("⚠️ Lỗi mở ảnh:", e)
        else:
            print("⚠️ Không tìm thấy ảnh:", img_path)

# if 'final_train_df' in globals():
#     sample1 = final_train_df[final_train_df['dataset'] == 'ViTextVQA'].sample(1).iloc[0]
#     summarize_sample_from_df(sample1)

#     if 'ViOCRVQA' in final_train_df['dataset'].unique():
#         sample2 = final_train_df[final_train_df['dataset'] == 'ViOCRVQA'].sample(1).iloc[0]
#         summarize_sample_from_df(sample2)
# else:
#     print("Không tìm thấy biến 'final_train_df'. Hãy chạy cell gộp dữ liệu trước.")