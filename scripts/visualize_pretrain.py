# scripts/visualize_pretrain.py
"""
Visualizes PreSTU SplitOCR model predictions:
1. Ground-Truth: Image + Prefix BBoxes (Blue) + Suffix BBoxes (Green) + Target Text
2. Model Prediction: Image + Predicted BBoxes (Red) + Generated Text (T5 Decoder)
3. Model Attention: Visual Search / Patch Attention Heatmap overlaid on image

Can be run from CLI or imported as a function inside a Jupyter / Kaggle notebook:
    from scripts.visualize_pretrain import visualize_pretrain_samples
    visualize_pretrain_samples(checkpoint="./output/pretrain", num_samples=3)
"""

import os
import sys
import argparse
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.model_config import OpenViVQAConfig
from configs.ocr_config import DEFAULT_OCR_CONFIG
from models.openvivqa_model import OpenViVQAModel
from models.modules.ocr_encoder_feature import Vision_Encode_Ocr_Feature
from data.dataset import ViT5VQADataset
from data.collator import ViT5VQADataCollator
from utils.model_utils import safe_load_tokenizer


def find_checkpoint_weights(ckpt_dir):
    """Locate model weights in ckpt_dir or its checkpoint-* subdirectories."""
    candidates = [ckpt_dir]
    if os.path.isdir(ckpt_dir):
        subdirs = [os.path.join(ckpt_dir, d) for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
        subdirs.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0, reverse=True)
        candidates = subdirs + candidates

    for c in candidates:
        sp = os.path.join(c, "model.safetensors")
        bp = os.path.join(c, "pytorch_model.bin")
        if os.path.exists(sp):
            return c, sp, "safetensors"
        if os.path.exists(bp):
            return c, bp, "bin"
    return ckpt_dir, None, None


def find_val_csv(val_path):
    """Find validation CSV across standard project paths."""
    if val_path and os.path.exists(val_path):
        return val_path
    standard_paths = [
        val_path,
        "./output/pretrain/merged_val.csv",
        "./output/merged_val.csv",
        "/kaggle/working/pretrain_output/merged_val.csv",
        "/kaggle/working/ViSceT5/output/pretrain/merged_val.csv",
        "/kaggle/working/ViSceT5/output/merged_val.csv",
        "./output/pretrain/val.csv",
        "./output/val.csv",
        "./datasets/processed/merged_val.csv",
        "./datasets/merged_val.csv",
    ]
    for p in standard_paths:
        if p and os.path.exists(p):
            return p
    return None


def visualize_pretrain_samples(
    checkpoint="./output/pretrain",
    val_csv=None,
    sample_idx=0,
    num_samples=3,
    save_dir="./output/pretrain/visualizations",
    show_plot=True,
    device=None,
):
    """
    Core function to visualize pretraining predictions and attention.
    Callable directly inside Kaggle Notebook cells.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Visualization device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # 1. Resolve Checkpoint & Weights
    best_ckpt, weights_path, w_type = find_checkpoint_weights(checkpoint)
    print(f"Checkpoint directory: {best_ckpt}")

    tokenizer = safe_load_tokenizer(best_ckpt, local_files_only=True, use_fast=False)
    if tokenizer is None:
        tokenizer = safe_load_tokenizer("VietAI/vit5-base", use_fast=False)

    try:
        config = OpenViVQAConfig.from_pretrained(best_ckpt, local_files_only=True)
    except Exception:
        config = OpenViVQAConfig()

    model = OpenViVQAModel(config)

    if weights_path:
        print(f"Loading weights from: {weights_path}")
        if w_type == "safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
        clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        res = model.load_state_dict(clean_state_dict, strict=False)
        print(f"Loaded weights: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    else:
        print("Warning: No pretrain weights file found. Visualizing with current model weights.")

    model.pretrain = True
    model.to(device)
    model.eval()

    # 2. Resolve Validation Data
    resolved_val_csv = find_val_csv(val_csv)
    if not resolved_val_csv:
        print(f"Error: Could not locate validation CSV. Checked {val_csv} and standard paths.")
        return []

    print(f"Using validation CSV: {resolved_val_csv}")
    val_df = pd.read_csv(resolved_val_csv)

    vision_ocr = Vision_Encode_Ocr_Feature(DEFAULT_OCR_CONFIG)
    val_dataset = ViT5VQADataset(val_df)

    collator = ViT5VQADataCollator(
        tokenizer=tokenizer,
        image_processor=model.image_processor,
        ocr_encoder=vision_ocr,
        config=model.config,
        term_vocab_path="configs/data/term_vocab.txt",
        viet_vocab_path="configs/data/viet_vocab.txt",
        eng_vocab_path="",
        dataframe=val_df,
        pretrain=True
    )

    total_val = len(val_dataset)
    start_idx = min(sample_idx, max(0, total_val - 1))
    end_idx = min(start_idx + num_samples, total_val)

    print(f"\nProcessing {end_idx - start_idx} samples (indices {start_idx} to {end_idx - 1})...\n")

    saved_figures = []

    for s_i in range(start_idx, end_idx):
        sample = val_dataset[s_i]
        batch = collator([sample])
        pil_img = batch["pil_images"][0]
        W0, H0 = pil_img.size

        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        inputs["return_visual_search_debug"] = True

        with torch.no_grad():
            outputs = model(**inputs)
            enc_out = outputs["encoder_outputs"]
            fused_mask = outputs["attention_mask"]
            gen_ids = model.vit5.generate(
                encoder_outputs=enc_out,
                attention_mask=fused_mask,
                max_new_tokens=40,
                num_beams=2,
                do_sample=False
            )
            pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        input_prompt = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True).strip()
        lbl_ids = batch["labels"][0].clone()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id or 0
        target_text = tokenizer.decode(lbl_ids, skip_special_tokens=True).strip()

        # Prefix BBoxes
        p_mask = batch["prefix_box_mask"][0].bool()
        p_boxes = batch["prefix_box_coords"][0][p_mask].cpu().numpy()

        # Target Suffix BBoxes. Số bin lấy từ config chứ KHÔNG hard-code: nó là tham số
        # (num_bbox_bins, mặc định 200) — hard-code 1000 làm hàm này chết ngay khi đổi.
        n_bins = int(getattr(model.config, "num_bbox_bins", 200))
        gt_bins = batch["target_bbox_bins"][0]
        valid_bbox_mask = (gt_bins[:, 0] != -100)
        gt_boxes = (gt_bins[valid_bbox_mask].float() / float(n_bins)).cpu().numpy()

        # Predicted Suffix BBoxes (Soft-argmax)
        bbox_logits = outputs.get("bbox_logits")
        if bbox_logits is not None:
            probs = torch.softmax(bbox_logits[0], dim=-1)
            # Lấy thẳng từ shape của logits: khớp cả khi checkpoint cũ dùng số bin khác config.
            bins = torch.linspace(0.0, 1.0, probs.size(-1),
                                  device=probs.device, dtype=probs.dtype)
            pred_coords = torch.sum(probs * bins, dim=-1)
            pred_boxes = pred_coords[valid_bbox_mask.to(pred_coords.device)].cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4))

        def _rect(draw, box, W, H, **kw):
            """Vẽ một box, tự sắp lại toạ độ trước.

            bbox head dự đoán 4 toạ độ ĐỘC LẬP, không gì ép x1<=x2 hay y1<=y2, nên
            checkpoint chưa hội tụ hay sinh ra box lộn ngược và PIL ném
            "y1 must be greater than or equal to y0" — chết đúng ở bước trực quan hoá
            ngay sau khi train xong. Sắp lại rồi kẹp vào trong khung ảnh.
            """
            x1, x2 = sorted((float(box[0]) * W, float(box[2]) * W))
            y1, y2 = sorted((float(box[1]) * H, float(box[3]) * H))
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            if x2 - x1 < 1 or y2 - y1 < 1:      # box suy biến: bỏ qua thay vì nổ
                return
            draw.rectangle([x1, y1, x2, y2], **kw)

        # Panel 1: Ground-Truth
        img_gt = pil_img.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        for box in p_boxes:
            _rect(draw_gt, box, W0, H0, outline="blue", width=2)
        for box in gt_boxes:
            _rect(draw_gt, box, W0, H0, outline="green", width=3)

        # Panel 2: Prediction
        img_pred = pil_img.copy()
        draw_pred = ImageDraw.Draw(img_pred)
        for box in pred_boxes:
            _rect(draw_pred, box, W0, H0, outline="red", width=3)

        # Panel 3: Attention Heatmap
        overlay = pil_img.copy()
        vs_out = outputs.get("vs_debug")
        if vs_out is not None and "attn_grids" in vs_out and vs_out["attn_grids"] is not None:
            attn_grid = vs_out["attn_grids"][0].detach().cpu().numpy()
            attn_resized = cv2.resize(attn_grid, (W0, H0), interpolation=cv2.INTER_CUBIC)
            heat_rgb = (cm.jet(attn_resized)[..., :3] * 255).astype(np.uint8)
            heat_pil = Image.fromarray(heat_rgb)
            overlay = Image.blend(pil_img.convert("RGB"), heat_pil, alpha=0.45)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_gt)
        gt_title = f'[Ground-Truth Suffix]\n"{target_text}"\n(Green: GT Suffix Boxes, Blue: Prefix Boxes)'
        axes[0].set_title(gt_title, color="green", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(img_pred)
        pred_title = f'[Model Prediction]\n"{pred_text}"\n(Red: Predicted Suffix Boxes)'
        axes[1].set_title(pred_title, color="red", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("[Visual Focus Attention Heatmap]\n(Bright Region = Model is Looking Here)", color="blue", fontsize=10)
        axes[2].axis("off")

        fig.suptitle(f"Sample #{s_i} | Prompt: {input_prompt[:75]}...", fontsize=12, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"sample_{s_i}_eval.png")
        plt.savefig(out_path, dpi=150)
        if show_plot:
            plt.show()
        plt.close(fig)

        saved_figures.append(out_path)
        print(f"Sample #{s_i}:")
        print(f"  Prompt : {input_prompt}")
        print(f"  GT     : {target_text}")
        print(f"  Pred   : {pred_text}")
        print(f"  Image  : {out_path}\n")

    print(f"All visualizations saved to: {save_dir}")
    return saved_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PreSTU SplitOCR Predictions and Attention")
    parser.add_argument("--checkpoint", type=str, default="./output/pretrain",
                        help="Path to checkpoint directory")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Path to validation CSV (auto-detected if omitted)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of sample to visualize")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to visualize")
    parser.add_argument("--save_dir", type=str, default="./output/pretrain/visualizations",
                        help="Directory to save output figures")
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_pretrain_samples(
        checkpoint=args.checkpoint,
        val_csv=args.val_csv,
        sample_idx=args.sample_idx,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        show_plot=False
    )


if __name__ == "__main__":
    main()
