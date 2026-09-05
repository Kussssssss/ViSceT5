# scripts/visualize_pretrain.py
"""
Visualizes PreSTU SplitOCR model predictions:
1. Ground-Truth: Image + GT Suffix BBoxes (Green) + Target Text
2. Model Prediction: Image + Predicted BBoxes (Red) + Generated Text (T5 Decoder)
3. Model Attention: Visual Search / Patch Attention Heatmap overlaid on image
"""

import os
import sys
import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PreSTU SplitOCR Predictions and Attention")
    parser.add_argument("--checkpoint", type=str, default="./output/pretrain",
                        help="Path to checkpoint directory")
    parser.add_argument("--val_csv", type=str, default="./output/pretrain/merged_val.csv",
                        help="Path to merged validation CSV")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of sample to visualize")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to visualize")
    parser.add_argument("--save_dir", type=str, default="./output/pretrain/visualizations",
                        help="Directory to save output figures")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_dir = args.checkpoint
    print(f"Loading checkpoint from: {ckpt_dir}")

    tokenizer = safe_load_tokenizer(ckpt_dir, local_files_only=True, use_fast=False)
    if tokenizer is None:
        tokenizer = safe_load_tokenizer("VietAI/vit5-base", use_fast=False)

    try:
        config = OpenViVQAConfig.from_pretrained(ckpt_dir, local_files_only=True)
    except Exception:
        config = OpenViVQAConfig()

    model = OpenViVQAModel(config)

    from safetensors.torch import load_file
    safe_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    state_dict = None
    if os.path.exists(safe_path):
        state_dict = load_file(safe_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.exists(ckpt_dir):
        subdirs = [os.path.join(ckpt_dir, d) for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
        if subdirs:
            subdirs.sort(key=lambda x: int(x.split("-")[-1]))
            latest_ckpt = subdirs[-1]
            print(f"Found latest checkpoint subdir: {latest_ckpt}")
            sp = os.path.join(latest_ckpt, "model.safetensors")
            bp = os.path.join(latest_ckpt, "pytorch_model.bin")
            state_dict = load_file(sp) if os.path.exists(sp) else (torch.load(bp, map_location="cpu") if os.path.exists(bp) else None)

    if state_dict is not None:
        clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        res = model.load_state_dict(clean_state_dict, strict=False)
        print(f"Loaded weights: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    else:
        print("Warning: No weights found in checkpoint path.")

    model.pretrain = True
    model.to(device)
    model.eval()

    val_csv = args.val_csv
    if not os.path.exists(val_csv):
        for alt in ["./output/pretrain/val.csv", "./datasets/processed/merged_val.csv"]:
            if os.path.exists(alt):
                val_csv = alt
                break

    if not os.path.exists(val_csv):
        print(f"Error: Validation CSV not found at {args.val_csv}")
        return

    val_df = pd.read_csv(val_csv)
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
    start_idx = min(args.sample_idx, max(0, total_val - 1))
    end_idx = min(start_idx + args.num_samples, total_val)

    print(f"Visualizing {end_idx - start_idx} samples (indices {start_idx} to {end_idx - 1})...\n")

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

        gt_bins = batch["target_bbox_bins"][0]
        valid_bbox_mask = (gt_bins[:, 0] != -100)
        gt_boxes = (gt_bins[valid_bbox_mask].float() / 1000.0).cpu().numpy()

        bbox_logits = outputs.get("bbox_logits")
        if bbox_logits is not None:
            probs = torch.softmax(bbox_logits[0], dim=-1)
            bins = torch.linspace(0.0, 1.0, 1000, device=probs.device, dtype=probs.dtype)
            pred_coords = torch.sum(probs * bins, dim=-1)
            pred_boxes = pred_coords[valid_bbox_mask].cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4))

        img_gt = pil_img.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        for box in gt_boxes:
            x1, y1, x2, y2 = box[0] * W0, box[1] * H0, box[2] * W0, box[3] * H0
            draw_gt.rectangle([x1, y1, x2, y2], outline="green", width=3)

        img_pred = pil_img.copy()
        draw_pred = ImageDraw.Draw(img_pred)
        for box in pred_boxes:
            x1, y1, x2, y2 = box[0] * W0, box[1] * H0, box[2] * W0, box[3] * H0
            draw_pred.rectangle([x1, y1, x2, y2], outline="red", width=3)

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
        axes[0].set_title(f"[GT Suffix Text]\n\"{target_text}\"\n(Green = GT Suffix Boxes)", color="green", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(img_pred)
        axes[1].set_title(f"[Model Predicted Suffix Text]\n\"{pred_text}\"\n(Red = Predicted Suffix Boxes)", color="red", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"[Visual Focus Attention Map]\n(Bright = Region model focuses on)", color="blue", fontsize=11)
        axes[2].axis("off")

        fig.suptitle(f"Sample #{s_i} | Prompt: {input_prompt[:80]}...", fontsize=12, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(args.save_dir, f"sample_{s_i}_eval.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to: {out_path}")
        print(f"  Prompt : {input_prompt}")
        print(f"  GT     : {target_text}")
        print(f"  Pred   : {pred_text}\n")

    print(f"All visualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
