"""
training/evaluate.py
Evaluation / prediction pipeline.
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure absolute project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, gc, json, torch, numpy as np
from torch.utils.data import Subset
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    GenerationConfig,
    Seq2SeqTrainingArguments,
)

# Project imports
from configs.base_config import configure_env, OUTPUT_PATH, SEED
from configs.ocr_config import DEFAULT_OCR_CONFIG
from models import OpenViVQAModel
from models.modules import Vision_Encode_Ocr_Feature
from data import ViT5VQADataCollator, ViT5VQADataset
from data.data_loader import train_df, val_df
from utils.model_utils import safe_load_tokenizer
from training.metrics import (
    TaskSpecificTrainer,
    build_compute_metrics_finetune,
    get_model_fingerprint,
    print_consistency_check
)

# Configure environment
configure_env(output_path=OUTPUT_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_DIR   = OUTPUT_PATH+"/best_bundle"
PINNED_DIR = OUTPUT_PATH+"/pinned_best_ckpt"
EVAL_DIR       = OUTPUT_PATH+"/eval_only"
LOGS_EVAL_DIR  = OUTPUT_PATH+"/logs_eval"

def main():
    print("=" * 80)
    print("PHASE 3: RELOAD FINETUNED MODEL & EVALUATE/PREDICT")
    print("=" * 80)

    os.makedirs(EVAL_DIR,      exist_ok=True)
    os.makedirs(LOGS_EVAL_DIR, exist_ok=True)

    # Initialize OCR and datasets
    vision_ocr = Vision_Encode_Ocr_Feature(DEFAULT_OCR_CONFIG)
    val_dataset = ViT5VQADataset(val_df)

    bundle_dir = PINNED_DIR if (os.path.exists(PINNED_DIR) and os.listdir(PINNED_DIR)) else BEST_DIR
    print(f"Target checkpoint dir: {bundle_dir}")

    torch.cuda.empty_cache()
    gc.collect()

    fp_finetune_ref = None
    fp_finetune_path = os.path.join(BEST_DIR, "fingerprint_finetune.json")
    if os.path.exists(fp_finetune_path):
        with open(fp_finetune_path, "r") as f:
            fp_finetune_ref = json.load(f)
        print("Loaded fingerprint_finetune.json for consistency check.")
    else:
        print("⚠️ fingerprint_finetune.json not found. Will skip consistency check Finetune -> Reload.")

    best_model = OpenViVQAModel.from_pretrained(
        bundle_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    best_model.eval()

    if fp_finetune_ref is not None:
        fp_reloaded = get_model_fingerprint(best_model)
        print_consistency_check(fp_finetune_ref, fp_reloaded,
                                title="CHECKPOINT CONSISTENCY (Finetuned bundle -> Reloaded)")

    tok_reload = safe_load_tokenizer(BEST_DIR, local_files_only=True, use_fast=False)
    ip_reload = CLIPImageProcessor.from_pretrained(os.path.join(BEST_DIR, "image_processor"))

    best_model.image_processor = ip_reload
    # visual_search bị gỡ hẳn khi ablation tắt AVF → phải kiểm tra trước khi chạm vào.
    _vs = getattr(best_model, "visual_search", None)
    if _vs is not None:
        if hasattr(_vs, "vit_processor"):
            _vs.vit_processor = ip_reload
        elif hasattr(_vs, "processor"):
            _vs.processor = ip_reload

    best_model.pretrain = False

    gen_max_new   = int(getattr(best_model.config, "generation_max_new_tokens", 27))
    gen_num_beams = int(getattr(best_model.config, "generation_num_beams", 4))
    best_model.generation_config = GenerationConfig(
        max_new_tokens=gen_max_new,
        num_beams=gen_num_beams,
        do_sample=False,
        pad_token_id=best_model.config.pad_token_id,
        eos_token_id=best_model.config.eos_token_id,
        decoder_start_token_id=best_model.config.decoder_start_token_id,
    )

    data_collator_reload = ViT5VQADataCollator(
        tokenizer=tok_reload,
        image_processor=best_model.image_processor,
        ocr_encoder=vision_ocr,
        config=best_model.config,
        term_vocab_path="configs/data/term_vocab.txt",
        viet_vocab_path="configs/data/viet_vocab.txt",
        eng_vocab_path="",
        dataframe=train_df,
        pretrain=False,
    )
    if hasattr(data_collator_reload, "set_mode"):
        data_collator_reload.set_mode(pretrain=False, mask_prob=0.0)

    eval_args = Seq2SeqTrainingArguments(
        output_dir=EVAL_DIR,
        logging_dir=LOGS_EVAL_DIR,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        dataloader_pin_memory=True,
        predict_with_generate=True,
        remove_unused_columns=False,
        generation_max_length=gen_max_new,
        generation_num_beams=gen_num_beams,
        fp16=False,
        dataloader_num_workers=4,
        report_to=None,
    )

    compute_metrics_fn = build_compute_metrics_finetune(tok_reload)

    trainer_reload = TaskSpecificTrainer(
        model=best_model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=val_dataset,
        data_collator=data_collator_reload,
        compute_metrics=compute_metrics_fn,
        processing_class=tok_reload,
    )

    print("=" * 80)
    print("EVALUATING VAL SET (RELOADED MODEL)")
    print("=" * 80)

    val_metrics = trainer_reload.evaluate(eval_dataset=val_dataset)
    print(json.dumps(val_metrics, indent=2))

    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)

    visualize_samples(
        model=best_model, # Model đã load
        dataset=val_dataset, # Dataset kiểm tra
        collator=data_collator_reload, # Collator đã cấu hình
        tokenizer=tok_reload, # Tokenizer đã load
        device=DEVICE,
        K=3,
        seed=1,
        mode="finetune"
    )

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
from typing import Optional, List, Dict, Any

# Hàm chọn ngẫu nhiên index nhưng cố định theo seed để so sánh công bằng
def pick_consistent_indices(N, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if N <= k:
        return list(range(N))
    return sorted(np.random.choice(N, k, replace=False).tolist())

# Hàm chuyển tensor sang device
def _to_device(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else x

# Hàm phân tích kết quả MLM (Masked Language Modeling)
def _gather_mask_debug_and_sentences(
    tokenizer, tcls_scores: torch.Tensor, cmb_labels: torch.Tensor,
    mlm_q_ids: torch.Tensor, mlm_ocr_ids: torch.Tensor,
    q_len: int
) -> Dict[str, Any]:
    b = 0
    scores = tcls_scores[b]
    labels = cmb_labels[b]
    preds = scores.argmax(dim=-1)

    in_ids_q = mlm_q_ids[b]
    in_ids_ocr = mlm_ocr_ids[b]

    # Tìm các vị trí bị mask (label != -1)
    masked_idx = (labels != -1).nonzero(as_tuple=False).view(-1).tolist()
    rows = []

    for pos in masked_idx:
        tgt_id = int(labels[pos].item())
        pred_id = int(preds[pos].item())
        src = "Q" if pos < q_len else "OCR"

        in_id = -1
        if src == "Q":
            in_id = int(in_ids_q[pos].item())
        else:
            local_pos = pos - q_len
            if local_pos < len(in_ids_ocr):
                in_id = int(in_ids_ocr[local_pos].item())

        rows.append({
            "pos": pos,
            "src": src,
            "in_token": tokenizer.convert_ids_to_tokens([in_id])[0] if in_id != -1 else "N/A",
            "pred_token": tokenizer.convert_ids_to_tokens([pred_id])[0],
            "tgt_token": tokenizer.convert_ids_to_tokens([tgt_id])[0],
            "tgt_id": tgt_id,
            "pred_id": pred_id
        })

    q_masked = tokenizer.decode(in_ids_q, skip_special_tokens=True)
    o_masked = tokenizer.decode(in_ids_ocr, skip_special_tokens=True)

    return {"rows": rows, "q_masked": q_masked, "ocr_masked": o_masked}

@torch.inference_mode()
def visualize_samples(model, dataset, collator, tokenizer, device: str = "cuda", K: int = 5, seed: Optional[int] = None, mode: Optional[str] = None):
    model.eval().to(device)

    # 1. Xác định chế độ (Pretrain hay Finetune)
    is_pretrain = bool(getattr(model, "pretrain", False))
    if mode is not None:
        is_pretrain = (mode.strip().lower() == "pretrain")

    # Gán tạm thời để model chạy đúng luồng forward
    model.pretrain = is_pretrain

    # Cấu hình lại collator (để sinh mask nếu là pretrain)
    if hasattr(collator, "set_mode"):
        collator.set_mode(pretrain=is_pretrain, mask_prob=0.15 if is_pretrain else 0.0)

    print(f"[Inference] Mode: {'PRETRAIN (Masked Modeling)' if is_pretrain else 'FINETUNE (Generation)'}")

    indices = pick_consistent_indices(len(dataset), K, seed)

    for bidx, sample_idx in enumerate(indices, 1):
        print("\n" + "="*70)
        print(f"Mẫu {bidx}/{len(indices)} (Dataset index: {sample_idx})")
        print("="*70)

        # 2. Chuẩn bị batch
        raw_sample = dataset[sample_idx]
        batch = collator([raw_sample])

        batch_dev = {}
        for k, v in batch.items():
            # Loại bỏ các key không phải tensor hoặc không cần thiết đưa vào GPU
            if k not in ['ocr_info', 'pil_images', 'debug_raw_goc', 'debug_raw_rel', 'debug_action', 'debug_raw_questions', 'debug_ocr_source', 'debug_image_path']:
                 batch_dev[k] = _to_device(v, device)

        pil_list = batch.get("pil_images", [])
        pil = pil_list[0].copy().convert("RGB") if pil_list else None # Đảm bảo ảnh là RGB

        question = raw_sample.get("question", "")
        answer = raw_sample.get("answer", "")

        # 3. Các tham số chung cho cả 2 mode
        forward_kwargs = {
            "pixel_values": batch_dev.get("pixel_values"),
            "pil_images": batch.get("pil_images"),
            "ocr_info": batch.get("ocr_info"),
            "ocr_mask_token": batch_dev.get("ocr_mask_token"),
            "ocr_mask_box": batch_dev.get("ocr_mask_box"),
            "twa_ocr_char": batch_dev.get("twa_ocr_char"),
            "twa_ocr_char_mask": batch_dev.get("twa_ocr_char_mask"),
            "twa_word_ids": batch_dev.get("twa_word_ids"),
            "ocr_to_word_map": batch_dev.get("ocr_to_word_map"),
            "return_visual_search_debug": True # Bắt buộc True để lấy Attention Map
        }

        # 4. Xử lý theo từng Mode
        if is_pretrain:
            # --- PRETRAIN MODE ---
            forward_kwargs["mlm_input_ids"] = batch_dev.get("mlm_input_ids")
            forward_kwargs["cmb_text_mask_label"] = batch_dev.get("cmb_text_mask_label")
            forward_kwargs["tag_pollute"] = batch_dev.get("tag_pollute")
            forward_kwargs["o2r_labels"] = batch_dev.get("o2r_labels")
            forward_kwargs["r2o_labels"] = batch_dev.get("r2o_labels")

            out = model.forward(**forward_kwargs)

            # Hiển thị kết quả MLM
            tcls = out.get("textcls_scores")
            pcls = out.get("pollutecls_scores")

            if tcls is not None:
                mlm_input = batch_dev["mlm_input_ids"]
                q_attn = batch_dev["attention_mask"]
                q_len = q_attn.size(1)

                mlm_q_ids = mlm_input[:, :q_len]
                mlm_ocr_ids = mlm_input[:, q_len:]


                dbg_data = _gather_mask_debug_and_sentences(
                    tokenizer, tcls, batch_dev["cmb_text_mask_label"],
                    mlm_q_ids, mlm_ocr_ids, q_len
                )

                print("\n[Pretrain] Bảng phân tích MLM (Điền từ vào chỗ trống):")
                print("-" * 90)
                print(f"{'Pos':<5} | {'Src':<3} | {'Input Token':<20} | {'Target Token':<20} | {'Predict Token':<20} | {'Correct'}")
                print("-" * 90)
                for row in dbg_data["rows"]:
                    ok = "✅" if row["tgt_id"] == row["pred_id"] else "❌"
                    print(f"{row['pos']:<5} | {row['src']:<3} | {row['in_token']:<20} | {row['tgt_token']:<20} | {row['pred_token']:<20} | {ok}")
                print("-" * 90)

                if pcls is not None:
                    pollute_prob = torch.sigmoid(pcls[0]).item() if pcls.numel() == 1 else torch.sigmoid(pcls[0,0]).item()
                    pollute_tgt = batch_dev["tag_pollute"][0].item() if batch_dev.get("tag_pollute") is not None else -1
                    print(f"\n[Pretrain] Pollute/ITM Prob: {pollute_prob:.4f} | Target: {pollute_tgt}")

        else:
            # --- FINETUNE MODE ---
            forward_kwargs["input_ids"] = batch_dev.get("input_ids")
            forward_kwargs["attention_mask"] = batch_dev.get("attention_mask")

            # Sinh câu trả lời
            base_model = model.module if hasattr(model, "module") else model
            gen_out = base_model.generate(
                input_ids=forward_kwargs["input_ids"],
                attention_mask=forward_kwargs["attention_mask"],
                pixel_values=forward_kwargs["pixel_values"],
                pil_images=forward_kwargs["pil_images"],
                ocr_info=forward_kwargs["ocr_info"],
                ocr_mask_token=forward_kwargs["ocr_mask_token"],
                ocr_mask_box=forward_kwargs["ocr_mask_box"],
                twa_ocr_char=forward_kwargs["twa_ocr_char"],
                twa_ocr_char_mask=forward_kwargs["twa_ocr_char_mask"],
                twa_word_ids=forward_kwargs["twa_word_ids"],
                ocr_to_word_map=forward_kwargs["ocr_to_word_map"],
                max_new_tokens=50,
                num_beams=4
            )

            pred = tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0]
            print(f"\n❓ Câu hỏi: {question}")
            print(f"🎯 Nhãn (GT): {answer}")
            print(f"🤖 Dự đoán:  {pred}")

            # Chạy forward pass một lần nữa để lấy Attention Map (vì hàm generate không trả về)
            out = model.forward(**forward_kwargs)

        # 5. Vẽ hình (Visual Search Attention)
        if "vs_debug" in out and pil:
            vs = out["vs_debug"]
            # Lấy attention map
            attn_grid_b = vs["attn_grids"][0].detach().cpu()
            # Vẽ heatmap lên ảnh gốc
            heat_on_orig = model.visual_search._heat_on_original(pil, attn_grid_b)
            overlay = Image.blend(pil, heat_on_orig, alpha=0.55)

            # Vẽ bounding box vùng Visual Search chọn
            boxes_224 = vs.get("boxes_224")
            if boxes_224 is not None:
                # Map box từ 224x224 về kích thước gốc
                boxes_orig = model.visual_search._map_boxes_to_original(boxes_224, [pil])
                box_orig = boxes_orig[0] # Lấy box của ảnh đầu tiên
                draw = ImageDraw.Draw(overlay)
                # Vẽ khung chữ nhật màu xanh lá cây
                draw.rectangle(box_orig, outline=(0, 255, 0), width=5)

            # Hiển thị plot
            fig = plt.figure(figsize=(14, 7))
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(pil)
            ax1.axis("off")
            ax1.set_title("Original Image")

            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(overlay)
            ax2.axis("off")
            title_text = "Visual Search Attention (Pretrain)" if is_pretrain else "Visual Search Attention (Finetune)"
            ax2.set_title(title_text)

            plt.tight_layout()
            plt.show()
            plt.close(fig) # Giải phóng bộ nhớ plot

if __name__ == "__main__":
    main()
