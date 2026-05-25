"""
utils/debug_tools.py
Debug/inspection utilities: run_mode_inspection(), get_dummy_batch().
"""

import torch
import random
import numpy as np
import os
import collections
import shutil
import re
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

# --- HELPER ---
def get_safe_filename(text):
    clean = re.sub(r'[\\/*?:"<>|]', "", str(text))
    return clean[:30] if clean else "unknown"

# --- CONFIG ---
TEST = False
TEST_SEED = 1
TARGET_IMG_IDS = ['14703.jpg']
MAX_OCR_LENGTH_FILTER = 200
N_DEBUG_SAMPLES = 5
DEBUG_DIR = "debug_crops_output"

if TEST:
    # 1. SETUP
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)

    # Cấu hình Collator (Pretrain Mode)
    if hasattr(data_collator, "set_mode"):
        mp = float(getattr(model.config, "pretrain_mask_prob", 0.15) or 0.15)
        mask_seed = int(getattr(model.config, "pretrain_mask_seed", 42))
        data_collator.set_mode(pretrain=True, mask_prob=mp, mask_seed=mask_seed)
        print(f"✅ Collator Mode: Pretrain=True, MaskProb={mp}")

    # 2. FILTER DATASET
    final_dataset = train_dataset
    if len(TARGET_IMG_IDS) > 0:
        target_indices = []
        if hasattr(train_dataset, 'df'):
            for idx, row in train_dataset.df.iterrows():
                path = str(row['image_path'])
                name = os.path.basename(path).strip()
                if name in TARGET_IMG_IDS:
                    target_indices.append(idx)
        if len(target_indices) > 0:
            final_dataset = Subset(train_dataset, target_indices)
            N_DEBUG_SAMPLES = len(target_indices)
        else:
            TARGET_IMG_IDS = []

    # 3. DATALOADER
    debug_loader = DataLoader(
        final_dataset,
        batch_size=4,
        shuffle=(len(TARGET_IMG_IDS) == 0),
        num_workers=0,
        collate_fn=data_collator
    )

    # 4. PREPARE OUTPUT
    if os.path.exists(DEBUG_DIR):
        try: shutil.rmtree(DEBUG_DIR)
        except: pass
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"--- OUTPUT DEBUG IMAGES SAVED TO: {DEBUG_DIR} ---")

    samples_processed = 0

    # 5. MAIN LOOP
    for batch_idx, batch in enumerate(debug_loader):
        if samples_processed >= N_DEBUG_SAMPLES: break

        if "debug_raw_goc" not in batch:
            print("❌ Error: Collator debug keys missing.")
            break

        current_img_paths = batch["debug_image_path"]
        ocr_source_paths = batch["debug_ocr_source"]
        raw_goc_batch = batch["debug_raw_goc"]
        raw_rel_batch = batch["debug_raw_rel"]
        action_batch = batch["debug_action"]
        raw_q_batch = batch["debug_raw_questions"]

        # [FIX] Lấy Answer từ key debug mới
        raw_a_batch = batch.get("debug_raw_answers", [""] * len(current_img_paths))

        twa_word_ids = batch["twa_word_ids"]
        ocr_to_word_map = batch["ocr_to_word_map"]
        pil_images = batch.get("pil_images", [None]*len(current_img_paths))
        batch_ocr_info = batch.get("ocr_info", [None]*len(current_img_paths))
        mlm_input_ids = batch.get("mlm_input_ids", None)
        cmb_labels = batch.get("cmb_text_mask_label", None)

        batch_len = len(current_img_paths)

        for i in range(batch_len):
            if samples_processed >= N_DEBUG_SAMPLES: break

            curr_img_path = current_img_paths[i]
            ocr_src_path = ocr_source_paths[i]
            curr_img_name = os.path.basename(curr_img_path).strip()
            ocr_src_name = os.path.basename(ocr_src_path).strip()

            if curr_img_name != ocr_src_name: continue

            raw_str_goc = raw_goc_batch[i]
            ocr_count = len(raw_str_goc)

            if len(TARGET_IMG_IDS) == 0 and ocr_count >= MAX_OCR_LENGTH_FILTER:
                continue

            img_obj = pil_images[i]
            if img_obj is None:
                try: img_obj = Image.open(curr_img_path).convert("RGB")
                except: img_obj = None

            ocr_info = batch_ocr_info[i]
            boxes_word_level = ocr_info["boxes"] if ocr_info else None
            raw_str_rel = raw_rel_batch[i]
            actions_str = action_batch[i]
            current_map = ocr_to_word_map[i]
            current_tokens = twa_word_ids[i]

            print("\n" + "=" * 120)
            print(f"SAMPLE #{samples_processed} | IMG: {curr_img_name} | OCR LENGTH: {ocr_count}")
            print("=" * 120)

            # --- VISUALIZE ---
            if img_obj and boxes_word_level is not None:
                W, H = img_obj.size
                safe_img_name = get_safe_filename(curr_img_name)
                sample_dir = os.path.join(DEBUG_DIR, f"{samples_processed}_{safe_img_name}")
                os.makedirs(sample_dir, exist_ok=True)

                img_obj.save(os.path.join(sample_dir, "original.jpg"))
                vis_img = img_obj.copy()
                draw = ImageDraw.Draw(vis_img)

                crops_to_show = []
                print("\n[IMAGE PROCESS] Processing crops...")

                for w_idx, w_text in enumerate(raw_str_goc):
                    if w_text == tokenizer.pad_token: continue
                    if w_idx >= boxes_word_level.size(0): continue

                    token_indices = (current_map == w_idx).nonzero(as_tuple=True)[0]
                    decoded_str = ""
                    if len(token_indices) > 0:
                        tok_ids = current_tokens[token_indices]
                        valid_ids = [tid.item() for tid in tok_ids if tid >= 0]
                        decoded_str = tokenizer.decode(valid_ids, skip_special_tokens=True)

                    box_norm = boxes_word_level[w_idx].tolist()
                    x1, y1, x2, y2 = box_norm[0]*W, box_norm[1]*H, box_norm[2]*W, box_norm[3]*H
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                    try:
                        pad = 2
                        c_x1, c_y1 = max(0, x1-pad), max(0, y1-pad)
                        c_x2, c_y2 = min(W, x2+pad), min(H, y2+pad)

                        if c_x2 > c_x1 and c_y2 > c_y1:
                            crop_img = img_obj.crop((c_x1, c_y1, c_x2, c_y2))
                            safe_word = get_safe_filename(w_text)
                            crop_fname = f"{w_idx:03d}_{safe_word}.jpg"
                            crop_img.save(os.path.join(sample_dir, crop_fname))
                            display_title = f"{w_idx}: {safe_word}\n(Tok: {decoded_str})"
                            crops_to_show.append((crop_img, display_title))
                    except: pass

                vis_img.save(os.path.join(sample_dir, "visualized_full.jpg"))

                if len(crops_to_show) > 0 and len(crops_to_show) <= 20:
                    cols = 5
                    rows = math.ceil(len(crops_to_show) / cols)
                    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
                    if isinstance(axes, np.ndarray): axes = axes.flatten()
                    else: axes = [axes]
                    for idx, ax in enumerate(axes):
                        if idx < len(crops_to_show):
                            img_crop, lbl = crops_to_show[idx]
                            ax.imshow(img_crop)
                            ax.set_title(lbl, fontsize=8, color='darkblue')
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()

            # --- DATA ALIGNMENT CHECK ---
            reconstructed_goc = collections.defaultdict(list)
            reconstructed_rel = collections.defaultdict(list)
            n_goc = len(raw_str_goc)

            for w_id, w_map in zip(current_tokens.tolist(), current_map.tolist()):
                if w_map == -1 or w_id < 0: continue
                if w_map < n_goc: reconstructed_goc[w_map].append(w_id)
                else: reconstructed_rel[w_map - n_goc].append(w_id)

            print("\n[DATA ALIGNMENT CHECK]")
            print(f"{'RAW GỐC':<20} | {'DECODED (IN)':<15} || {'ACTION':<12} || {'RAW LIÊN QUAN':<15} | {'DECODED (OUT)':<15} | {'MATCH?'}")
            print("-" * 150)

            for j in range(len(raw_str_goc)):
                if j >= len(raw_str_rel): break
                r_goc = raw_str_goc[j]
                r_rel = raw_str_rel[j]
                act = actions_str[j]
                if r_goc == tokenizer.pad_token: continue

                ids_in = reconstructed_goc.get(j, [])
                ids_out = reconstructed_rel.get(j, [])
                tok_in_str = tokenizer.decode(ids_in, skip_special_tokens=True) if ids_in else ""
                tok_out_str = tokenizer.decode(ids_out, skip_special_tokens=True) if ids_out else ""
                match_status = "✅" if (str(r_goc).lower().replace(" ", "") in str(tok_in_str).lower().replace(" ", "")) else "⚠️"
                print(f"{str(r_goc):<20} | {tok_in_str:<15} || {str(act):<12} || {str(r_rel):<15} | {tok_out_str:<15} | {match_status}")

            # --- MLM & QA CHECK ---
            print(f"\n📝 [INPUT] RAW QUESTION: {raw_q_batch[i]}")

            # [FIX] In Answer từ raw batch
            print(f"📝 [TARGET] RAW ANSWER:   {raw_a_batch[i]}")

            if mlm_input_ids is not None and cmb_labels is not None:
                q_in = mlm_input_ids[i]
                q_lbl = cmb_labels[i]

                if (q_lbl != -100).any():
                    print("\n[MLM MASKING DETAIL]")
                    print(f"{'INPUT TOKEN':<25} | {'TARGET (LABEL)':<25} | {'STATUS'}")
                    print("-" * 80)

                    has_mask = False
                    for inp_id, lbl_id in zip(q_in.tolist(), q_lbl.tolist()):
                        if inp_id == tokenizer.pad_token_id: continue
                        try: inp_str = tokenizer.decode([inp_id])
                        except: inp_str = "[ERR]"

                        if lbl_id != -100:
                            has_mask = True
                            try: orig_str = tokenizer.decode([lbl_id])
                            except: orig_str = "[ERR]"
                            print(f"{inp_str:<25} | {orig_str:<25} | 👺 MASKED")
                        else:
                            print(f"{inp_str:<25} | {'(Ignore)':<25} | ✅ KEEP")

                    if not has_mask: print("(No tokens were masked in this question)")

            samples_processed += 1

    shutil.make_archive(DEBUG_DIR, 'zip', DEBUG_DIR)
    print(f"\n✅ DONE! Check folder {DEBUG_DIR} or download {DEBUG_DIR}.zip")

import torch
from torch.utils.data import DataLoader
import warnings

# Tắt cảnh báo rối mắt
warnings.filterwarnings("ignore", category=DeprecationWarning)

def run_mode_inspection(mode_name, dataset, collator, tokenizer):
    is_pretrain = (mode_name == "pretrain")

    print("\n" + "#"*80)
    print(f"🕹️  CHẾ ĐỘ KIỂM TRA: {mode_name.upper()}")
    print("#"*80)

    # 1. CẤU HÌNH COLLATOR
    # Gọi hàm set_mode để thay đổi logic bên trong collator
    if hasattr(collator, "set_mode"):
        # Pretrain: mask_prob=0.15, Finetune: mask_prob=0.0
        mp = 0.15 if is_pretrain else 0.0
        collator.set_mode(pretrain=is_pretrain, mask_prob=mp, debug=True)
        print(f"⚙️  Đã set Collator: Pretrain={is_pretrain}, Mask_Prob={mp}")
    else:
        print("⚠️ Cảnh báo: Collator không có hàm set_mode!")

    # 2. TẠO BATCH
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator, shuffle=True)

    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"❌ Lỗi tạo batch: {e}")
        return

    # 3. PHÂN TÍCH CHI TIẾT
    print(f"\n📦 Phân tích Batch ({mode_name.upper()}):")

    # --- A. KIỂM TRA LABELS (QUAN TRỌNG NHẤT) ---
    if 'labels' in batch:
        lbl_tensor = batch['labels'][0]
        # Lọc bỏ -100 (ignore index) để decode
        valid_lbl = lbl_tensor[lbl_tensor != -100]
        decoded_lbl = tokenizer.decode(valid_lbl, skip_special_tokens=True)

        print(f"\n🎯 [TARGET] LABELS:")
        print(f"   • Raw Tensor Shape: {lbl_tensor.shape}")
        print(f"   • Decoded Text: '{decoded_lbl}'")

        if is_pretrain:
            if decoded_lbl.strip() == "":
                print("   ✅ ĐÁNH GIÁ: CHUẨN. (Pretrain ẩn label để học MLM)")
            else:
                print(f"   ❓ ĐÁNH GIÁ: LẠ. (Pretrain thường không hiện text label)")
        else:
            if decoded_lbl.strip() != "":
                print("   ✅ ĐÁNH GIÁ: CHUẨN. (Finetune hiện câu trả lời)")
            else:
                print("   ❌ ĐÁNH GIÁ: LỖI. (Finetune bị mất label)")

    # --- B. KIỂM TRA INPUT MASKING ---
    print(f"\n🧩 [INPUT] MASKING & IDS:")
    # Pretrain dùng key 'mlm_input_ids', Finetune dùng 'input_ids'
    input_key = 'mlm_input_ids' if is_pretrain else 'input_ids'

    if input_key in batch:
        inp_tensor = batch[input_key][0]
        mask_id = collator.mask_token_id
        # Đếm số lượng mask token <extra_id_0>
        mask_count = (inp_tensor == mask_id).sum().item()

        print(f"   • Key sử dụng: '{input_key}'")
        print(f"   • Số lượng Mask Token: {mask_count}")

        if is_pretrain:
            if mask_count > 0:
                print("   ✅ ĐÁNH GIÁ: CHUẨN (Có che từ để học).")
            else:
                print("   ⚠️ ĐÁNH GIÁ: ÍT MASK (Có thể do câu ngắn hoặc random).")
        else:
            if mask_count == 0:
                print("   ✅ ĐÁNH GIÁ: CHUẨN (Input nguyên vẹn).")
            else:
                print("   ❌ ĐÁNH GIÁ: SAI (Finetune không nên bị che từ).")
    else:
        print(f"   ❌ Lỗi: Không tìm thấy key '{input_key}' trong batch.")

    # --- C. KIỂM TRA POLLUTION (Tráo ảnh) ---
    print(f"\n🌪️ [AUGMENTATION] POLLUTION:")
    if 'tag_pollute' in batch:
        vals = batch['tag_pollute'].tolist()
        print(f"   • Tag Pollute: {vals}")
        if is_pretrain:
            print("   ✅ ĐÁNH GIÁ: Có dữ liệu pollution (Pretrain cần cái này).")
        else:
            print("   ⚠️ ĐÁNH GIÁ: Finetune thường không dùng, nhưng có cũng không sao (miễn là 0).")
    else:
        if is_pretrain:
            print("   ❌ ĐÁNH GIÁ: THIẾU (Pretrain cần tag_pollute).")
        else:
            print("   ✅ ĐÁNH GIÁ: CHUẨN (Finetune không cần pollution).")

    # --- D. KIỂM TRA LOGIC OCR ---
    if 'ocr_to_word_map' in batch:
        print(f"\n👀 [OCR] MAPPING:")
        print(f"   • Map Shape: {batch['ocr_to_word_map'].shape}")
        print("   ✅ Collator đã xử lý OCR thành công.")

# ==============================================================================
# CHẠY THỰC NGHIỆM
# ==============================================================================

if TEST and 'train_dataset' in locals() and 'data_collator' in locals():

    # 1. KIỂM TRA PRETRAIN
    run_mode_inspection("pretrain", train_dataset, data_collator, tokenizer)

    print("\n\n" + " "*30 + "⬇️  CHUYỂN CHẾ ĐỘ  ⬇️" + "\n")

    # 2. KIỂM TRA FINETUNE
    run_mode_inspection("finetune", train_dataset, data_collator, tokenizer)

    print("\n" + "="*80)
    print("🏁 HOÀN TẤT.")
else:
    print("Cần khởi tạo dataset, collator và tokenizer trước.")

from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import math

MAX_SAMPLES = 10

if TEST:
    print("=" * 60)
    print(">>> PHASE 1: TESTING PRE-TRAINING PIPELINE")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.pretrain = True

    if hasattr(data_collator, "set_mode"):
        mp = float(getattr(model.config, "pretrain_mask_prob", 0.15) or 0.15)
        mask_seed = int(getattr(model.config, "pretrain_mask_seed", 42))
        data_collator.set_mode(pretrain=True, mask_prob=mp, mask_seed=mask_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    pretrain_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        generator=g,
        num_workers=0,
        collate_fn=data_collator,
        drop_last=False,
    )

    loss_fn = ViT5PretrainLoss().to(device)
    acc_fn = PreTrainMLMAccuracy()

    total_loss = 0.0
    total_acc = 0.0
    seen_samples = 0

    print(f"[Config] Model.pretrain={model.pretrain} | Collator.pretrain={data_collator.pretrain}")

    for step, batch in enumerate(pretrain_loader):
        if seen_samples >= MAX_SAMPLES: break

        if "mlm_input_ids" not in batch:
            print("❌ Error: 'mlm_input_ids' missing in Pretrain batch!")
            break

        batch_on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        out = model(**batch_on_device)

        if step == 0:
            print("\n[Pretrain Output Check]")
            print(f"  - MLM Scores: {out.get('textcls_scores', torch.empty(0)).shape}")
            print(f"  - Pollute Scores: {out.get('pollutecls_scores', torch.empty(0)).shape}")
            contra = out.get('contrastive_scores')
            print(f"  - Contrastive Scores: {contra.shape if contra is not None else 'None'}")

        model.zero_grad(set_to_none=True)
        loss = loss_fn(batch_on_device, out)
        acc = acc_fn(batch_on_device, out)

        loss.backward()

        total_loss += loss.item()
        total_acc += acc.item()
        seen_samples += batch_on_device["mlm_input_ids"].size(0)

        print(f"  Batch {step}: Loss={loss.item():.4f} | Acc={acc.item():.4f}")

    print(f"\n>>> Pretrain Test Done. Avg Loss: {total_loss/(step+1):.4f}\n")

    print("=" * 60)
    print(">>> PHASE 2: TESTING FINETUNE/GENERATION PIPELINE")
    print("=" * 60)

    model.pretrain = False

    if hasattr(data_collator, "set_mode"):
        data_collator.set_mode(pretrain=False, mask_prob=0.0)

    print(f"[Config] Model.pretrain={model.pretrain} | Collator.pretrain={data_collator.pretrain}")

    finetune_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        generator=g,
        num_workers=0,
        collate_fn=data_collator,
        drop_last=False,
    )

    seen_samples = 0
    for step, batch in enumerate(finetune_loader):
        if seen_samples >= 4: break

        if "input_ids" not in batch:
            print("❌ Error: 'input_ids' missing in Finetune batch!")
            break

        batch_on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        if step == 0:
            print("\n[Finetune Forward Check]")
            out = model(**batch_on_device)
            print(f"  - Loss: {out.get('loss', 'None')}")
            print(f"  - Logits: {out.get('logits', torch.empty(0)).shape}")

        print(f"\n[Generation Sample Batch {step}]")
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=batch_on_device["input_ids"],
                pixel_values=batch_on_device["pixel_values"],
                attention_mask=batch_on_device["attention_mask"],
                pil_images=batch_on_device["pil_images"],
                ocr_info=batch_on_device["ocr_info"],
                ocr_mask_token=batch_on_device["ocr_mask_token"],
                ocr_mask_box=batch_on_device["ocr_mask_box"],
                twa_ocr_char=batch_on_device["twa_ocr_char"],
                twa_ocr_char_mask=batch_on_device["twa_ocr_char_mask"],
                twa_word_ids=batch_on_device["twa_word_ids"],
                ocr_to_word_map=batch_on_device["ocr_to_word_map"],
                max_new_tokens=20
            )

        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        for i, text in enumerate(decoded):
            print(f"  - Generated [{i}]: {text}")

        seen_samples += batch_on_device["input_ids"].size(0)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
    print("=" * 60)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def test_vqa_collator_visual(data_collator, dataset, num_samples: int = 1,
                             show_max_boxes: int = 10, show_image: bool = True,
                             print_raw: bool = True):
    # Lấy mẫu
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]

    # Auto-detect mode
    is_pretrain = getattr(data_collator, "pretrain", False)
    print(f"\nRunning Test with Collator Mode: {'PRETRAIN' if is_pretrain else 'FINETUNE'}")

    batch = data_collator(samples)
    tokenizer = data_collator.tokenizer

    print("\n========== [VISUAL OCR BOX–TOKEN ALIGNMENT TEST (FIXED)] ==========")

    for idx in range(len(samples)):
        info = batch["ocr_info"][idx]
        image = batch["pil_images"][idx]

        twa_word_ids = batch["twa_word_ids"][idx]
        ocr_to_word_map = batch["ocr_to_word_map"][idx]

        # [FIX 1] Chọn đúng nguồn Box
        # Trong Pretrain, Collator tạo ra 'boxes_word_all' (gấp đôi số box gốc)
        if "boxes_word_all" in info:
            boxes_source = info["boxes_word_all"]
            source_type = "Augmented Boxes (boxes_word_all)"
        else:
            boxes_source = info["boxes"]
            source_type = "Original Boxes (boxes)"

        # Determine text source
        raw_texts_debug = batch.get("debug_raw_goc", None)
        if raw_texts_debug is not None and len(raw_texts_debug) > idx:
            raw_texts = raw_texts_debug[idx]
            mode_display = "Pretrain (TWA Noisy)"
        else:
            raw_texts = info["texts"]
            mode_display = "Finetune (Clean)"

        W_img, H_img = image.size

        print(f"🟩 Sample {idx+1}")
        print(f"  Mode:              {mode_display}")
        print(f"  Box Source:        {source_type}")
        print(f"  Box Dim:           {boxes_source.shape}")
        print(f"  Tokens Dim:        {twa_word_ids.shape}")
        print(f"  Image size:        {W_img}x{H_img}")
        print("-----------------------------------------------------------")

        vis_boxes, vis_labels = [], []

        # Số lượng từ OCR thực tế (Word Level)
        # Lưu ý: raw_texts chỉ là 1 nửa (phần gốc) trong pretrain,
        # nhưng twa_word_ids chứa cả 2 phần. Ta chỉ visualize phần đầu.
        n_words_to_show = min(len(raw_texts), show_max_boxes)

        for word_idx in range(n_words_to_show):
            text_label = raw_texts[word_idx]

            if text_label == tokenizer.pad_token: continue

            # Tìm các sub-token thuộc về từ này
            # ocr_to_word_map map từ Token Index -> Word Index
            subword_indices = (ocr_to_word_map == word_idx).nonzero(as_tuple=True)[0]

            if len(subword_indices) == 0:
                print(f"[{word_idx:02d}] '{text_label}' - No tokens (Truncated/Filtered)")
                continue

            token_ids = twa_word_ids[subword_indices]
            # Safe decode (bỏ qua -100 nếu có)
            valid_ids = [tid.item() for tid in token_ids if tid >= 0]
            decoded_subwords = tokenizer.decode(valid_ids, skip_special_tokens=True)

            # [FIX 2] Lấy Box dựa trên WORD INDEX (word_idx), KHÔNG PHẢI token index
            if word_idx < boxes_source.size(0):
                box_norm = boxes_source[word_idx] # <--- FIX QUAN TRỌNG Ở ĐÂY

                x1, y1, x2, y2 = box_norm.tolist()
                x1, x2 = x1 * W_img, x2 * W_img
                y1, y2 = y1 * H_img, y2 * H_img

                vis_boxes.append([x1, y1, x2, y2])
                vis_labels.append(f"{word_idx}:{text_label}")

                if print_raw:
                    print(f"[{word_idx:02d}] Word: '{text_label}'")
                    print(f"      ↳ Subwords: {tokenizer.convert_ids_to_tokens(valid_ids)}")
                    print(f"      ↳ Box:      [{box_norm[0]:.3f}, {box_norm[1]:.3f}, ...]")
            else:
                print(f"[{word_idx:02d}] '{text_label}' - Box Index Out of Bounds")

        print("===========================================================\n")

        if show_image:
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(image)
            ax.set_title(f"Sample {idx+1}: {mode_display}\nYellow: Word-Level Box Alignment Check")

            for i, (box, label) in enumerate(zip(vis_boxes, vis_labels)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                # Vẽ Box
                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none"
                )
                ax.add_patch(rect)

                # Vẽ Label (Word Index : Text)
                ax.text(x1, max(0, y1 - 5), label,
                        color="yellow", fontsize=9, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))

            ax.set_xlim([0, W_img]); ax.set_ylim([H_img, 0])
            plt.axis("off"); plt.tight_layout()
            plt.show()

    print("✅ Visualization completed.\n")
    return batch

# --- CHẠY TEST ---
if TEST:
    # Test Pretrain Mode
    if hasattr(data_collator, "set_mode"):
        data_collator.set_mode(pretrain=True, mask_prob=0.15)
    print(">>> TESTING PRETRAIN MODE (Visualizing Augmented Boxes)...")
    _ = test_vqa_collator_visual(data_collator, train_dataset, num_samples=4, show_max_boxes=10)

    # Test Finetune Mode
    if hasattr(data_collator, "set_mode"):
        data_collator.set_mode(pretrain=False, mask_prob=0.0)
    print(">>> TESTING FINETUNE MODE (Visualizing Original Boxes)...")
    _ = test_vqa_collator_visual(data_collator, train_dataset, num_samples=4, show_max_boxes=10)

import torch
import os
import textwrap
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

# --- CONFIG TÙY CHỈNH ---
# Điền tên file ảnh bạn muốn kiểm tra (ví dụ '14703.jpg').
# Code sẽ tìm TẤT CẢ các câu hỏi liên quan đến ảnh này trong train_dataset.
TARGET_IMAGE_ID = "14703.jpg"
# -----------------------

def pick_consistent_indices(N, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if N <= k:
        return list(range(N))
    return sorted(np.random.choice(N, k, replace=False).tolist())

def _ensure_tokenizer(cfg_like):
    name = getattr(cfg_like, "vit5_name", None) or "VietAI/vit5-base"
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return AutoTokenizer.from_pretrained("VietAI/vit5-base")

def _to_device(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else x

def eval_visualsearch_mode(model, batch, tokenizer, mode_name="", question_text=""):
    """
    Hàm thực thi model và visualize Attention Map.
    """
    device = next(model.parameters()).device
    model.eval()

    pixel_values = _to_device(batch["pixel_values"], device)

    # Xử lý input (Input IDs hoặc MLM IDs)
    if "input_ids" in batch:
        input_ids = _to_device(batch["input_ids"], device)
        attn_mask = _to_device(batch.get("attention_mask", None), device)
        mlm_input_ids = None
        effective_input = input_ids
    elif "mlm_input_ids" in batch:
        input_ids = None
        mlm_input_ids = _to_device(batch["mlm_input_ids"], device)
        attn_mask = _to_device(batch.get("attention_mask", None), device)
        effective_input = mlm_input_ids
    else:
        input_ids = None; mlm_input_ids = None; attn_mask = None; effective_input = None

    pil_images = batch.get("pil_images", None) or []
    ocr_info = batch.get("ocr_info", None)

    # Move các tensor phụ trợ sang device
    ocr_mask_tok = _to_device(batch.get("ocr_mask_token", None), device)
    ocr_mask_box = _to_device(batch.get("ocr_mask_box", None), device)
    twa_word_ids = _to_device(batch.get("twa_word_ids", None), device)
    twa_ocr_char = _to_device(batch.get("twa_ocr_char", None), device)
    twa_ocr_char_mask = _to_device(batch.get("twa_ocr_char_mask", None), device)
    ocr_to_word_map = _to_device(batch.get("ocr_to_word_map", None), device)
    cmb_text_mask_label = _to_device(batch.get("cmb_text_mask_label", None), device)
    tag_pollute = _to_device(batch.get("tag_pollute", None), device)
    o2r_labels = _to_device(batch.get("o2r_labels", None), device)
    r2o_labels = _to_device(batch.get("r2o_labels", None), device)

    # Forward Pass
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=None,
            pixel_values=pixel_values,
            pil_images=pil_images,
            ocr_info=ocr_info,
            ocr_mask_token=ocr_mask_tok,
            ocr_mask_box=ocr_mask_box,
            twa_word_ids=twa_word_ids,
            twa_ocr_char=twa_ocr_char,
            twa_ocr_char_mask=twa_ocr_char_mask,
            ocr_to_word_map=ocr_to_word_map,
            mlm_input_ids=mlm_input_ids,
            cmb_text_mask_label=cmb_text_mask_label,
            tag_pollute=tag_pollute,
            o2r_labels=o2r_labels,
            r2o_labels=r2o_labels,
            return_visual_search_debug=True, # Yêu cầu trả về Attention Map
        )

    # --- VISUALIZATION LOGIC ---
    if "vs_debug" in out:
        vs = out["vs_debug"]
        b = 0 # Visualize sample đầu tiên trong batch

        if vs.get("attn_grids") is not None:
            attn_grid_b = vs["attn_grids"][b].detach().cpu()

            if len(pil_images) > b:
                pil = pil_images[b].copy()
                boxes_224 = vs["boxes_224"]

                # Map lại box về ảnh gốc để vẽ
                boxes_orig_list = model.visual_search._map_boxes_to_original(boxes_224, pil_images)
                box_orig = boxes_orig_list[b]

                # Tạo Heatmap overlay
                heat_on_orig = model.visual_search._heat_on_original(pil, attn_grid_b)
                overlay = Image.blend(pil, heat_on_orig, alpha=0.6)

                # Vẽ Box tập trung nhất
                draw = ImageDraw.Draw(overlay)
                if box_orig is not None:
                    draw.rectangle(box_orig, outline=(0, 255, 0), width=5)

                # Format Title
                wrapped_q = "\n".join(textwrap.wrap(question_text, width=50))
                title_str = f"[{mode_name}]\nQ: {wrapped_q}"

                # Plot
                plt.figure(figsize=(14, 6))
                plt.subplot(1, 2, 1); plt.imshow(pil); plt.axis("off"); plt.title("Original Image", fontsize=10)
                plt.subplot(1, 2, 2); plt.imshow(overlay); plt.axis("off"); plt.title(title_str, fontsize=10, color='darkblue', weight='bold')
                plt.tight_layout(); plt.show()

    return out

if TEST:
    print("=" * 80)
    print(f"🚀 VISUAL SEARCH DEBUGGER: TARGET '{TARGET_IMAGE_ID}'")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer = _ensure_tokenizer(getattr(model, "config", None))

    indices = []

    # --- 1. TÌM KIẾM TOÀN BỘ CÁC SAMPLES CỦA ẢNH TRONG TRAIN_DATASET ---
    if TARGET_IMAGE_ID:
        print(f"🔎 Scanning TRAIN dataset for ALL samples containing: {TARGET_IMAGE_ID}...")

        if hasattr(train_dataset, 'df'):
            for idx, row in train_dataset.df.iterrows():
                path = str(row['image_path'])
                filename = os.path.basename(path).strip()
                if TARGET_IMAGE_ID in filename:
                    indices.append(idx)
        else:
            print("   (Manual scan...)")
            for idx in range(len(train_dataset)):
                sample = train_dataset[idx]
                path = sample.get("image_path", "")
                filename = os.path.basename(path).strip()
                if TARGET_IMAGE_ID in filename:
                    indices.append(idx)

        if len(indices) > 0:
            print(f"✅ Found {len(indices)} samples associated with '{TARGET_IMAGE_ID}'.")
        else:
            print(f"⚠️ Image '{TARGET_IMAGE_ID}' not found in Train Set. Switching to Random Mode.")

    if not indices:
        indices = pick_consistent_indices(len(train_dataset), 2, seed=999)
        print(f"🎲 Random Indices: {indices}")

    # --- 2. THỰC THI VÒNG LẶP TRÊN TỪNG SAMPLE ---
    for i, idx in enumerate(indices, 1):
        sample = train_dataset[idx]
        question = sample['question']
        answer = sample.get('answer', 'N/A')

        print(f"\n" + "-"*80)
        print(f"📸 SAMPLE #{i}/{len(indices)} (Index: {idx})")
        print(f"❓ Question: {question}")
        print(f"✅ GT Answer: {answer}")
        print("-" * 80)

        # Chúng ta chỉ cần chạy chế độ Finetune để xem Attention Map rõ nhất
        # (Vì Pretrain có Masking nên Attention có thể bị nhiễu do token <extra_id_0>)
        print(f"\n>>> VISUALIZING ATTENTION MAP (Clean Input)")
        model.pretrain = False
        if hasattr(data_collator, "set_mode"):
            data_collator.set_mode(pretrain=False, mask_prob=0.0)

        batch_ft = data_collator([sample])
        _ = eval_visualsearch_mode(model, batch_ft, tokenizer, mode_name="ATTENTION CHECK", question_text=question)

        # Thử sinh câu trả lời
        batch_on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_ft.items()}
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=batch_on_device["input_ids"],
                pixel_values=batch_on_device["pixel_values"],
                attention_mask=batch_on_device.get("attention_mask"),
                pil_images=batch_on_device.get("pil_images"),
                ocr_info=batch_on_device.get("ocr_info"),
                ocr_mask_token=batch_on_device.get("ocr_mask_token"),
                ocr_mask_box=batch_on_device.get("ocr_mask_box"),
                twa_ocr_char=batch_on_device.get("twa_ocr_char"),
                twa_ocr_char_mask=batch_on_device.get("twa_ocr_char_mask"),
                twa_word_ids=batch_on_device.get("twa_word_ids"),
                ocr_to_word_map=batch_on_device.get("ocr_to_word_map"),
                max_new_tokens=20,
                num_beams=3,
            )
        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        print(f"🤖 Model Prediction: {decoded[0]}")

    print("\n" + "=" * 80)
    print("DONE ALL SAMPLES.")

def get_dummy_batch(config, device, mode="pretrain"):
    img_size = int(getattr(config, "vs_target_size", 224))
    seq_len = int(getattr(config, "text_max_input_length", 32))
    ocr_max_scene_text = int(getattr(config, "ocr_max_scene_text", 200))
    char_num_vocab = int(getattr(config, "char_num", 3000))
    char_max_num_per_word = int(getattr(config, "char_max_num", 50))

    dummy_num_words = 16

    vocab_size = 32000
    batch_size = 2

    inputs = {
        "pixel_values": torch.randn(batch_size, 3, img_size, img_size, device=device),
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),

        "twa_word_ids": torch.randint(0, vocab_size, (batch_size, ocr_max_scene_text), device=device),
        "twa_ocr_char": torch.randint(0, char_num_vocab, (batch_size, dummy_num_words, char_max_num_per_word), device=device),
        "twa_ocr_char_mask": torch.ones(batch_size, dummy_num_words, char_max_num_per_word, device=device),
        "ocr_mask_box": torch.zeros(batch_size, ocr_max_scene_text, device=device),
        "ocr_mask_token": torch.zeros(batch_size, ocr_max_scene_text, device=device),
    }

    inputs["ocr_mask_box"][:, :dummy_num_words] = 1
    inputs["ocr_mask_token"][:, :dummy_num_words] = 1

    ocr_to_word_map = torch.full((batch_size, ocr_max_scene_text), -1, dtype=torch.long, device=device)
    for i in range(batch_size):
        limit = min(ocr_max_scene_text, dummy_num_words)
        ocr_to_word_map[i, :limit] = torch.arange(limit, device=device)
    inputs["ocr_to_word_map"] = ocr_to_word_map

    inputs["ocr_info"] = [{
        "width": 800, "height": 600,
        "boxes_word_all": torch.rand(dummy_num_words, 4, device=device),
        "word_mask_all": torch.ones(dummy_num_words, device=device)
    } for _ in range(batch_size)]

    if mode == "pretrain":
        inputs["mlm_input_ids"] = inputs["input_ids"].clone()
        inputs["cmb_text_mask_label"] = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        inputs["tag_pollute"] = torch.tensor([0, 1], device=device).float()

        half_words = dummy_num_words // 2
        inputs["o2r_labels"] = torch.eye(half_words, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        inputs["labels"] = None
    else:
        # Finetune
        inputs["mlm_input_ids"] = None
        inputs["cmb_text_mask_label"] = None
        inputs["tag_pollute"] = None
        inputs["o2r_labels"] = None
        inputs["labels"] = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Filter None
    return {k: v for k, v in inputs.items() if v is not None}

def inspect_model_mode(model, mode="pretrain"):
    is_pretrain = (mode == "pretrain")

    print("\n" + "="*80)
    print(f"🧪  ĐANG KIỂM TRA CHẾ ĐỘ: {mode.upper()}")
    print("="*80)

    # 1. Setup Model State
    model.train()
    model.zero_grad()
    model.pretrain = is_pretrain # Override flag trong model

    # 2. Forward & Loss Calculation
    try:
        inputs = get_dummy_batch(model.config, model.device, mode=mode)
        outputs = model(**inputs)

        loss = None
        if is_pretrain:
            # Dùng Loss Class thực tế
            criterion = ViT5PretrainLoss()
            # inputs đóng vai trò là sample_list
            loss = criterion(inputs, outputs)
            print(f"✅ Pretrain Loss (Combined): {loss.item():.5f}")
        else:
            # Finetune: Loss nằm sẵn trong output (thường là CrossEntropy của Decoder)
            loss = outputs.get("loss", None)
            if loss is None:
                 print("❌ Lỗi: Không tìm thấy 'loss' trong output finetune!")
                 return
            print(f"✅ Finetune Loss (Generation): {loss.item():.5f}")

        # 3. Backward
        loss.backward()
        print("✅ Backward thành công.")

    except Exception as e:
        print(f"❌ CRITICAL RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n📊  KẾT QUẢ KIỂM TRA GRADIENT:")
    print("-" * 80)

    allowed_dead = []
    if not is_pretrain:
        allowed_dead = ["pollute_head", "logit_scale"]

    missing_grads = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                # Kiểm tra xem có được phép dead không
                is_expected = any(k in name for k in allowed_dead)

                if not is_expected:
                    severity = "🔴 NGHIÊM TRỌNG (Đứt kết nối)"
                    if "bias" in name and "LayerNorm" in name:
                        severity = "🟡 Cảnh báo nhẹ (LN bias)"
                    missing_grads.append((name, severity))

    if len(missing_grads) == 0:
        print("🎉 TUYỆT VỜI! Tất cả modules cần thiết đều Healthy.")
    else:
        print(f"{'PARAM NAME':<55} | {'STATUS'}")
        print("-" * 80)
        for name, status in missing_grads:
            print(f"{name:<55} | {status}")

    # 5. Module Health Summary
    print("\n🏥  TỔNG QUAN MODULE:")
    modules_to_check = ["vit5", "ocr_encoder", "visual_search", "qa_clip", "pollute_head"]

    for mod_name in modules_to_check:
        if hasattr(model, mod_name):
            mod = getattr(model, mod_name)
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            with_grad = sum(p.numel() for p in mod.parameters() if p.requires_grad and p.grad is not None)

            if trainable == 0:
                status = "❄️ FROZEN"
            elif with_grad == trainable:
                status = "✅ HEALTHY"
            elif with_grad == 0:
                # Nếu finetune mà pollute_head chết thì là OK
                if not is_pretrain and mod_name in ["pollute_head"]:
                    status = "💤 IDLE (Expected)"
                else:
                    status = "❌ DEAD"
            else:
                status = f"⚠️ PARTIAL ({with_grad}/{trainable})"

            print(f"   • {mod_name:<20}: {status}")
if TEST:
  gc.collect()
  torch.cuda.empty_cache()

  if 'model' in locals():
      # Lưu lại cờ gốc
      orig_pretrain = getattr(model, "pretrain", True)

      # 1. TEST PRETRAIN
      inspect_model_mode(model, mode="pretrain")

      print("\n" + " "*30 + "⬇️  CHUYỂN CHẾ ĐỘ  ⬇️" + "\n")

      # 2. TEST FINETUNE
      inspect_model_mode(model, mode="finetune")

      # Khôi phục
      model.pretrain = orig_pretrain
      print("\n🏁  HOÀN TẤT KIỂM TRA TOÀN DIỆN.")
  else:
      print("Vui lòng khởi tạo biến 'model' trước khi chạy.")