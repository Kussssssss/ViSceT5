"""probe_memorization.py — pretrain PreSTU có đang ĐỌC ảnh, hay chỉ NHỚ ảnh?

Vì sao cần: pretext "sinh các từ OCR không nằm trong prefix" giải được TRỌN VẸN bằng trí
nhớ. Đo trên CHÍNH corpus pretrain (VinText 1.985 + EVJVQA 3.503 = 5.488 ảnh dùng được,
score ≥ 0.3):

  - 99,1% ảnh có "chữ ký OCR" DUY NHẤT (chữ ký = tập từ OCR của ảnh đó). Nên đường
    "ảnh → chuỗi OCR đã nhớ → trừ prefix → target" luôn đi được, không cần đọc pixel nào.
  - Riêng CHUỖI PREFIX đã đủ định danh duy nhất một ảnh trong 81,5% lượt bốc
    (VinText 88,7% | EVJVQA 77,2%). Theo độ dài prefix: 34,7% ở 1-2 từ, 91,4% ở 3-5 từ,
    97,5% ở 6-10 từ, 99,7% khi dài hơn. Ảnh trung bình có 20,1 từ OCR.
  - Nhớ toàn bộ tốn ~401 KB văn bản, trong khi model có 291M tham số ≈ 554 MB ở bf16 —
    dư khoảng 1.400 lần.

Nên `loss_text` giảm KHÔNG chứng minh model biết đọc; nó có thể chỉ đang tra bảng
"ảnh này → chuỗi OCR này". (Tính bằng ngữ nghĩa tập hợp, bỏ qua số lần lặp của từ, nên
các tỉ lệ trên là CHẶN DƯỚI.)

Với nhánh grounding, câu hỏi sống còn không phải "có đọc được không" mà là **grounding
có tổng quát hoá sang ảnh CHƯA THẤY không** — vì đó mới là thứ chuyển giao sang finetune.
Nên ngoài EM/F1 của chuỗi sinh ra, probe còn đo `point` = pointing accuracy: argmax của
bản đồ liên quan có rơi vào đúng vùng target không. `point` cao trên train mà thấp trên
val nghĩa là bản đồ chỉ được nhớ theo ảnh, và QA-CLIP/AVF sẽ KHÔNG khá lên ở finetune.

Kịch bản đo, 4 điều kiện:

    train  (ảnh ĐÃ thấy)    train + làm mờ
    val    (ảnh CHƯA thấy)  val   + làm mờ

Làm mờ đủ mạnh để phá nét chữ nhưng GIỮ bố cục/màu — tức giữ nguyên "dấu vân" để nhận
dạng ảnh, chỉ lấy đi khả năng đọc. Nhờ vậy tách được hai giả thuyết mà một mình khoảng
cách train/val không tách nổi:

    train ≫ val                  → có memorization (đo bằng `gap`)
    train_blur ≈ train           → NHỚ chứ không ĐỌC: xoá chữ mà điểm không giảm
    val giảm mạnh khi làm mờ     → có ĐỌC thật trên ảnh chưa thấy  ← điều ta muốn

Cách chạy:
    python scripts/probe_memorization.py --checkpoint /kaggle/working/pretrain_output \
        --num_samples 200 --blur_radius 4.0
"""
import argparse
import os
import sys

import pandas as pd
import torch
from PIL import ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.ocr_config import DEFAULT_OCR_CONFIG
from data.collator import ViT5VQADataCollator
from data.dataset import ViT5VQADataset
from models.modules.ocr_encoder_feature import Vision_Encode_Ocr_Feature
from models.openvivqa_model import OpenViVQAModel


def _norm_words(s):
    return [w for w in str(s).strip().lower().split() if w]


def _scores(pred, gold):
    """Trả (exact-match, F1 theo từ). F1 quan trọng hơn EM vì target dài 1-5 từ."""
    p, g = _norm_words(pred), _norm_words(gold)
    em = float(p == g)
    if not p or not g:
        return em, float(p == g)
    common = 0
    gg = list(g)
    for w in p:
        if w in gg:
            gg.remove(w)
            common += 1
    if common == 0:
        return em, 0.0
    prec, rec = common / len(p), common / len(g)
    return em, 2 * prec * rec / (prec + rec)


@torch.no_grad()
def run_condition(model, collator, dataset, idxs, tokenizer, device, blur=0.0, bs=4):
    em_sum = f1_sum = n = 0
    pt_sum = pt_n = 0
    for s in range(0, len(idxs), bs):
        rows = [dataset[i] for i in idxs[s:s + bs]]
        batch = collator(rows)
        if blur > 0:
            # Làm mờ ẢNH GỐC rồi cho qua đúng image_processor, để pixel_values thống nhất
            # với lúc train (không tự chuẩn hoá tay, tránh lệch mean/std).
            pil = [im.filter(ImageFilter.GaussianBlur(radius=blur)) for im in batch["pil_images"]]
            batch["pil_images"] = pil
            batch["pixel_values"] = model.image_processor(
                images=pil, return_tensors="pt")["pixel_values"].to(device)
        gold_ids = batch["labels"].clone()
        gold_ids[gold_ids == -100] = tokenizer.pad_token_id
        gold = tokenizer.batch_decode(gold_ids, skip_special_tokens=True)

        dev_batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items() if v is not None}

        # (a) pointing accuracy — argmax bản đồ liên quan có nằm trong vùng target không
        tpm = dev_batch.get("target_patch_mask")
        if tpm is not None:
            fo = model(**dev_batch, return_visual_search_debug=True)
            rel = fo.get("img_relevance")
            if rel is not None and rel.shape == tpm.shape:
                valid = (tpm >= 0).all(dim=-1)          # bỏ hàng sentinel (chế độ full-OCR)
                if bool(valid.any()):
                    am = rel[valid].argmax(dim=-1)
                    hit = tpm[valid].gather(1, am.unsqueeze(1)).squeeze(1) > 0
                    pt_sum += int(hit.sum()); pt_n += int(valid.sum())

        # (b) EM/F1 của chuỗi target sinh ra
        gen_kwargs = {k: v for k, v in dev_batch.items() if k not in ("labels",)}
        out = model.generate(max_new_tokens=24, num_beams=1, **gen_kwargs)
        pred = tokenizer.batch_decode(out, skip_special_tokens=True)

        for p, g in zip(pred, gold):
            if not str(g).strip():
                continue
            em, f1 = _scores(p, g)
            em_sum += em
            f1_sum += f1
            n += 1
    return (100 * em_sum / max(n, 1)), (100 * f1_sum / max(n, 1)), \
           (100 * pt_sum / max(pt_n, 1)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="./output/pretrain")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--blur_radius", type=float, default=4.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> checkpoint: {a.checkpoint}  |  device: {device}")
    model = OpenViVQAModel.from_pretrained(a.checkpoint).to(device).eval()
    model.pretrain = True
    model.config.pretrain = True
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model.config.vit5_model_name)

    train_df = pd.read_csv(os.path.join(a.data_dir, "merged_train.csv"))
    val_df = pd.read_csv(os.path.join(a.data_dir, "merged_val.csv"))

    # Ảnh chồng lấn giữa hai split sẽ làm khoảng cách train/val vô nghĩa — kiểm trước.
    col = "image_path" if "image_path" in train_df.columns else "image_filename"
    ov = set(train_df[col]) & set(val_df[col])
    print(f">>> ảnh trùng giữa train và val: {len(ov)}"
          + ("  ← CẢNH BÁO: val bị nhiễm, khoảng cách bên dưới là chặn DƯỚI"
             if ov else "  (sạch)"))

    ocr_cfg = dict(DEFAULT_OCR_CONFIG)
    vision_ocr = Vision_Encode_Ocr_Feature(ocr_cfg)
    collator = ViT5VQADataCollator(
        tokenizer=tokenizer, image_processor=model.image_processor,
        ocr_encoder=vision_ocr, config=model.config,
        term_vocab_path="", viet_vocab_path="", eng_vocab_path="",
        dataframe=train_df, pretrain=True, debug=False)

    ds_tr, ds_va = ViT5VQADataset(train_df), ViT5VQADataset(val_df)
    g = torch.Generator().manual_seed(a.seed)
    i_tr = torch.randperm(len(ds_tr), generator=g)[:a.num_samples].tolist()
    i_va = torch.randperm(len(ds_va), generator=g)[:a.num_samples].tolist()

    res = {}
    for name, ds, idxs, blur in [
        ("train        ", ds_tr, i_tr, 0.0),
        ("train + mờ   ", ds_tr, i_tr, a.blur_radius),
        ("val (chưa thấy)", ds_va, i_va, 0.0),
        ("val + mờ     ", ds_va, i_va, a.blur_radius),
    ]:
        # Cùng seed cho mọi điều kiện: split prefix/target là NGẪU NHIÊN mỗi lần gọi
        # collator, nên không cố định seed thì 4 con số không so sánh được với nhau.
        torch.manual_seed(a.seed)
        import random as _r
        _r.seed(a.seed)
        em, f1, pt, n = run_condition(model, collator, ds, idxs, tokenizer, device,
                                      blur=blur, bs=a.batch_size)
        res[name.strip()] = (em, f1, pt)
        print(f"  {name}  EM {em:6.2f}%   F1 {f1:6.2f}%   point {pt:6.2f}%   (n={n})")

    tr, va = res["train"][1], res["val (chưa thấy)"][1]
    trb, vab = res["train + mờ"][1], res["val + mờ"][1]
    ptr, pva = res["train"][2], res["val (chưa thấy)"][2]
    gap = tr - va
    print("\n--- Kết luận (theo F1) ---")
    print(f"  khoảng cách train − val          : {gap:+.2f} điểm")
    print(f"  train tụt bao nhiêu khi làm mờ   : {tr - trb:+.2f}")
    print(f"  val   tụt bao nhiêu khi làm mờ   : {va - vab:+.2f}")
    if gap > 15:
        print("  ⚠️  Khoảng cách lớn → có memorization. Dừng sớm theo F1 trên val, "
              "đừng theo loss train.")
    if (tr - trb) < 5:
        print("  ⚠️  Xoá nét chữ mà điểm gần như không giảm → model đang NHỚ chứ không ĐỌC. "
              "Tăng augmentation ảnh, giảm số epoch.")
    if (va - vab) > 15:
        print("  ✅ Trên ảnh CHƯA THẤY, mất nét chữ thì điểm sập → model thật sự đang đọc.")

    print(f"\n--- Grounding (pointing accuracy) — thước đo quyết định cho nhánh này ---")
    print(f"  train {ptr:.2f}%   |   val (chưa thấy) {pva:.2f}%   |   chênh {ptr - pva:+.2f}")
    print(f"  ngẫu nhiên ≈ {100 * 8 / 196:.1f}% (vùng target trung bình ~8/196 ô)")
    if pva < 25:
        print("  ⚠️  Trên ảnh CHƯA THẤY, bản đồ liên quan gần như chỉ bừa → grounding KHÔNG "
              "tổng quát hoá, QA-CLIP/AVF sẽ không khá lên ở finetune. Giảm epoch, "
              "tăng augmentation ảnh, hoặc tăng lambda_ground.")
    elif ptr - pva > 25:
        print("  ⚠️  Grounding chỉ đúng trên ảnh đã thấy → bản đồ bị NHỚ theo ảnh. "
              "Dừng sớm theo chính chỉ số point trên val.")
    else:
        print("  ✅ Grounding tổng quát hoá sang ảnh chưa thấy → đây mới là cái chuyển giao "
              "được sang finetune.")


if __name__ == "__main__":
    main()
