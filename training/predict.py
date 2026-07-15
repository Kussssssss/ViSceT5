"""
training/predict.py
Sinh file nộp `ID,Answer` cho leaderboard ViTextVQA (chấm Exact Match).

- Load bundle đã finetune (OpenViVQAModel + tokenizer + image_processor).
- Chạy beam-search generation trên split dev (validation) hoặc test.
- Ghi output/submission_{split}.csv với header `ID,Answer`, trong đó ID là
  annotation id (question id) lấy TRỰC TIẾP từ file JSON gốc của ViTextVQA
  (không dựa vào thứ tự CSV vì CSV đã bị reorder + có cặp (image_id,question) trùng).

Cách dùng:
    python -m training.predict --split dev
    python -m training.predict --split test --ckpt_dir output/finetune
    python -m training.predict --split both --batch_size 8 --num_beams 4

Định dạng nộp (đúng theo overview):
    ID,Answer
    2,tôi yêu bạn
    5,anh yêu em
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gc
import json
import argparse
import unicodedata
import pandas as pd
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, GenerationConfig

from configs.base_config import configure_env, OUTPUT_PATH
from configs.ocr_config import DEFAULT_OCR_CONFIG
from models import OpenViVQAModel
from models.modules import Vision_Encode_Ocr_Feature
from data import ViT5VQADataCollator, ViT5VQADataset

configure_env(output_path=OUTPUT_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# split (đối số) -> tên split trong file JSON gốc
SPLIT_TO_JSON = {"dev": "validation", "val": "validation", "test": "test"}
# split (đối số) -> tên CSV merged (để lấy image_id -> image_path/ocr_path)
SPLIT_TO_CSV = {"dev": "merged_val", "val": "merged_val", "test": "merged_test"}


def _norm_answer(x: str) -> str:
    """Chuẩn hoá text đầu ra: NFC + gộp khoảng trắng. Không hạ chữ hoa để giữ
    nguyên những gì model sinh (model đã được train trên target dạng thường)."""
    if x is None:
        return ""
    x = unicodedata.normalize("NFC", str(x))
    return " ".join(x.split())


def _norm_em(x: str) -> str:
    """Chuẩn hoá để đo EM cục bộ (khớp _normalize_txt của training/metrics.py)."""
    return " ".join(unicodedata.normalize("NFC", str(x or "")).strip().lower().split())


def resolve_ckpt_dir(user_dir: str) -> str:
    """Tìm thư mục bundle đã finetune."""
    if user_dir:
        if not os.path.isdir(user_dir):
            raise FileNotFoundError(f"--ckpt_dir không tồn tại: {user_dir}")
        return user_dir
    candidates = [
        os.path.join(OUTPUT_PATH, "pinned_best_ckpt"),
        os.path.join(OUTPUT_PATH, "best_bundle"),
        os.path.join(OUTPUT_PATH, "finetune"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            return c
    raise FileNotFoundError(
        "Không tìm thấy bundle finetune. Truyền --ckpt_dir <đường_dẫn> "
        f"(đã thử: {candidates})."
    )


def load_dfs_via_hub(dataset_name="ViTextVQA", data_dir="./datasets"):
    """Chuẩn bị dataset (tải + giải nén ảnh/OCR/JSON nếu chưa có) và trả về
    dict {'train','validation','test'} DataFrame. Nhờ mapper đã lưu `id`, mỗi df
    có sẵn cột `id` (submission ID) + image_path/ocr_path đã phân giải, ĐÚNG thứ
    tự annotation — không cần merged CSV, không lo reorder/trùng khoá.
    prepare() là idempotent: nếu ảnh/OCR đã giải nén thì bỏ qua tải lại."""
    import yaml
    from data.dataset_hub import DatasetHubLoader
    raw_dir = os.path.join(data_dir, "raw")
    out_dir = os.path.join(data_dir, "processed")
    hub = DatasetHubLoader(raw_dir, out_dir)
    cfg_path = f"configs/data/{dataset_name}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)
    hub.register_dataset(
        dataset_name=ds["dataset_name"], task_type="VQA",
        image_zip_id=ds["image"].get("drive_id"), image_dir_override="",
        ocr_zip_id=ds["ocr"].get("drive_id"), ocr_dir_override="",
        splits={s: {"id": ds["dataset"][s].get("drive_id") or ds["dataset"][s].get("dir"),
                    "url": None} for s in ["train", "validation", "test"]},
    )
    hub.prepare(dataset_name)
    return hub.load_task(dataset_name)


def df_from_hub(dfs, split):
    """Lấy df cho split từ dict Hub, chuẩn hoá cột answer (test = rỗng để không
    tính nhầm EM trên placeholder)."""
    key = "validation" if split in ("dev", "val") else "test"
    df = dfs[key].copy()
    if "id" not in df.columns:
        raise KeyError("df thiếu cột 'id' — mapper chưa được cập nhật?")
    if split == "test":
        df["answer"] = ""
    else:
        df["answer"] = df["answer"].fillna("").astype(str)
    print(f"[{split}] Hub -> {len(df)} dòng | id duy nhất: {df['id'].nunique()} | "
          f"thiếu image_path: {df['image_path'].isna().sum()}")
    return df


def build_eval_df(split: str, json_path: str, merged_csv: str,
                  image_dir: str, ocr_dir: str) -> pd.DataFrame:
    """Dựng DataFrame theo ĐÚNG thứ tự annotation trong JSON, mỗi dòng mang `id`
    (submission ID) riêng. image_path/ocr_path lấy qua map image_id (đã xác nhận
    mỗi image_id chỉ ứng 1 path)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns = data.get("annotations", [])
    if isinstance(anns, dict):
        anns = list(anns.values())

    # map image_id -> (image_path, ocr_path)
    img2img, img2ocr = {}, {}
    if merged_csv and os.path.exists(merged_csv):
        cdf = pd.read_csv(merged_csv)
        for iid, sub in cdf.groupby("image_id"):
            img2img[iid] = sub["image_path"].iloc[0]
            op = sub["ocr_path"].iloc[0]
            img2ocr[iid] = None if pd.isna(op) else op
    else:
        # fallback: dựng từ JSON images + thư mục do người dùng cung cấp
        if not image_dir:
            raise FileNotFoundError(
                f"Không thấy CSV merged ({merged_csv}). Hãy chạy bước merge dataset "
                "hoặc truyền --image_dir (và --ocr_dir) để tự phân giải đường dẫn."
            )
        imgs = data.get("images", [])
        id2fn = {}
        if isinstance(imgs, list):
            for im in imgs:
                id2fn[im.get("id")] = im.get("filename")
        elif isinstance(imgs, dict):
            id2fn = dict(imgs)
        for iid, fn in id2fn.items():
            img2img[iid] = os.path.join(image_dir, fn) if fn else None
            op = None
            if ocr_dir and fn:
                stem = os.path.splitext(os.path.basename(fn))[0]
                for cand in (f"{stem}.npy", f"{stem}.npz"):
                    p = os.path.join(ocr_dir, cand)
                    if os.path.isfile(p):
                        op = p
                        break
            img2ocr[iid] = op

    rows, missing = [], 0
    for a in anns:
        iid = a.get("image_id")
        ip = img2img.get(iid)
        if ip is None:
            missing += 1
        ans_list = a.get("answers") or ([a["answer"]] if a.get("answer") is not None else [])
        # test: answers là placeholder ["your answer"] -> coi như rỗng để không tính nhầm EM
        gt = ""
        if split in ("dev", "val") and ans_list:
            gt = str(ans_list[0])
        rows.append({
            "id": a.get("id"),
            "image_id": iid,
            "question": a.get("question"),
            "answer": gt,
            "image_path": ip,
            "ocr_path": img2ocr.get(iid),
        })
    if missing:
        print(f"⚠️  {missing}/{len(rows)} annotation không phân giải được image_path.")
    df = pd.DataFrame(rows)
    print(f"[{split}] dựng {len(df)} dòng từ {json_path} | id duy nhất: {df['id'].nunique()}")
    return df


def _to_dev(x):
    return x.to(DEVICE) if isinstance(x, torch.Tensor) else x


@torch.inference_mode()
def run_split(model, tokenizer, collator, df, split, out_csv,
              batch_size, num_beams, max_new_tokens, report_em):
    dataset = ViT5VQADataset(df)
    ids = df["id"].tolist()
    gts = df["answer"].tolist()
    n = len(dataset)
    preds = []

    print("=" * 72)
    print(f"PREDICT [{split}] n={n} bs={batch_size} beams={num_beams} "
          f"max_new_tokens={max_new_tokens}")
    print("=" * 72)

    skip_keys = {"ocr_info", "pil_images", "debug_raw_goc", "debug_raw_rel",
                 "debug_action", "debug_raw_questions", "debug_ocr_source",
                 "debug_image_path"}

    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        samples = [dataset[j] for j in idxs]
        batch = collator(samples)
        bd = {k: _to_dev(v) for k, v in batch.items() if k not in skip_keys}

        gen_out = model.generate(
            input_ids=bd.get("input_ids"),
            attention_mask=bd.get("attention_mask"),
            pixel_values=bd.get("pixel_values"),
            pil_images=batch.get("pil_images"),
            ocr_info=batch.get("ocr_info"),
            ocr_mask_token=bd.get("ocr_mask_token"),
            ocr_mask_box=bd.get("ocr_mask_box"),
            twa_ocr_char=bd.get("twa_ocr_char"),
            twa_ocr_char_mask=bd.get("twa_ocr_char_mask"),
            twa_word_ids=bd.get("twa_word_ids"),
            ocr_to_word_map=bd.get("ocr_to_word_map"),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        preds.extend(_norm_answer(t) for t in decoded)

        done = min(start + batch_size, n)
        if (start // batch_size) % 20 == 0 or done == n:
            print(f"  {done}/{n} ...")

    assert len(preds) == n == len(ids), f"len mismatch: preds={len(preds)} ids={len(ids)}"

    sub = pd.DataFrame({"ID": ids, "Answer": preds})
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    sub.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Ghi {len(sub)} dòng -> {out_csv}")
    print(sub.head(5).to_string(index=False))

    if report_em and any(g for g in gts):
        em = sum(1.0 for p, g in zip(preds, gts) if _norm_em(p) == _norm_em(g)) / n
        print(f"📊 [{split}] EM cục bộ (so với answers trong JSON) = {em:.4f} ({em*100:.2f}%)")
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "val", "test", "both"], default="dev",
                    help="dev=validation (public LB), test=private LB, both=cả hai")
    ap.add_argument("--ckpt_dir", default="",
                    help="Thư mục bundle finetune (mặc định tự dò pinned_best_ckpt/best_bundle/finetune)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=0,
                    help="0 = dùng generation_max_new_tokens của config (mặc định)")
    ap.add_argument("--out_dir", default=OUTPUT_PATH)
    ap.add_argument("--data_dir", default="./datasets",
                    help="Thư mục dataset cho Hub (tự tải/giải nén ảnh+OCR+JSON)")
    ap.add_argument("--no_hub", action="store_true",
                    help="Không dùng Hub; đọc JSON + merged CSV local (offline)")
    ap.add_argument("--json_dir", default="datasets/processed/ViTextVQA",
                    help="[fallback --no_hub] thư mục chứa ViTextVQA_*.json")
    ap.add_argument("--image_dir", default="", help="[fallback] nếu thiếu CSV merged")
    ap.add_argument("--ocr_dir", default="", help="[fallback] nếu thiếu CSV merged")
    ap.add_argument("--no_em", action="store_true", help="Bỏ tính EM cục bộ trên dev")
    args = ap.parse_args()

    splits = ["dev", "test"] if args.split == "both" else [args.split]

    ckpt_dir = resolve_ckpt_dir(args.ckpt_dir)
    print(f">>> Bundle: {ckpt_dir}")

    vision_ocr = Vision_Encode_Ocr_Feature(DEFAULT_OCR_CONFIG)
    model = OpenViVQAModel.from_pretrained(
        ckpt_dir, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    model.pretrain = False

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)

    ip_dir = os.path.join(ckpt_dir, "image_processor")
    if os.path.isdir(ip_dir):
        ip = CLIPImageProcessor.from_pretrained(ip_dir)
        model.image_processor = ip
        if hasattr(model.visual_search, "vit_processor"):
            model.visual_search.vit_processor = ip
    # else: dùng model.image_processor đã dựng trong __init__ (mặc định CLIP)

    mnt = int(getattr(model.config, "generation_max_new_tokens", 56))
    beams = args.num_beams
    model.generation_config = GenerationConfig(
        max_new_tokens=mnt, num_beams=beams, do_sample=False,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    max_new = args.max_new_tokens if args.max_new_tokens > 0 else mnt

    # dataframe cho collator (build vocab) — dùng train cache nếu có
    train_csv = os.path.join(OUTPUT_PATH, "merged_train.csv")
    ref_df = pd.read_csv(train_csv) if os.path.exists(train_csv) else pd.DataFrame(
        columns=["question", "answer", "image_path", "ocr_path"])

    collator = ViT5VQADataCollator(
        tokenizer=tokenizer,
        image_processor=model.image_processor,
        ocr_encoder=vision_ocr,
        config=model.config,
        term_vocab_path="configs/data/term_vocab.txt",
        viet_vocab_path="configs/data/viet_vocab.txt",
        eng_vocab_path="",
        dataframe=ref_df,
        pretrain=False,
    )
    if hasattr(collator, "set_mode"):
        collator.set_mode(pretrain=False, mask_prob=0.0)

    # Nguồn dữ liệu chính: DatasetHubLoader (tự tải/giải nén, df có sẵn `id`).
    # Bỏ qua nếu người dùng ép offline bằng --no_hub (dùng JSON + merged CSV local).
    dfs_hub = None
    if not args.no_hub:
        try:
            dfs_hub = load_dfs_via_hub(data_dir=args.data_dir)
            if "id" not in dfs_hub["test"].columns:
                print("⚠️  df từ Hub thiếu cột 'id' — chuyển sang fallback JSON/CSV.")
                dfs_hub = None
        except Exception as e:
            print(f"⚠️  Chuẩn bị dataset qua Hub thất bại ({e}); fallback JSON/CSV local.")
            dfs_hub = None

    outs = []
    for sp in splits:
        if dfs_hub is not None:
            df = df_from_hub(dfs_hub, sp)
        else:
            json_path = os.path.join(args.json_dir, f"ViTextVQA_{SPLIT_TO_JSON[sp]}.json")
            merged_csv = os.path.join(OUTPUT_PATH, f"{SPLIT_TO_CSV[sp]}.csv")
            df = build_eval_df(sp, json_path, merged_csv, args.image_dir, args.ocr_dir)
        out_csv = os.path.join(args.out_dir, f"submission_{sp}.csv")
        outs.append(run_split(
            model, tokenizer, collator, df, sp, out_csv,
            args.batch_size, beams, max_new, report_em=not args.no_em,
        ))
        torch.cuda.empty_cache()
        gc.collect()

    print("\n=== HOÀN TẤT ===")
    for o in outs:
        print("  ", o)


if __name__ == "__main__":
    main()
