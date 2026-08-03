#!/usr/bin/env python
"""
scripts/verify_data_resolution.py

KIỂM CHỨNG divergence ⓑ: repo (có fallback tìm file toàn cây `_resolve_in_tree`)
vs notebook (chỉ dùng 1 leaf `_find_leaf_dir`, KHÔNG fallback) resolve khác nhau
bao nhiêu dòng ảnh/OCR.

Cách làm: dùng CHÍNH mapper của repo (data.dataset_hub._default_vqa_mapper), chạy
2 lần trên đúng bộ dữ liệu đã prepare:
  1) REPO  = fallback BẬT  (mặc định).
  2) NOTEBOOK = monkeypatch `_resolve_in_tree` -> None để mô phỏng notebook (không
     fallback). Mọi thứ còn lại (leaf, candidate trong leaf) y hệt nhau, nên phần
     lệch số đo được CHÍNH XÁC là do fallback đa-thư-mục — tức divergence ⓑ.

Chạy (trong môi trường ĐÃ có/đã tải dữ liệu, vd Colab sau prepare):
    python scripts/verify_data_resolution.py --config configs/data/OpenViVQA.yaml --data_dir ./datasets

Nếu ảnh/OCR đã giải nén sẵn ở thư mục phẳng, có thể trỏ thẳng:
    python scripts/verify_data_resolution.py --config configs/data/OpenViVQA.yaml \
        --image_dir /path/to/images --ocr_dir /path/to/ocr
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows console mặc định cp1252 → emoji/tiếng Việt vỡ. Ép UTF-8 (Colab đã sẵn utf-8).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import yaml
import pandas as pd

import data.dataset_hub as H
from data.dataset_hub import DatasetHubLoader, _is_img, _is_ocr, _find_leaf_dir


def _isfile(p):
    return bool(p) and os.path.isfile(p)


def _split_counts(df: pd.DataFrame):
    """Đếm ocr_None và image_missing cho 1 dataframe split."""
    n = len(df)
    if n == 0:
        return {"n": 0, "ocr_none": 0, "img_missing": 0}
    ocr_none = int(df["ocr_path"].isna().sum() + (df["ocr_path"] == None).sum())  # noqa: E711
    # image_missing: path None hoặc file không tồn tại trên đĩa
    img_missing = int(sum(0 if _isfile(p) else 1 for p in df["image_path"].tolist()))
    # ocr_none tính lại chắc chắn theo None thực sự
    ocr_none = int(sum(1 for p in df["ocr_path"].tolist() if not p))
    return {"n": n, "ocr_none": ocr_none, "img_missing": img_missing}


def _build_task(hub: DatasetHubLoader, name: str):
    """Trả về dict split->df (train/validation/test) từ build_df hiện tại."""
    return hub.load_task(name)


def _describe_leaf(leaf_dir, pred, label):
    """In leaf đang dùng + các thư mục anh-em (để lộ layout tách-the-split)."""
    if not leaf_dir or not os.path.isdir(leaf_dir):
        print(f"   {label}: (không có / chưa prepare)")
        return
    n_here = len([f for f in os.listdir(leaf_dir)
                  if os.path.isfile(os.path.join(leaf_dir, f)) and pred(f)])
    parent = os.path.dirname(leaf_dir.rstrip("/\\")) or leaf_dir
    print(f"   {label}: leaf = {leaf_dir}  ({n_here:,} file)")
    # liệt kê các thư mục con khác dưới cùng parent (nơi fallback sẽ tìm thêm)
    try:
        sibs = []
        for cur, _dirs, files in os.walk(parent):
            if os.path.abspath(cur) == os.path.abspath(leaf_dir):
                continue
            c = len([f for f in files if pred(f)])
            if c > 0:
                sibs.append((cur, c))
        if sibs:
            print(f"      → parent={parent} còn {len(sibs)} thư mục khác CÓ file "
                  f"(fallback repo sẽ với tới, notebook thì KHÔNG):")
            for d, c in sorted(sibs, key=lambda x: -x[1])[:8]:
                print(f"         {c:>8,}  {d}")
        else:
            print(f"      → parent={parent}: không có thư mục anh-em nào chứa file "
                  f"(layout PHẲNG → 2 cách resolve sẽ GIỐNG nhau).")
    except Exception as e:
        print(f"      (không quét được parent: {e})")


def main():
    ap = argparse.ArgumentParser(description="Verify OCR/image path-resolution divergence (ⓑ)")
    ap.add_argument("--config", default="configs/data/OpenViVQA.yaml")
    ap.add_argument("--data_dir", default="./datasets")
    ap.add_argument("--image_dir", default="", help="(tuỳ chọn) trỏ thẳng thư mục ảnh đã giải nén")
    ap.add_argument("--ocr_dir", default="", help="(tuỳ chọn) trỏ thẳng thư mục OCR đã giải nén")
    ap.add_argument("--examples", type=int, default=5, help="Số ví dụ dòng 'repo có OCR / notebook mất' để in")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    name = cfg["dataset_name"]

    # --- resolve nguồn ảnh/OCR giống prepare_dataset ---
    image_dir_override = args.image_dir or (cfg["image"].get("dir", "") if os.path.isdir(cfg["image"].get("dir", "")) else "")
    image_zip_id = None if image_dir_override else cfg["image"].get("drive_id", "")
    ocr_cfg = cfg.get("ocr", {})
    ocr_dir_override = args.ocr_dir or (ocr_cfg.get("dir", "") if os.path.isdir(ocr_cfg.get("dir", "")) else "")
    ocr_zip_id = None if ocr_dir_override else ocr_cfg.get("drive_id", "")

    def resolve_split(section):
        d, i = section.get("dir", ""), section.get("drive_id", "")
        return d if (d and os.path.exists(d)) else i

    raw_dir = os.path.join(args.data_dir, "raw")
    out_dir = os.path.join(args.data_dir, "processed")
    hub = DatasetHubLoader(raw_dir, out_dir)
    hub.register_dataset(
        dataset_name=name, task_type="VQA",
        image_zip_id=image_zip_id, image_dir_override=image_dir_override,
        ocr_zip_id=ocr_zip_id, ocr_dir_override=ocr_dir_override,
        splits={
            "train":      {"id": resolve_split(cfg["dataset"]["train"]), "url": None},
            "validation": {"id": resolve_split(cfg["dataset"]["validation"]), "url": None},
            "test":       {"id": resolve_split(cfg["dataset"]["test"]), "url": None},
        },
    )

    print(f"⬇️  prepare({name}) — tải/giải nén nếu chưa có (dùng lại nếu đã có)...")
    info = hub.prepare(name)
    print("\n📂 LEAF đang dùng (cả repo & notebook đều chọn leaf này qua _find_leaf_dir):")
    _describe_leaf(info.get("image_dir"), _is_img, "IMAGE")
    _describe_leaf(info.get("ocr_dir"), _is_ocr, "OCR  ")

    # --- 1) REPO: fallback BẬT ---
    H._FILE_INDEX_CACHE.clear()
    repo_task = _build_task(hub, name)

    # --- 2) NOTEBOOK: fallback TẮT (monkeypatch _resolve_in_tree -> None) ---
    _orig = H._resolve_in_tree
    H._resolve_in_tree = lambda *a, **k: None
    try:
        nb_task = _build_task(hub, name)
    finally:
        H._resolve_in_tree = _orig

    # --- báo cáo ---
    print("\n" + "=" * 78)
    print(f"KẾT QUẢ ⓑ — dataset={name}")
    print("=" * 78)
    header = f"{'split':<12}{'rows':>8} | {'OCR=None':>18} | {'IMG thiếu':>18}"
    print(header)
    print(f"{'':<12}{'':>8} | {'repo':>8}{'nb':>10} | {'repo':>8}{'nb':>10}")
    print("-" * 78)
    total_gain_ocr = 0
    total_gain_img = 0
    for split in ["train", "validation", "test"]:
        r = _split_counts(repo_task[split])
        b = _split_counts(nb_task[split])
        gain_ocr = b["ocr_none"] - r["ocr_none"]   # notebook thiếu, repo có
        gain_img = b["img_missing"] - r["img_missing"]
        total_gain_ocr += max(0, gain_ocr)
        total_gain_img += max(0, gain_img)
        print(f"{split:<12}{r['n']:>8} | {r['ocr_none']:>8}{b['ocr_none']:>10} | "
              f"{r['img_missing']:>8}{b['img_missing']:>10}")
    print("-" * 78)
    print(f"→ Số dòng REPO resolve được OCR mà NOTEBOOK bị None (nhờ fallback): {total_gain_ocr:,}")
    print(f"→ Số dòng REPO resolve được ẢNH mà NOTEBOOK bị thiếu file:        {total_gain_img:,}")

    if total_gain_ocr == 0 and total_gain_img == 0:
        print("\n✅ KHÔNG lệch: layout phẳng — repo và notebook resolve GIỐNG nhau. "
              "ⓑ không ảnh hưởng ở bộ dữ liệu/máy này.")
    else:
        print("\n⚠️  CÓ lệch: notebook âm thầm bỏ OCR/ảnh cho các dòng ở split khác leaf; "
              "repo (fallback) resolve đủ. Đây chính là divergence ⓑ → repo là bản ĐÚNG hơn.")

        # in vài ví dụ dòng repo-có-OCR / notebook-None để soi
        if args.examples > 0:
            print(f"\nVí dụ (tối đa {args.examples}) dòng repo CÓ ocr_path còn notebook None:")
            shown = 0
            for split in ["train", "validation", "test"]:
                rdf, bdf = repo_task[split], nb_task[split]
                if len(rdf) == 0:
                    continue
                merged = rdf[["image_filename", "ocr_path"]].copy()
                merged.columns = ["image_filename", "ocr_repo"]
                merged["ocr_nb"] = bdf["ocr_path"].values if len(bdf) == len(rdf) else None
                diff = merged[(merged["ocr_repo"].astype(bool)) & (~merged["ocr_nb"].astype(bool))]
                for _, row in diff.head(args.examples - shown).iterrows():
                    print(f"   [{split}] {row['image_filename']}  ->  {row['ocr_repo']}")
                    shown += 1
                    if shown >= args.examples:
                        break
                if shown >= args.examples:
                    break

    print("\n(Ghi chú: 2 lần chạy chỉ khác nhau ở fallback `_resolve_in_tree`; mọi bước "
          "khác — leaf, candidate trong leaf — y hệt, nên chênh lệch trên = tác động ⓑ.)")


if __name__ == "__main__":
    main()
