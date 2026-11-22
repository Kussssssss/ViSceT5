"""Profile of one dataset against the fixed limits of the model, to compare ViTextVQA / OpenViVQA / ViOCRVQA.

The model hyper-parameters are identical for every dataset (same code, same config); what changes is the data. This script
measures the data-side quantities that those fixed hyper-parameters interact with:

  - sizes: rows per split, images, questions per image
  - question length in ViT5 tokens vs text_max_input_length (longer questions are truncated)
  - answer length in ViT5 tokens vs text_max_target_length (longer answers are cut in training; generation stops at the same limit)
  - OCR: share of rows with an OCR file, fields present (texts / boxes / det_features / rec_features / width / height), feature
    dimension and the share of all-zero vectors (a missing field is silently replaced by zeros), words per image vs ocr_max_scene_text,
    words longer than char_max_num
  - how often the answer can be read from the OCR text (substring / all words present)

Usage on Colab, in the notebook of the dataset (data already prepared by the run):
  python scripts/dataset_profile.py --dataset OpenViVQA          # writes output/dataset_profile_OpenViVQA.json
  python scripts/dataset_profile.py --compare output/dataset_profile_ViTextVQA.json output/dataset_profile_OpenViVQA.json ...
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LIMITS = {"question_tokens": 32, "answer_tokens": 56, "ocr_words": 180, "word_chars": 50}
SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))


def _limits():
    try:
        from configs.model_config import OpenViVQAConfig
        c = OpenViVQAConfig()
        return {"question_tokens": c.text_max_input_length, "answer_tokens": c.text_max_target_length,
                "ocr_words": c.ocr_max_scene_text, "word_chars": c.char_max_num}
    except Exception:
        return dict(LIMITS)


def normalize(text):
    return " ".join(re.sub(r"[^\w]+", " ", str(text).lower()).split())


def load_ocr(path):
    """npz / pickled dict of arrays -> dict, or None."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if isinstance(data, np.lib.npyio.NpzFile):
        return {key: data[key] for key in data.files}
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        item = data.item()
        return item if isinstance(item, dict) else None
    return None


def quantiles(values):
    if not len(values):
        return {"p50": None, "p95": None, "max": None, "mean": None}
    v = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(v.max()), "mean": float(v.mean())}


def share(values, limit):
    return float(np.mean([x > limit for x in values])) if len(values) else None


def profile_split(frame, tokenizer, limits, sample, seed=0):
    out = {"rows": int(len(frame)), "images": int(frame["image_path"].nunique()) if "image_path" in frame else None}
    out["questions_per_image"] = out["rows"] / out["images"] if out["images"] else None
    rows = frame.sample(n=min(sample, len(frame)), random_state=seed) if len(frame) else frame
    out["sampled"] = int(len(rows))
    q_tok = [len(ids) for ids in tokenizer(rows["question"].astype(str).tolist()).input_ids]
    out["question_tokens"] = quantiles(q_tok)
    out["question_over_limit"] = share(q_tok, limits["question_tokens"])
    answers = rows["answer"].fillna("").astype(str)
    answers = answers[answers.str.strip().ne("")]
    if len(answers) and answers.str.lower().ne("your answer").any():
        a_tok = [len(ids) for ids in tokenizer(answers.tolist()).input_ids]
        out["answer_tokens"] = quantiles(a_tok)
        out["answer_over_limit"] = share(a_tok, limits["answer_tokens"])
        out["answer_words_mean"] = float(np.mean([len(a.split()) for a in answers]))
        out["answer_distinct"] = int(answers.str.lower().nunique())
    paths = rows["ocr_path"] if "ocr_path" in rows else pd.Series([], dtype=object)
    exists = [isinstance(p, str) and os.path.isfile(p) for p in paths]
    out["ocr_file_share"] = float(np.mean(exists)) if len(exists) else None
    fields, words, long_words, in_text, all_words, answered, dims, zero = {}, [], 0, 0, 0, 0, {}, []
    loaded = 0
    for (_, row), ok in zip(rows.iterrows(), exists):
        if not ok:
            continue
        ocr = load_ocr(row["ocr_path"])
        if ocr is None:
            continue
        loaded += 1
        for key in ("texts", "boxes", "det_features", "rec_features", "width", "height"):
            if key in ocr:
                fields[key] = fields.get(key, 0) + 1
        texts = [str(t) for t in np.asarray(ocr.get("texts", [])).tolist()]
        words.append(len(texts))
        long_words += sum(len(t) > limits["word_chars"] for t in texts)
        for key in ("det_features", "rec_features"):
            arr = ocr.get(key)
            if arr is not None and getattr(arr, "ndim", 0) == 2 and len(arr):
                dims[key] = int(arr.shape[1])
                zero.append(float(np.mean(~np.asarray(arr).astype(bool).any(axis=1))))
        a = normalize(row["answer"]) if isinstance(row.get("answer"), str) and row["answer"].strip() else ""
        if a and a != "your answer":
            answered += 1
            joined = " " + " ".join(normalize(t) for t in texts) + " "
            in_text += f" {a} " in joined
            all_words += set(a.split()) <= set(joined.split())
    out["ocr_loaded"] = loaded
    out["ocr_fields"] = {k: v / loaded for k, v in fields.items()} if loaded else {}
    out["ocr_feature_dims"] = dims
    out["ocr_zero_vector_share"] = float(np.mean(zero)) if zero else None
    out["ocr_words_per_image"] = quantiles(words)
    out["ocr_images_without_words"] = float(np.mean([w == 0 for w in words])) if words else None
    out["ocr_images_over_word_limit"] = share(words, limits["ocr_words"])
    out["ocr_word_over_char_limit"] = (long_words / max(sum(words), 1)) if words else None
    out["answer_in_ocr_text"] = in_text / answered if answered else None
    out["answer_words_in_ocr"] = all_words / answered if answered else None
    return out


def profile(dataset, data_dir, sample):
    from transformers import AutoTokenizer
    from configs.base_config import OUTPUT_PATH
    from training.common import load_dataframes

    limits = _limits()
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    frames = load_dataframes(dataset, data_dir)
    result = {"dataset": dataset, "limits": limits, "splits": {}}
    for split, _ in SPLITS:
        if split in frames and len(frames[split]):
            result["splits"][split] = profile_split(frames[split], tokenizer, limits, sample)
            print(f"[{dataset}] {split}: {result['splits'][split]['rows']} rows profiled")
    out = Path(OUTPUT_PATH) / f"dataset_profile_{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out)
    return result


def fmt(value, kind="f"):
    if value is None:
        return "-"
    if kind == "%":
        return f"{100 * value:.1f}%"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def compare(paths, split="train"):
    profs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    rows = [
        ("rows", lambda s: fmt(s["rows"])),
        ("images", lambda s: fmt(s["images"])),
        ("questions / image", lambda s: fmt(s["questions_per_image"])),
        ("question tokens p50 / p95 / max", lambda s: "/".join(fmt(s["question_tokens"][k]) for k in ("p50", "p95", "max"))),
        ("question truncated (> limit)", lambda s: fmt(s["question_over_limit"], "%")),
        ("answer tokens p50 / p95 / max", lambda s: "/".join(fmt(s.get("answer_tokens", {}).get(k)) for k in ("p50", "p95", "max"))),
        ("answer cut (> limit)", lambda s: fmt(s.get("answer_over_limit"), "%")),
        ("answer words (mean)", lambda s: fmt(s.get("answer_words_mean"))),
        ("rows with OCR file", lambda s: fmt(s["ocr_file_share"], "%")),
        ("OCR fields present", lambda s: ",".join(f"{k[:3]}:{100 * v:.0f}" for k, v in s["ocr_fields"].items())),
        ("det / rec dim", lambda s: "/".join(str(s["ocr_feature_dims"].get(k, "-")) for k in ("det_features", "rec_features"))),
        ("all-zero det/rec vectors", lambda s: fmt(s["ocr_zero_vector_share"], "%")),
        ("OCR words / image p50 / p95 / max", lambda s: "/".join(fmt(s["ocr_words_per_image"][k]) for k in ("p50", "p95", "max"))),
        ("images with 0 OCR words", lambda s: fmt(s["ocr_images_without_words"], "%")),
        ("images over OCR word limit", lambda s: fmt(s["ocr_images_over_word_limit"], "%")),
        ("OCR words over char limit", lambda s: fmt(s["ocr_word_over_char_limit"], "%")),
        ("answer found in OCR text", lambda s: fmt(s["answer_in_ocr_text"], "%")),
        ("all answer words in OCR", lambda s: fmt(s["answer_words_in_ocr"], "%")),
    ]
    names = [p["dataset"] for p in profs]
    print(f"split = {split} | limits {profs[0]['limits']}")
    print(f"{'':36s}" + "".join(f"{n:>34s}" for n in names))
    for label, fn in rows:
        cells = []
        for p in profs:
            s = p["splits"].get(split)
            try:
                cells.append(fn(s) if s else "-")
            except Exception:
                cells.append("-")
        print(f"{label:36s}" + "".join(f"{c:>34s}" for c in cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--data_dir", default="./datasets")
    parser.add_argument("--sample", type=int, default=2000, help="rows per split used for token / OCR statistics")
    parser.add_argument("--compare", nargs="+", help="profile JSON files to print side by side")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.split)
    else:
        assert args.dataset, "--dataset or --compare"
        profile(args.dataset, args.data_dir, args.sample)


if __name__ == "__main__":
    main()
