"""
training/benchmark_eval.py
Chấm điểm một file submission theo ĐÚNG giao thức đánh giá benchmark
(notebook benchmark_evaluation.ipynb — cùng cách đã dùng cho các dòng baseline/LMM
trong bài), để mọi dòng trong bảng so sánh được với nhau.

  • Chuẩn hoá  : prep(x) = preprocess_sentence(normalize_text(x))  (utils.py của tác giả
                 bộ dữ liệu: bỏ dấu câu, lower, NFC, tách khoảng trắng quanh ký hiệu).
  • EM         : prep(pred) == prep(label), single-reference (answers[0]).
  • F1         : token-level và char-level, cùng _f1_single (rỗng-rỗng = 1).
  • BLEU 1-4, BLEU-avg, CIDEr: pycocoevalcap Bleu(4) / Cider trên chuỗi đã prep.
  • BERTScore  : evaluate("bertscore"), model bert-base-multilingual-cased (P/R/F1).
  • Chỉ chấm các ID có ở CẢ ground-truth và submission (giao ID).

Khác training/eval_submission.py (giữ cho khớp val eval lúc train: không bỏ dấu câu,
không BERTScore) — số liệu đưa vào bài dùng script NÀY.

Cách dùng:
    python -m training.benchmark_eval --gt ViTextVQA_test_gt.json \
        --pred output/submission_test.csv --out output/eval_test.json
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================================================
# 0. BẢN VÁ BERTSCORE CHO TRANSFORMERS MỚI (giữ y như notebook)
# =========================================================
def _patch_tokenizer_special_tokens():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls_id = [self.cls_token_id] if self.cls_token_id is not None else []
        sep_id = [self.sep_token_id] if self.sep_token_id is not None else []
        if token_ids_1 is None:
            return cls_id + token_ids_0 + sep_id
        return cls_id + token_ids_0 + sep_id + token_ids_1 + sep_id

    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs_with_special_tokens
    PreTrainedTokenizerFast.build_inputs_with_special_tokens = _build_inputs_with_special_tokens


# =========================================================
# 1. PREPROCESSING (đồng bộ utils.py của tác giả bộ dữ liệu)
# =========================================================
def normalize_text(text: str) -> str:
    if text is None or pd.isna(text):
        return ""
    text = str(text).translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text


def preprocess_sentence(sentence: str) -> str:
    if not sentence:
        return ""
    sentence = sentence.lower()
    sentence = unicodedata.normalize("NFC", sentence)
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def prep(text: str) -> str:
    return preprocess_sentence(normalize_text(text))


# =========================================================
# 2. EM & F1
# =========================================================
def compute_em(preds: List[str], labels: List[str]) -> float:
    return float(np.mean([1.0 if prep(p) == prep(l) else 0.0 for p, l in zip(preds, labels)]))


def _f1_single(pred_items: List[str], label_items: List[str]) -> float:
    if len(pred_items) == 0 or len(label_items) == 0:
        return 1.0 if pred_items == label_items else 0.0
    common = Counter(pred_items) & Counter(label_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_items)
    recall = num_same / len(label_items)
    return 2 * (precision * recall) / (precision + recall)


def compute_f1(preds: List[str], labels: List[str]):
    f1_tokens, f1_chars = [], []
    for p, l in zip(preds, labels):
        p_clean, l_clean = prep(p), prep(l)
        f1_tokens.append(_f1_single(p_clean.split(), l_clean.split()))
        f1_chars.append(_f1_single(list(p_clean), list(l_clean)))
    return float(np.mean(f1_tokens)), float(np.mean(f1_chars))


# =========================================================
# 3. BLEU 1-4, BLEU-avg & CIDEr
# =========================================================
def compute_bleu_cider(preds: List[str], labels: List[str]) -> Dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [prep(l)] for i, l in enumerate(labels)}
    gens = {i: [prep(p)] for i, p in enumerate(preds)}
    bleu_scores, _ = Bleu(4).compute_score(gts, gens)
    cider_score, _ = Cider().compute_score(gts, gens)
    return {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-2": float(bleu_scores[1]),
        "BLEU-3": float(bleu_scores[2]),
        "BLEU-4": float(bleu_scores[3]),
        "BLEU-avg": float(np.mean(bleu_scores)),
        "CIDEr": float(cider_score),
    }


# =========================================================
# 4. BERTScore
# =========================================================
def compute_bertscore(preds: List[str], labels: List[str],
                      model_type: str = "bert-base-multilingual-cased") -> Dict[str, float]:
    from evaluate import load as load_metric

    _patch_tokenizer_special_tokens()
    print(f"   -> Loading HuggingFace evaluate('bertscore') with {model_type}...")
    bertscore_metric = load_metric("bertscore")
    results = bertscore_metric.compute(
        predictions=[prep(x) for x in preds],
        references=[prep(x) for x in labels],
        model_type=model_type,
    )
    return {
        "BERTScore_F1": float(np.mean(results["f1"])),
        "BERTScore_P": float(np.mean(results["precision"])),
        "BERTScore_R": float(np.mean(results["recall"])),
    }


# =========================================================
# 5. PIPELINE (giao ID giữa GT và submission)
# =========================================================
def evaluate_submission(gt_path: str, pred_path: str, bertscore: bool = True) -> Dict[str, float]:
    print("⏳ Loading data...")
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    id_to_label = {int(ann["id"]): ann["answers"][0] for ann in gt_data["annotations"]}

    if pred_path.endswith(".json"):
        with open(pred_path, encoding="utf-8") as f:
            pred_data = json.load(f)
        id_to_pred = {int(k): str(v) for k, v in pred_data.items()}
    else:
        df = pd.read_csv(pred_path)
        id_to_pred = {}
        for _, row in df.iterrows():
            val = row.iloc[1]
            id_to_pred[int(row.iloc[0])] = str(val) if pd.notna(val) else ""

    common_ids = sorted(set(id_to_label.keys()) & set(id_to_pred.keys()))
    print("-" * 50)
    print(f"📌 Tổng số ID trong Ground Truth  : {len(id_to_label):,}")
    print(f"📌 Tổng số ID trong Submission    : {len(id_to_pred):,}")
    print(f"🎯 Số lượng mẫu CHUNG được đánh giá : {len(common_ids):,}")
    print("-" * 50)
    if len(common_ids) == 0:
        raise RuntimeError("❌ Không tìm thấy ID nào trùng khớp giữa GT và Submission!")
    if len(common_ids) < len(id_to_label):
        print(f"⚠️ {len(id_to_label) - len(common_ids):,} ID của GT không có trong submission → không được chấm.")

    preds = [id_to_pred[i] for i in common_ids]
    labels = [id_to_label[i] for i in common_ids]

    print("⏳ Computing EM & F1...")
    em_score = compute_em(preds, labels)
    f1_token, f1_char = compute_f1(preds, labels)
    print("⏳ Computing BLEU (1-4, avg) & CIDEr...")
    bleu_cider = compute_bleu_cider(preds, labels)
    bert = {}
    if bertscore:
        print("⏳ Computing BERTScore...")
        bert = compute_bertscore(preds, labels, model_type="bert-base-multilingual-cased")

    results = {
        "Exact Match (EM)": em_score,
        "F1 (Token-level)": f1_token,
        "F1 (Char-level)": f1_char,
        **bleu_cider,
        **bert,
    }
    print("\n" + "=" * 40)
    print("         FINAL EVALUATION RESULTS         ")
    print("=" * 40)
    for k, v in results.items():
        print(f"{k:<20}: {v:.4f}")
    print("=" * 40)
    results["n_evaluated"] = len(common_ids)
    results["n_gt"] = len(id_to_label)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="JSON ground-truth ({'annotations': [{'id', 'answers'}]})")
    ap.add_argument("--pred", required=True, help="Submission CSV (ID,Answer) hoặc JSON {id: answer}")
    ap.add_argument("--out", default="", help="(tuỳ chọn) lưu kết quả ra JSON")
    ap.add_argument("--no_bertscore", action="store_true", help="Bỏ BERTScore (không tải mBERT)")
    args = ap.parse_args()
    res = evaluate_submission(args.gt, args.pred, bertscore=not args.no_bertscore)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
