"""
training/eval_submission.py
Chấm điểm OFFLINE một file submission bằng cách MAP THEO id với ground-truth —
dùng khi hệ thống nộp (Codalab) đã đóng nhưng đã xin được tập test có đáp án.

Tính đúng BỘ METRIC như lúc đánh giá val trong finetune
(training.metrics.build_compute_metrics_finetune):
  • EM, F1(token)  : single-reference, chuẩn hoá bằng _normalize_txt (NFC+strip+
                     lower+gộp khoảng trắng) — TÁI DÙNG trực tiếp training.metrics
                     nếu import được (khớp tuyệt đối), nếu không thì bản sao inline.
  • BLEU-1..4, CIDEr: pycocoevalcap.Bleu(4)/Cider — CÙNG thư viện val eval dùng.

KHÁC BIỆT DUY NHẤT so với val eval (đã đo, không đáng kể — xem docs/):
  - val eval tokenize BLEU/CIDEr bằng PTBTokenizer (cần Java); script này chuẩn hoá
    bằng _normalize_txt rồi tách khoảng trắng. Trên dữ liệu tiếng Việt (đã có khoảng
    trắng quanh dấu câu) sai lệch đo được ≤ 0.008 CIDEr / ≤ 0.002 BLEU.
  - val eval so với nhãn đã round-trip qua tokenizer ViT5; ở đây so với GT thô (đúng
    hơn cho điểm test chính thức). Chênh EM ~0.2pp / F1 ~0.003.

Cách dùng:
    python -m training.eval_submission \
        --gt   datasets/.../<dataset>_test_gt.json \
        --pred output/submission_<dataset>_test.csv \
        --out  output/eval_<dataset>_test.json

Ghi chú OpenViVQA/VLSP: trên leaderboard Codalab, cột "F1" THỰC CHẤT là CIDEr.
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import unicodedata
import collections
import pandas as pd


# ── EM/F1: tái dùng training.metrics để KHỚP TUYỆT ĐỐI val eval; fallback inline ──
try:
    from training.metrics import _normalize_txt as _norm, compute_f1_em as _cf1
    _EMF1_SRC = "training.metrics (khớp tuyệt đối val eval)"
except Exception:
    def _norm(x: str) -> str:
        if x is None:
            return ""
        x = unicodedata.normalize("NFC", str(x))
        return " ".join(x.strip().lower().split())

    def _cf1(preds, labels):
        f1s, ems = [], []
        for p, l in zip(preds, labels):
            p, l = _norm(p), _norm(l)
            ems.append(1.0 if p == l else 0.0)
            pt, lt = p.split(), l.split()
            if not pt or not lt:
                f1s.append(0.0); continue
            common = collections.Counter(pt) & collections.Counter(lt)
            ns = sum(common.values())
            if ns == 0:
                f1s.append(0.0); continue
            pr, rc = ns / len(pt), ns / len(lt)
            f1s.append(2 * pr * rc / (pr + rc))
        import numpy as _np
        return float(_np.mean(f1s)), float(_np.mean(ems))
    _EMF1_SRC = "bản sao inline (giống hệt training.metrics)"


def load_gt(path):
    """id(str) -> list[answer]. Hỗ trợ annotations là list (id inline) HOẶC dict
    (id = KHÓA, kiểu OpenViVQA/VLSP)."""
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    anns = j.get("annotations", j)
    out = {}

    def _answers(v):
        a = v.get("answers")
        if a is None and v.get("answer") is not None:
            a = [v.get("answer")]
        return [str(x) for x in (a or [])]

    if isinstance(anns, dict):
        for k, v in anns.items():
            if not isinstance(v, dict):
                continue
            aid = v.get("id", k)   # OpenViVQA: value không có id → dùng khóa
            out[str(aid)] = _answers(v)
    elif isinstance(anns, list):
        for v in anns:
            if isinstance(v, dict) and v.get("id") is not None:
                out[str(v["id"])] = _answers(v)
    return out


def load_pred(path):
    """id(str) -> answer(str). Đọc CSV/txt header ID,Answer (định dạng predict.py)."""
    df = pd.read_csv(path)
    col = {c.lower(): c for c in df.columns}
    idc, ac = col.get("id", df.columns[0]), col.get("answer", df.columns[1])
    return {str(r[idc]): ("" if pd.isna(r[ac]) else str(r[ac])) for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="JSON ground-truth (có đáp án thật)")
    ap.add_argument("--pred", required=True, help="CSV/txt submission (ID,Answer)")
    ap.add_argument("--out", default="", help="(tùy chọn) lưu kết quả ra JSON")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    pred = load_pred(args.pred)
    ids = [i for i in gt if i in pred]
    n = len(ids)
    n_miss = len([i for i in gt if i not in pred])
    print(f"GT: {len(gt)} | Pred: {len(pred)} | KHỚP id: {n} | GT thiếu pred: {n_miss}")
    if n == 0:
        raise SystemExit("Không có id nào khớp giữa GT và pred.")
    nref = collections.Counter(len(gt[i]) for i in ids)
    print(f"số answers/câu: {dict(nref)} | EM/F1 nguồn: {_EMF1_SRC}")

    # EM/F1 — single-reference (answers[0]), đúng như val eval
    preds = [pred[i] for i in ids]
    refs1 = [gt[i][0] if gt[i] else "" for i in ids]
    f1, em = _cf1(preds, refs1)

    res = {"n": n, "em": round(em, 6), "f1": round(f1, 6)}

    # BLEU/CIDEr — đa tham chiếu (bằng single khi mỗi câu 1 ref), pycocoevalcap
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [_norm(a) for a in gt[i]] for i in ids}
        gens = {i: [_norm(pred[i])] for i in ids}
        (b1, b2, b3, b4), _ = Bleu(4).compute_score(gts, gens)
        cider, _ = Cider().compute_score(gts, gens)
        res.update({"bleu1": round(b1, 6), "bleu2": round(b2, 6), "bleu3": round(b3, 6),
                    "bleu4": round(b4, 6), "cider": round(cider, 6)})
    except Exception as e:
        print(f"⚠️ BLEU/CIDEr bỏ qua ({type(e).__name__}: {e}). "
              f"Cài: pip install pycocoevalcap. (EM/F1 vẫn có ở trên.)")

    print("\n== KẾT QUẢ (map theo id, {} câu) ==".format(n))
    for k in ["cider", "bleu1", "bleu2", "bleu3", "bleu4", "f1", "em"]:
        if k in res:
            tag = "  <-- = 'F1' trên Codalab" if k == "cider" else ""
            print(f"  {k.upper():6s} = {res[k]:.4f}{tag}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
