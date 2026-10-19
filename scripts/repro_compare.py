"""
scripts/repro_compare.py
So nhiều file nộp test với một file THAM CHIẾU (vd submission gốc của run Vast): EM/F1 theo
đúng chuẩn hoá của benchmark (training/benchmark_eval.prep), tỷ lệ dự đoán trùng tham chiếu,
số câu chỉ một bên đúng, và khoảng tin cậy bootstrap cho chênh lệch EM.

  python scripts/repro_compare.py --gt eval_gt/ViTextVQA_test_gt.json --ref vast.csv \
      --pred colab_ckpt=a.csv retrain=b.csv --out compare.json
"""
import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.benchmark_eval import _f1_single, prep  # noqa: E402


def load(path, ids):
    df = pd.read_csv(path)
    m = {int(r.iloc[0]): (str(r.iloc[1]) if pd.notna(r.iloc[1]) else "") for _, r in df.iterrows()}
    return {i: prep(m.get(i, "")) for i in ids}, sum(i not in m for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ref", required=True, help="file tham chiếu (vd submission gốc trên Vast)")
    ap.add_argument("--pred", nargs="+", required=True, help="TÊN=đường_dẫn, bỏ qua file không tồn tại")
    ap.add_argument("--out", default="")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()

    gt = {int(x["id"]): prep(x["answers"][0])
          for x in json.load(open(a.gt, encoding="utf-8"))["annotations"]}
    ids = sorted(gt)
    ref, _ = load(a.ref, ids)
    ref_hit = {i: int(ref[i] == gt[i]) for i in ids}

    rows = []
    for spec in a.pred:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            print(f"ℹ️ bỏ qua {name}: không có {path}")
            continue
        pred, missing = load(path, ids)
        hit = {i: int(pred[i] == gt[i]) for i in ids}
        rng = random.Random(0)
        diffs = sorted(sum(hit[i] - ref_hit[i] for i in s) / len(ids) * 100
                       for s in ([rng.choice(ids) for _ in ids] for _ in range(a.n_boot)))
        rows.append({
            "run": name,
            "EM": round(100 * sum(hit.values()) / len(ids), 2),
            "F1": round(100 * sum(_f1_single(pred[i].split(), gt[i].split()) for i in ids) / len(ids), 2),
            "same_as_ref_%": round(100 * sum(pred[i] == ref[i] for i in ids) / len(ids), 2),
            "only_ref_right": sum(ref_hit[i] and not hit[i] for i in ids),
            "only_this_right": sum(hit[i] and not ref_hit[i] for i in ids),
            "dEM_vs_ref": round(100 * (sum(hit.values()) - sum(ref_hit.values())) / len(ids), 2),
            "dEM_95CI": [round(diffs[int(0.025 * len(diffs))], 2), round(diffs[int(0.975 * len(diffs)) - 1], 2)],
            "missing": missing,
        })

    print(f"Tham chiếu: {a.ref}  (n={len(ids)})")
    print(pd.DataFrame(rows).to_string(index=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"ref": a.ref, "n": len(ids), "rows": rows}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
