"""Test ground truth for benchmark_eval, for any dataset: eval_gt/<DATASET>_test_gt.json.

The test annotation JSON (configs/data/<DATASET>.yaml, dataset.test) already holds the answers
(OpenViVQA: the 3536-question subset with GT). It is taken from the prepared data
(<data_dir>/processed/<DATASET>/<DATASET>_test.json) or downloaded, and written as
{"annotations": [{"id", "answers"}]}, the format of eval_gt/ViTextVQA_test_gt.json, so the scoring
is the same as for ViTextVQA. An existing file is kept.

Usage:  python scripts/make_test_gt.py --dataset OpenViVQA
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path.cwd()  # repo root (the notebook runs it from /content/ViSceT5, at any code commit)


def source(dataset, data_dir):
    path = Path(data_dir) / "processed" / dataset / f"{dataset}_test.json"
    if path.is_file():
        return path
    config = yaml.safe_load(open(ROOT / "configs" / "data" / f"{dataset}.yaml", encoding="utf-8"))
    drive_id = str(config["dataset"]["test"].get("drive_id") or "").split(",")[0].strip()
    if not drive_id:
        return None
    import gdown
    path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=drive_id, output=str(path), quiet=True)
    return path if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", default="./datasets")
    args = parser.parse_args()
    out = ROOT / "eval_gt" / f"{args.dataset}_test_gt.json"
    if out.is_file():
        print(f"GT test: {out} (đã có)")
        return
    path = source(args.dataset, args.data_dir)
    if path is None:
        print(f"GT test {args.dataset}: không có file test trong yaml → chưa chấm test.")
        return
    data = json.load(open(path, encoding="utf-8"))
    anns = data.get("annotations", []) if isinstance(data, dict) else data
    if isinstance(anns, dict):  # OpenViVQA: key = question id
        anns = [{**v, "id": v.get("id", k)} for k, v in anns.items() if isinstance(v, dict)]
    gt = []
    for ann in anns:
        answers = ann.get("answers") if "answers" in ann else [ann.get("answer")]
        answers = [a for a in (answers if isinstance(answers, list) else [answers]) if a is not None]
        if answers and str(answers[0]).strip():
            gt.append({"id": int(ann["id"]), "answers": [str(a) for a in answers]})
    if not gt:
        print(f"GT test {args.dataset}: {path} không có đáp án → chưa chấm test.")
        return
    out.parent.mkdir(exist_ok=True)
    json.dump({"annotations": gt}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"GT test: {out} | {len(gt)} câu có đáp án / {len(anns)} câu")


if __name__ == "__main__":
    sys.exit(main())
