"""Where do the wrong answers come from? Text only, no model and no GPU.

Every test question falls in one class, from the gold answer, the OCR words the model sees (same
filtering as the collator) and the prediction. Each class points to a different change:

  OCR has the whole answer
    outside   prediction has a word that is not in the OCR  -> copy from the OCR
    boundary  words all from the OCR, some shared with gold -> pick the right span
    region    words all from the OCR, none shared with gold -> relate the question to the OCR
  only the corrected copy completes the answer              -> related copy has value
  OCR has part of the answer / none of it                   -> OCR recall, the model cannot read it

python scripts/error_anatomy.py --gt eval_gt/ViTextVQA_test_gt.json --data_dir ./datasets \
    --pred base=a/submission_test.csv combo=b/submission_test.csv
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.model_config import OpenViVQAConfig
from training.benchmark_eval import _f1_single, prep
from training.common import PROJECT_ROOT, build_collator, build_corrector, load_dataframes

CLASSES = ("full_correct", "full_outside", "full_boundary", "full_region",
           "corrected_correct", "corrected_wrong", "partial_correct", "partial_wrong",
           "none_correct", "none_wrong")


def contains_span(words, span):
    return any(words[i:i + len(span)] == span for i in range(len(words) - len(span) + 1))


def classify(gold, prediction, ocr, union):
    right = prediction == gold
    if all(word in ocr for word in gold):
        if right:
            return "full_correct"
        if any(word not in ocr for word in prediction):
            return "full_outside"
        return "full_boundary" if set(prediction) & set(gold) else "full_region"
    if all(word in union for word in gold):
        return "corrected_correct" if right else "corrected_wrong"
    if any(word in union for word in gold):
        return "partial_correct" if right else "partial_wrong"
    return "none_correct" if right else "none_wrong"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", nargs="+", required=True, help="name=submission.csv; paired EM difference of every pair")
    parser.add_argument("--dataset", default="ViTextVQA")
    parser.add_argument("--data_dir", default="./datasets")
    args = parser.parse_args()

    values = yaml.safe_load(open(PROJECT_ROOT / "configs" / "finetune.yaml", encoding="utf-8"))
    config = OpenViVQAConfig(**{k: v for k, v in values.items() if k.startswith("ocr_") or k == "char_max_num"})
    tokenizer = AutoTokenizer.from_pretrained(config.vit5_name)
    frames = load_dataframes(args.dataset, args.data_dir)
    corrector = build_corrector(config, frames["train"])
    collator = build_collator(SimpleNamespace(config=config, image_processor=None), tokenizer, corrector)

    gold = {int(a["id"]): prep(a["answers"][0]).split()
            for a in json.load(open(args.gt, encoding="utf-8"))["annotations"]}
    test = frames["test"]
    test = test[test["id"].astype(int).isin(gold)]
    cache, questions = {}, []
    for _, row in test.iterrows():
        key = str(row["ocr_path"])
        if key not in cache:
            raw = collator.ocr_encoder([row["image_path"]], ocr_paths=[row["ocr_path"]])[0]
            texts = collator._prepare_ocr(raw)["texts"]
            original = prep(" ".join(texts)).split()
            cache[key] = (original, set(original) | set(prep(" ".join(corrector.correct(texts))).split()))
        questions.append((int(row["id"]), *cache[key]))
    count = len(questions)
    print(f"{count} questions | {len(cache)} images")
    answerable = sum(all(w in set(o) for w in gold[i]) for i, o, _ in questions)
    span = sum(contains_span(o, gold[i]) for i, o, _ in questions)
    union = sum(all(w in u for w in gold[i]) for i, _, u in questions)
    print(f"whole answer in the OCR {100 * answerable / count:.2f}% | as one contiguous span {100 * span / count:.2f}%"
          f" | in OCR + corrected copy {100 * union / count:.2f}%")

    table, scores = {}, {}
    for item in args.pred:
        name, path = item.split("=", 1)
        frame = pd.read_csv(path)
        answers = {int(r.iloc[0]): "" if pd.isna(r.iloc[1]) else str(r.iloc[1]) for _, r in frame.iterrows()}
        counts = dict.fromkeys(CLASSES, 0)
        em, f1, lengths, over, under = [], [], [], 0, 0
        for question, original, union_words in questions:
            prediction = prep(answers.get(question, "")).split()
            counts[classify(gold[question], prediction, set(original), union_words)] += 1
            em.append(float(prediction == gold[question]))
            f1.append(_f1_single(prediction, gold[question]))
            lengths.append(len(prediction))
            if prediction != gold[question]:
                over += contains_span(prediction, gold[question])
                under += bool(prediction) and contains_span(gold[question], prediction)
        table[name] = {k: round(100 * v / count, 2) for k, v in counts.items()}
        scores[name] = np.array(em)
        print(f"{name}: EM {100 * np.mean(em):.2f} F1 {100 * np.mean(f1):.2f}"
              f" | words per answer {np.mean(lengths):.2f} (gold {np.mean([len(gold[q[0]]) for q in questions]):.2f})"
              f" | right start and words, too long {100 * over / count:.2f}% too short {100 * under / count:.2f}%")
    print("\n% of all questions")
    print(pd.DataFrame(table).to_string())

    names = list(scores)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, count, size=(2000, count))
    for first, name in itertools.combinations(names, 2):
        diff = scores[name] - scores[first]
        low, high = np.percentile(diff[boot].mean(1), [2.5, 97.5])
        print(f"{name} - {first}: dEM {100 * diff.mean():+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"
              f" | wins {int((diff > 0).sum())} losses {int((diff < 0).sum())}")


if __name__ == "__main__":
    main()
