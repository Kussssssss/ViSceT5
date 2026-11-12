"""Where do empty answers come from? Follows one dataset through every loading step.

  1. raw JSON  (<data_dir>/processed/<name>/<name>_<split>.json, as downloaded)
  2. mapper    (data.dataset_hub._default_vqa_mapper: answer = first element of "answers" / "answer")
  3. CSV       (merged_{train,val,test}.csv in OUTPUT_PATH)
  4. pandas    (default read_csv vs training.common._read_split = keep_default_na=False)

Per split it prints how many annotations have no answer key, an empty list, a None / empty /
whitespace first answer while a later one is non-empty (lost by taking [0]), non-string answers,
answers that pandas' default turns into NaN ("None", "NA", "null", ...), rows whose image or OCR
file is missing, and rows lost between the JSON and the CSV. Examples are printed for each case.

Usage on Colab (env of the notebook cell 3):  python scripts/diagnose_answers.py
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import HfArgumentParser  # noqa: E402

from configs.arguments import CustomTrainingArguments, DataArguments, ModelArguments  # noqa: E402
from configs.base_config import OUTPUT_PATH  # noqa: E402
from run_pipeline import training_arguments  # noqa: E402
from training.common import _read_split, parse_args_with_yaml_and_cli  # noqa: E402

SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))
# pandas' default NA strings (pandas.io.parsers STR_NA_VALUES)
PANDAS_NA = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
             "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}


def blank(value):
    return value is None or (isinstance(value, float) and value != value) or \
        (isinstance(value, str) and not value.strip())


def annotations(data):
    raw = data.get("annotations", []) if isinstance(data, dict) else data
    if isinstance(raw, dict):
        return [{**v, "id": v.get("id", k)} for k, v in raw.items() if isinstance(v, dict)]
    return [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []


def show(title, items, limit=5):
    print(f"    {title}: {len(items)}")
    for item in items[:limit]:
        print(f"        {item}")


def check_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    anns = annotations(data)
    keys = Counter(k for a in anns for k in a)
    print(f"  JSON {path}: {len(anns)} annotations | keys {dict(keys.most_common(8))}")
    no_key, empty_list, first_blank, all_blank, not_str, na_like = [], [], [], [], [], []
    for a in anns:
        if "answers" in a:
            answers = a["answers"]
        elif "answer" in a:
            answers = [a["answer"]]
        else:
            no_key.append(a.get("id"))
            continue
        if not isinstance(answers, list):
            answers = [answers]
        if not answers:
            empty_list.append(a.get("id"))
            continue
        first = answers[0]
        if not isinstance(first, str) and first is not None:
            not_str.append((a.get("id"), type(first).__name__, repr(first)[:60]))
        if blank(first):
            (first_blank if any(not blank(x) for x in answers) else all_blank).append(
                (a.get("id"), answers[:4]))
        elif isinstance(first, str) and first.strip() in PANDAS_NA:
            na_like.append((a.get("id"), first))
    show("no 'answers' / 'answer' key", no_key)
    show("empty answer list", empty_list)
    show("first answer blank, a later one NOT (lost by answers[0])", first_blank)
    show("every answer blank", all_blank)
    show("first answer not a string", not_str)
    show("answer pandas' default reads as NaN", sorted(Counter(x for _, x in na_like).items()))
    return len(anns)


def check_csv(path, dataset, json_count):
    default = pd.read_csv(path)
    safe = _read_split(path)
    print(f"  CSV {path}: {len(safe)} rows (JSON {json_count}"
          f"{', LOST ' + str(json_count - len(safe)) if json_count is not None and json_count != len(safe) else ''})")
    if "dataset" in safe and set(safe["dataset"].dropna()) != {dataset}:
        print(f"    !! CSV belongs to {sorted(set(safe['dataset'].dropna()))}, not {dataset}: prepare again")
    turned = safe["answer"][default["answer"].isna() & safe["answer"].notna()]
    show("answers turned into NaN by pandas' default (fixed by _read_split)",
         sorted(Counter(turned.astype(str)).items()))
    empty = safe[safe["answer"].isna() | safe["answer"].astype(str).str.strip().eq("")]
    show("rows still without an answer after _read_split",
         [(r.get("id"), r.get("question"), r.get("all_answers")) for _, r in empty.iterrows()])
    for column in ("image_path", "ocr_path"):
        if column in safe:
            missing = [p for p in safe[column] if not (isinstance(p, str) and os.path.isfile(p))]
            show(f"missing {column}", missing, 3)
    if "question" in safe:
        show("rows without a question", safe[safe["question"].isna()]["id"].tolist())


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    _, data_args, _ = parse_args_with_yaml_and_cli(parser, training_arguments())
    name = data_args.dataset_name
    processed = Path(data_args.data_dir) / "processed" / name
    print(f"Dataset {name} | raw JSON in {processed} | CSV in {OUTPUT_PATH}")
    for split, suffix in SPLITS:
        print(f"\n[{split}]")
        json_path = processed / f"{name}_{split}.json"
        count = check_json(json_path) if json_path.is_file() else None
        if count is None:
            print(f"  JSON {json_path} not found")
        csv_path = Path(OUTPUT_PATH) / f"merged_{suffix}.csv"
        if csv_path.is_file():
            check_csv(csv_path, name, count)
        else:
            print(f"  CSV {csv_path} not found")
    print("\nTest answers are never read at predict; only train / validation need one.")


if __name__ == "__main__":
    main()
