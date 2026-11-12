"""Data check before training, for any dataset (ViTextVQA, OpenViVQA, ViOCRVQA, ...).

Same configuration as a run (the env variables of the notebook cell 3). Per split: rows, missing
image / OCR files, rows without an answer (dropped from train / validation by finetune), and the
share of answers longer than text_max_target_length tokens (cut in training, and the generation
limit at predict).

Usage on Colab:  python scripts/check_data.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer, HfArgumentParser  # noqa: E402

from configs.arguments import CustomTrainingArguments, DataArguments, ModelArguments  # noqa: E402
from configs.model_config import OpenViVQAConfig  # noqa: E402
from run_pipeline import training_arguments  # noqa: E402
from training.common import load_dataframes, parse_args_with_yaml_and_cli  # noqa: E402


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    _, data_args, training_args = parse_args_with_yaml_and_cli(parser, training_arguments())
    config = OpenViVQAConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.vit5_name)
    limit = config.text_max_target_length
    frames = load_dataframes(data_args.dataset_name, data_args.data_dir)
    print(f"Dataset {data_args.dataset_name} | answer limit {limit} tokens | "
          f"generation_max_length {training_args.generation_max_length}")
    for split, frame in frames.items():
        if not len(frame):
            print(f"  {split:10s} empty")
            continue
        images = sum(not (isinstance(p, str) and os.path.isfile(p)) for p in frame["image_path"])
        ocr = sum(not (isinstance(p, str) and os.path.isfile(p)) for p in frame.get("ocr_path", []))
        answers = frame["answer"].fillna("").astype(str)
        empty = int(answers.str.strip().eq("").sum())
        lengths = [len(ids) for ids in tokenizer(answers[answers.str.strip().ne("")].tolist()).input_ids]
        long = sum(length > limit for length in lengths)
        lengths.sort()
        p95 = lengths[int(0.95 * (len(lengths) - 1))] if lengths else 0
        print(f"  {split:10s} rows {len(frame):6d} | missing image {images:5d} | missing OCR {ocr:5d} | "
              f"no answer {empty:5d} | answer tokens p95 {p95:3d}, max {lengths[-1] if lengths else 0:3d}, "
              f"> {limit}: {long} ({100 * long / max(len(lengths), 1):.2f}%)")
    print("Rows without an answer are dropped from train / validation at finetune. Test answers are never read.")


if __name__ == "__main__":
    main()
