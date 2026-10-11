"""Evaluate an answer-generation checkpoint on validation or test data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Seq2SeqTrainingArguments, set_seed

from data.dataset import ViT5VQADataset
from training.common import build_collator, load_dataframes, load_model_and_tokenizer
from training.metrics import build_compute_metrics
from training.trainer import VQATrainer
from utils.debug_tools import visualize_samples


def main(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        "--model_name_or_path",
        dest="model_name_or_path",
        default="./output/finetune",
    )
    parser.add_argument("--dataset_name", default="ViTextVQA")
    parser.add_argument("--data_dir", default="./datasets")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output_dir", default="./output/evaluate")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--visualize_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(args_list)
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    dataset = ViT5VQADataset(
        load_dataframes(args.dataset_name, args.data_dir)[args.split]
    )
    if not len(dataset):
        raise ValueError(f"The {args.split} split is empty.")
    collator = build_collator(model, tokenizer)
    trainer = VQATrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        eval_dataset=dataset,
        compute_metrics=build_compute_metrics(tokenizer),
        args=Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.batch_size,
            remove_unused_columns=False,
            predict_with_generate=True,
            report_to="none",
            generation_max_length=model.config.generation_max_new_tokens,
            generation_num_beams=model.config.generation_num_beams,
        ),
    )
    metrics = trainer.evaluate(metric_key_prefix=args.split)
    trainer.log_metrics(args.split, metrics)
    trainer.save_metrics(args.split, metrics)
    if args.visualize_samples:
        visualize_samples(
            model, dataset, collator, tokenizer, args.visualize_samples, args.seed
        )
    return metrics


if __name__ == "__main__":
    main()
