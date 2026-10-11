"""Train ViSceT5 directly on question/answer supervision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import HfArgumentParser, set_seed

from configs.arguments import CustomTrainingArguments, DataArguments, ModelArguments
from data.dataset import ViT5VQADataset
from training.common import (
    PROJECT_ROOT,
    build_collator,
    load_dataframes,
    load_model_and_tokenizer,
    parse_args_with_yaml_and_cli,
)
from training.metrics import build_compute_metrics
from training.trainer import VQATrainer
from utils.io_utils import download_and_extract_checkpoint


def main(args_list=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parse_args_with_yaml_and_cli(
        parser,
        args_list,
        PROJECT_ROOT / "configs" / "finetune.yaml",
    )
    set_seed(training_args.seed)
    frames = load_dataframes(data_args.dataset_name, data_args.data_dir)
    if training_args.smoke_test:
        frames["train"] = frames["train"].head(training_args.smoke_train_samples)
        frames["validation"] = frames["validation"].head(
            training_args.smoke_eval_samples
        )
        training_args.max_steps = training_args.smoke_max_steps
        training_args.logging_steps = 1
        training_args.eval_strategy = "steps"
        training_args.save_strategy = "steps"
        training_args.eval_steps = training_args.smoke_max_steps
        training_args.save_steps = training_args.smoke_max_steps
        training_args.load_best_model_at_end = False
        training_args.dataloader_num_workers = 0
    if frames["train"].empty or frames["validation"].empty:
        raise ValueError("Both train and validation splits must contain samples.")

    if training_args.resume_checkpoint_id:
        checkpoint = str(Path(training_args.output_dir) / "resume_ckpt")
        if not download_and_extract_checkpoint(
            training_args.resume_checkpoint_id, checkpoint
        ):
            raise RuntimeError("Could not download the resume checkpoint.")
        training_args.resume_from_checkpoint = checkpoint
    checkpoint = model_args.model_name_or_path or training_args.resume_from_checkpoint
    model, tokenizer = load_model_and_tokenizer(checkpoint)
    for name in ("ablation_use_qaclip", "ablation_use_vs", "ablation_use_ocr"):
        setattr(model.config, name, getattr(model_args, name))

    trainer = VQATrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=ViT5VQADataset(frames["train"]),
        eval_dataset=ViT5VQADataset(frames["validation"]),
        data_collator=build_collator(model, tokenizer),
        compute_metrics=build_compute_metrics(tokenizer)
        if training_args.predict_with_generate
        else None,
    )
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    model.image_processor.save_pretrained(
        Path(training_args.output_dir) / "image_processor"
    )
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    return trainer


if __name__ == "__main__":
    main()
