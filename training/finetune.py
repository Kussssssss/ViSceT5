"""Train ViSceT5 on question/answer supervision (+ optional TWC auxiliary loss)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import HfArgumentParser, set_seed

from configs.arguments import CustomTrainingArguments, DataArguments, ModelArguments
from data.dataset import ViT5VQADataset
from training.common import (
    PROJECT_ROOT,
    build_collator,
    build_corrector,
    load_dataframes,
    load_model_and_tokenizer,
    parse_args_with_yaml_and_cli,
)
from training.hub_sync import PushEachCheckpoint, hub_target, resume_from_hub
from training.metrics import build_compute_metrics, java_available
from training.trainer import VQATrainer
from utils.debug_tools import print_parameter_summary
from utils.io_utils import download_and_extract_checkpoint

ARCHITECTURE_FLAGS = ("ablation_use_image", "ablation_use_qaclip", "ablation_use_ocr",
                      "ablation_use_ocr_aug", "use_twc", "twc_weight", "twc_loss", "twc_feature",
                      "twc_queue_size", "twc_queue_start", "twc_mask_same_text", "twc_objective",
                      "twc_answer_weight", "twc_related_train", "ocr_feature_norm", "clip_late_adapter",
                      "twc_temperature_init", "twc_adv_probability")


def main(args_list=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parse_args_with_yaml_and_cli(
        parser,
        args_list,
        PROJECT_ROOT / "configs" / "finetune.yaml",
    )
    # DataParallel chia sai các danh sách (ocr_info) giữa GPU → luôn chạy 1 GPU.
    if training_args.n_gpu > 1:
        training_args._n_gpu = 1
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

    # Checkpoint tốt nhất: trung bình EM và F1 trên val (OpenViVQA: CIDEr), không theo loss.
    best_metric = os.environ.get("METRIC_FOR_BEST_MODEL", "").strip() or (
        "cider" if data_args.dataset_name == "OpenViVQA" else "em_f1"
    )
    if best_metric == "cider" and not java_available():
        print("[finetune] CIDEr needs Java, which is missing -> best checkpoint by em_f1 instead.")
        best_metric = "em_f1"
    training_args.metric_for_best_model = best_metric
    training_args.greater_is_better = best_metric not in ("loss", "eval_loss")

    token, repo = hub_target()
    if training_args.resume_checkpoint_id:
        checkpoint = str(Path(training_args.output_dir) / "resume_ckpt")
        if not download_and_extract_checkpoint(
            training_args.resume_checkpoint_id, checkpoint
        ):
            raise RuntimeError("Could not download the resume checkpoint.")
        training_args.resume_from_checkpoint = checkpoint
    elif (token and not training_args.resume_from_checkpoint and not training_args.smoke_test
          and os.environ.get("RESUME_FROM_HF", "1").lower() not in ("0", "false", "no")):
        try:
            training_args.resume_from_checkpoint = resume_from_hub(
                repo, token, training_args.output_dir
            )
        except Exception as error:
            print(f"[HF] resume skipped ({type(error).__name__}: {error})")

    flags = {name: getattr(model_args, name) for name in ARCHITECTURE_FLAGS}
    checkpoint = model_args.model_name_or_path or training_args.resume_from_checkpoint
    model, tokenizer = load_model_and_tokenizer(checkpoint, overrides=flags)
    if training_args.should_log:
        print("Architecture:", {name: getattr(model.config, name) for name in ARCHITECTURE_FLAGS},
              "| best checkpoint by:", best_metric)
        print_parameter_summary(model)

    corrector = None
    if model.config.ablation_use_ocr_aug:
        corrector = build_corrector(
            model.config, frames["train"], data_args.term_vocab_path, data_args.viet_vocab_path
        )
    trainer = VQATrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=ViT5VQADataset(frames["train"]),
        eval_dataset=ViT5VQADataset(frames["validation"]),
        data_collator=build_collator(model, tokenizer, corrector, training=True,
                                     seed=training_args.seed),
        eval_data_collator=build_collator(model, tokenizer, corrector, training=False),
        compute_metrics=build_compute_metrics(tokenizer)
        if training_args.predict_with_generate
        else None,
    )
    if token and not training_args.smoke_test:
        trainer.add_callback(PushEachCheckpoint(repo, token))
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
