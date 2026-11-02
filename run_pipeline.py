"""Run VQA training or prediction, configured through environment variables.

STAGE=finetune (default) | predict
Training:   DATASET_NAME, OUTPUT_DIR, NUM_TRAIN_EPOCHS, MODEL_NAME_OR_PATH,
            RESUME_FROM_CHECKPOINT, ABLATION_USE_IMAGE/QACLIP/OCR/OCR_AUG/CONSFORMER, USE_TWC,
            TWC_WEIGHT, TWC_LOSS, TWC_FEATURE, TWC_QUEUE_SIZE, TWC_QUEUE_START,
            TWC_MASK_SAME_TEXT, TWC_OBJECTIVE, TWC_ANSWER_WEIGHT, TWC_RELATED_TRAIN, OCR_FEATURE_NORM, QAVIT_MODE (off|no_question|question),
            TWC_TEMPERATURE, TWC_ADV_PROBABILITY, TWC_VIEW, OCR_AUG_MODE, OCR_RELATED, LEARNING_RATE, NEW_MODULE_LR, SCALAR_LR,
            MOCK_TEST (+ SMOKE_*), HF_TOKEN + HF_REPO (push/resume/upload).
Prediction: PREDICT_CKPT_DIR, PREDICT_DATASET, PREDICT_SPLIT, PREDICT_BATCH_SIZE.
"""

import os
import re
import subprocess
import sys


def _env(name):
    return os.environ.get(name, "").strip()


def training_arguments():
    arguments = [_env("TRAIN_CONFIG") or "configs/finetune.yaml"]
    overrides = {
        "DATASET_NAME": "dataset_name",
        "NUM_TRAIN_EPOCHS": "num_train_epochs",
        "MODEL_NAME_OR_PATH": "model_name_or_path",
        "RESUME_FROM_CHECKPOINT": "resume_from_checkpoint",
        "RESUME_CHECKPOINT_ID": "resume_checkpoint_id",
        "PER_DEVICE_EVAL_BATCH_SIZE": "per_device_eval_batch_size",
        "ABLATION_USE_IMAGE": "ablation_use_image",
        "ABLATION_USE_QACLIP": "ablation_use_qaclip",
        "ABLATION_USE_OCR": "ablation_use_ocr",
        "ABLATION_USE_CONSFORMER": "ablation_use_consformer",
        "ABLATION_USE_OCR_AUG": "ablation_use_ocr_aug",
        "USE_TWC": "use_twc",
        "TWC_WEIGHT": "twc_weight",
        "TWC_LOSS": "twc_loss",
        "TWC_FEATURE": "twc_feature",
        "TWC_QUEUE_SIZE": "twc_queue_size",
        "TWC_QUEUE_START": "twc_queue_start",
        "TWC_MASK_SAME_TEXT": "twc_mask_same_text",
        "TWC_OBJECTIVE": "twc_objective",
        "TWC_ANSWER_WEIGHT": "twc_answer_weight",
        "TWC_RELATED_TRAIN": "twc_related_train",
        "OCR_FEATURE_NORM": "ocr_feature_norm",
        "QAVIT_MODE": "qavit_mode",
        "TWC_TEMPERATURE": "twc_temperature_init",
        "TWC_ADV_PROBABILITY": "twc_adv_probability",
        "TWC_VIEW": "twc_view",
        "OCR_AUG_MODE": "ocr_aug_mode",
        "OCR_RELATED": "ocr_related",
        "LEARNING_RATE": "learning_rate",
        "NEW_MODULE_LR": "new_module_learning_rate",
        "SCALAR_LR": "scalar_learning_rate",
        "SMOKE_TRAIN_SAMPLES": "smoke_train_samples",
        "SMOKE_EVAL_SAMPLES": "smoke_eval_samples",
        "SMOKE_MAX_STEPS": "smoke_max_steps",
    }
    for variable, flag in overrides.items():
        if _env(variable):
            arguments.extend([f"--{flag}", _env(variable)])
    if not _env("QAVIT_MODE") and (_env("CLIP_LATE_ADAPTER") or _env("ABLATION_USE_QACLIP")):
        # Notebooks from before QAVIT_MODE: derive the single QA-ViT switch so YAML cannot override it.
        question = _env("ABLATION_USE_QACLIP").lower() in ("1", "true", "yes")
        layers = _env("CLIP_LATE_ADAPTER").lower() in ("1", "true", "yes")
        arguments.extend(["--qavit_mode", "question" if question else ("no_question" if layers else "off")])
    if _env("OUTPUT_DIR"):
        output_dir = _env("OUTPUT_DIR")
        arguments.extend(["--output_dir", output_dir, "--logging_dir", output_dir.rstrip("/") + "/logs"])
    if _env("MOCK_TEST").lower() == "true":
        arguments.extend(["--smoke_test", "True"])
    return arguments


def upload_model(output_dir):
    token, repository = _env("HF_TOKEN"), _env("HF_REPO")
    if not (token and repository) or _env("MOCK_TEST").lower() == "true":
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repository, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=output_dir, repo_id=repository, repo_type="model")
    print(f"Uploaded VQA checkpoint to {repository}.")


def stop_vast_instance():
    label = os.environ.get("VAST_CONTAINERLABEL", "")
    api_key = os.environ.get("CONTAINER_API_KEY", "")
    if not (label and api_key):
        return
    match = re.search(r"(\d+)(?:\D*)$", label)
    if not match:
        raise ValueError(f"Cannot determine Vast.ai instance ID from {label!r}.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vastai"], check=True)
    subprocess.run(
        ["vastai", "stop", "instance", match.group(1), "--api-key", api_key], check=True
    )


def run():
    try:
        if _env("STAGE").lower() == "predict":
            from training.predict import main as predict

            arguments = ["--ckpt_dir", _env("PREDICT_CKPT_DIR") or "./output/finetune",
                         "--dataset", _env("PREDICT_DATASET") or "ViTextVQA",
                         "--split", _env("PREDICT_SPLIT") or "both"]
            if _env("PREDICT_BATCH_SIZE"):
                arguments += ["--batch_size", _env("PREDICT_BATCH_SIZE")]
            if _env("PREDICT_LIMIT"):
                arguments += ["--limit", _env("PREDICT_LIMIT")]
            return predict(arguments)
        from training.finetune import main

        trainer = main(training_arguments())
        upload_model(trainer.args.output_dir)
        return trainer
    finally:
        try:
            stop_vast_instance()
        except Exception as error:
            print(f"Could not automatically stop Vast.ai instance: {error}")


if __name__ == "__main__":
    run()
