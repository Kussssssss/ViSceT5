"""Run VQA training or prediction, configured through environment variables.

STAGE=finetune (default) | predict
Training:   DATASET_NAME, OUTPUT_DIR, NUM_TRAIN_EPOCHS, MODEL_NAME_OR_PATH,
            RESUME_FROM_CHECKPOINT, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
            module switches ABLATION_USE_QACLIP (QA-ViT), ABLATION_USE_OCR (Consformer), USE_TWC,
            TWC_ADV_PROBABILITY, TWC_TEMPERATURE, TWC_WEIGHT, LEARNING_RATE, NEW_MODULE_LR, SCALAR_LR,
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
        "PER_DEVICE_TRAIN_BATCH_SIZE": "per_device_train_batch_size",
        "GRADIENT_ACCUMULATION_STEPS": "gradient_accumulation_steps",
        "PER_DEVICE_EVAL_BATCH_SIZE": "per_device_eval_batch_size",
        "ABLATION_USE_QACLIP": "ablation_use_qaclip",
        "ABLATION_USE_OCR": "ablation_use_ocr",
        "USE_TWC": "use_twc",
        "TWC_WEIGHT": "twc_weight",
        "TWC_TEMPERATURE": "twc_temperature_init",
        "TWC_ADV_PROBABILITY": "twc_adv_probability",
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
