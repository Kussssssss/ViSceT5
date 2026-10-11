"""Run VQA training with optional Hub upload and Vast.ai shutdown."""

import os
import re
import subprocess
import sys


def training_arguments():
    arguments = [os.environ.get("TRAIN_CONFIG", "configs/finetune.yaml")]
    overrides = {
        "NUM_TRAIN_EPOCHS": "num_train_epochs",
        "MODEL_NAME_OR_PATH": "model_name_or_path",
        "RESUME_FROM_CHECKPOINT": "resume_from_checkpoint",
        "RESUME_CHECKPOINT_ID": "resume_checkpoint_id",
        "SMOKE_TRAIN_SAMPLES": "smoke_train_samples",
        "SMOKE_EVAL_SAMPLES": "smoke_eval_samples",
        "SMOKE_MAX_STEPS": "smoke_max_steps",
    }
    for variable, flag in overrides.items():
        value = os.environ.get(variable, "").strip()
        if value:
            arguments.extend([f"--{flag}", value])
    if os.environ.get("MOCK_TEST", "").lower() == "true":
        arguments.extend(["--smoke_test", "True"])
    return arguments


def upload_model(output_dir):
    token, repository = os.environ.get("HF_TOKEN"), os.environ.get("HF_REPO")
    if not (token and repository):
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
