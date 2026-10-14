"""Keep a training run on the Hugging Face Hub: push every checkpoint, resume after a
disconnect (Colab/Kaggle). Enabled only when HF_TOKEN and HF_REPO are set."""

import os

import torch
from transformers import TrainerCallback

INFERENCE_FILES = [
    "config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
    "spiece.model", "tokenizer.json", "tokenizer_config.json", "added_tokens.json",
]


def hub_target():
    token, repo = os.environ.get("HF_TOKEN", "").strip(), os.environ.get("HF_REPO", "").strip()
    return (token, repo) if token and repo else (None, None)


class PushEachCheckpoint(TrainerCallback):
    """Push each checkpoint (with optimizer state, so it can be resumed) right after saving.
    A checkpoint with NaN/inf weights is not pushed, so resume never reloads a broken state."""

    def __init__(self, repo, token):
        from huggingface_hub import HfApi

        self.api, self.repo = HfApi(token=token), repo
        self.api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)

    def on_save(self, args, state, control, model=None, **kwargs):
        name = f"checkpoint-{state.global_step}"
        folder = os.path.join(args.output_dir, name)
        if not os.path.isdir(folder):
            return
        if model is not None and any(not torch.isfinite(p).all() for p in model.parameters()):
            print(f"[HF] {name} has non-finite weights — not pushed.")
            return
        try:
            self.api.upload_folder(folder_path=folder, path_in_repo=name, repo_id=self.repo,
                                   repo_type="model")
            print(f"[HF] pushed {name} → {self.repo}")
        except Exception as error:
            print(f"[HF] push of {name} failed: {error}")


def resume_from_hub(repo, token, output_dir):
    """Download the latest full checkpoint (and the inference files of the best one, so
    load_best_model_at_end still finds it) and return its local path, or None."""
    import json

    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    files = set(list_repo_files(repo, token=token))
    checkpoints = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-")},
                         key=lambda name: int(name.split("-")[1]))
    if not checkpoints or f"{checkpoints[-1]}/optimizer.pt" not in files:
        return None
    latest = checkpoints[-1]
    patterns = [f"{latest}/*"]
    state = json.load(open(hf_hub_download(repo, f"{latest}/trainer_state.json", token=token)))
    best = os.path.basename(state.get("best_model_checkpoint") or "")
    if best and best != latest and best in checkpoints:
        patterns += [f"{best}/{name}" for name in INFERENCE_FILES + ["trainer_state.json"]]
    print(f"[HF] resuming from {latest} (best so far: {best or '-'})")
    snapshot_download(repo, repo_type="model", token=token, local_dir=output_dir,
                      allow_patterns=patterns)
    return os.path.join(output_dir, latest)
