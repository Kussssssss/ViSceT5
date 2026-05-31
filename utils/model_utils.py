"""
utils/model_utils.py
print_trainable_params(), safe_download_weights().
"""

import time
import gc
import torch
import torch.nn as nn
from collections import defaultdict
from huggingface_hub import snapshot_download

def safe_download_weights(repos, max_retries=3, delay=5):
    for repo in repos:
        print(f"📥 Downloading weights for {repo}...")
        for attempt in range(max_retries):
            try:
                snapshot_download(repo_id=repo, local_files_only=False)
                print(f"✅ Successfully downloaded {repo}")
                break
            except Exception as e:
                print(f"⚠️ Attempt {attempt+1}/{max_retries} failed for {repo}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"❌ Failed to download {repo} after {max_retries} attempts.")

def print_trainable_params(model: nn.Module, by_top_level: bool = True) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    def fmt(n):
        return f"{n:,} ({n/1e6:.2f}M)"

    print("\n" + "="*40)
    print("📊 MODEL PARAMETER SUMMARY")
    print("="*40)
    print(f"Total params    : {fmt(total)}")
    print(f"Trainable       : {fmt(trainable)} ({trainable/total:.2%})")
    print(f"Frozen          : {fmt(frozen)} ({frozen/total:.2%})")
    print("-" * 40)

    if by_top_level:
        buckets = defaultdict(int)
        for name, p in model.named_parameters():
            if p.requires_grad:
                top = name.split('.', 1)[0]
                buckets[top] += p.numel()