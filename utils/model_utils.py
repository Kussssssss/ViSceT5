"""
utils/model_utils.py
print_trainable_params(), safe_download_weights().
"""

import time
import gc
from collections import defaultdict
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

import time
import gc
import torch
import torch.nn as nn
from collections import defaultdict
from huggingface_hub import snapshot_download
from IPython.display import clear_output

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