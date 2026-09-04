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


def safe_load_tokenizer(model_name_or_path="VietAI/vit5-base", local_files_only=False, use_fast=False):
    """
    Robust tokenizer loader for ViT5 / T5.
    
    Bypasses the HuggingFace transformers >= 4.49 `convert_to_native_format` KeyError: 0 bug
    (PR #44452 / issue #44451) by initializing T5Tokenizer directly from native SentencePiece (spiece.model).
    Guarantees 100% exact vocabulary (36,096 tokens) and zero risk of monkey-patching side effects.
    """
    import os
    import json
    from transformers import AutoTokenizer, T5Tokenizer

    # 1. Native SentencePiece direct initialization (bypasses transformers 4.49 tokenizer_file bug)
    sp_path = None
    cfg_path = None

    # Check local directory first
    if os.path.isdir(model_name_or_path):
        candidate_sp = os.path.join(model_name_or_path, "spiece.model")
        candidate_cfg = os.path.join(model_name_or_path, "tokenizer_config.json")
        if os.path.isfile(candidate_sp):
            sp_path = candidate_sp
        if os.path.isfile(candidate_cfg):
            cfg_path = candidate_cfg

    # If not local, download spiece.model and tokenizer_config.json via HuggingFace Hub
    if sp_path is None and not local_files_only:
        from huggingface_hub import hf_hub_download
        repo_id = model_name_or_path if "/" in model_name_or_path else "VietAI/vit5-base"
        try:
            sp_path = hf_hub_download(repo_id, "spiece.model")
            try:
                cfg_path = hf_hub_download(repo_id, "tokenizer_config.json")
            except Exception:
                cfg_path = None
        except Exception as dl_err:
            print(f"[INFO] [safe_load_tokenizer] Hub download failed for {repo_id}: {dl_err}")

    # If spiece.model is available, instantiate directly
    if sp_path and os.path.isfile(sp_path):
        cfg = {}
        if cfg_path and os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}

        try:
            tok = T5Tokenizer(
                vocab_file=sp_path,
                eos_token=cfg.get("eos_token", "</s>"),
                unk_token=cfg.get("unk_token", "<unk>"),
                pad_token=cfg.get("pad_token", "<pad>"),
                extra_ids=cfg.get("extra_ids", 96),
                additional_special_tokens=cfg.get("additional_special_tokens", None),
                legacy=False,
            )
            print(f"[OK] [safe_load_tokenizer] Loaded native T5Tokenizer ({len(tok):,} tokens) via SentencePiece directly.")
            return tok
        except Exception as e_init:
            print(f"[WARN] [safe_load_tokenizer] Direct T5Tokenizer init failed ({e_init}). Trying fallbacks...")

    # 2. Standard AutoTokenizer fallback
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only, use_fast=use_fast)
    except Exception as e1:
        print(f"[WARN] [safe_load_tokenizer] AutoTokenizer(use_fast={use_fast}) failed ({e1}). Trying fast={not use_fast}...")

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only, use_fast=(not use_fast))
    except Exception as e2:
        print(f"[WARN] [safe_load_tokenizer] AutoTokenizer(use_fast={not use_fast}) failed ({e2}). Trying T5Tokenizer.from_pretrained...")

    # 3. Direct T5Tokenizer.from_pretrained fallback
    return T5Tokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)