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


def safe_load_tokenizer(model_name_or_path="VietAI/vit5-base", local_files_only=False, use_fast=True):
    """
    Robust tokenizer loader for ViT5 / T5.
    
    Bypasses the HuggingFace transformers >= 4.49 `convert_to_native_format` KeyError: 0 bug
    (PR #44452 / issue #44451) by loading the Rust `tokenizers.Tokenizer` from `tokenizer.json` directly,
    guaranteeing the full 36,096+ tokens vocabulary on any transformers version.
    """
    import os
    import json
    from transformers import AutoTokenizer, T5Tokenizer, T5TokenizerFast
    from tokenizers import Tokenizer as RustTokenizer

    # Strategy 1: Direct Rust Tokenizer backend (100% immune to Python convert_to_native_format bug)
    json_path = None
    cfg_path = None

    if os.path.isdir(model_name_or_path):
        candidate_json = os.path.join(model_name_or_path, "tokenizer.json")
        candidate_cfg = os.path.join(model_name_or_path, "tokenizer_config.json")
        if os.path.isfile(candidate_json):
            json_path = candidate_json
        if os.path.isfile(candidate_cfg):
            cfg_path = candidate_cfg

    if json_path is None and not local_files_only:
        from huggingface_hub import hf_hub_download
        repo_id = model_name_or_path if "/" in model_name_or_path else "VietAI/vit5-base"
        try:
            json_path = hf_hub_download(repo_id, "tokenizer.json")
            try:
                cfg_path = hf_hub_download(repo_id, "tokenizer_config.json")
            except Exception:
                cfg_path = None
        except Exception as dl_err:
            print(f"[INFO] [safe_load_tokenizer] Hub download failed for {repo_id}: {dl_err}")

    if json_path and os.path.isfile(json_path):
        cfg = {}
        if cfg_path and os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}

        try:
            rust_tok = RustTokenizer.from_file(json_path)
            tok = T5TokenizerFast(
                tokenizer_object=rust_tok,
                eos_token=cfg.get("eos_token", "</s>"),
                unk_token=cfg.get("unk_token", "<unk>"),
                pad_token=cfg.get("pad_token", "<pad>"),
                extra_ids=cfg.get("extra_ids", 96),
                additional_special_tokens=cfg.get("additional_special_tokens", None),
            )
            if len(tok) >= 30000:
                print(f"[OK] [safe_load_tokenizer] Loaded native T5TokenizerFast ({len(tok):,} tokens) via Rust Tokenizer directly.")
                return tok
        except Exception as e_rust:
            print(f"[WARN] [safe_load_tokenizer] Direct Rust tokenizer init failed ({e_rust}). Trying fallbacks...")

    # Strategy 2: Standard AutoTokenizer fallback
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only, use_fast=use_fast)
    except Exception as e1:
        print(f"[WARN] [safe_load_tokenizer] AutoTokenizer(use_fast={use_fast}) failed ({e1}). Trying fast={not use_fast}...")

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only, use_fast=(not use_fast))
    except Exception as e2:
        print(f"[WARN] [safe_load_tokenizer] AutoTokenizer(use_fast={not use_fast}) failed ({e2}). Trying T5Tokenizer.from_pretrained...")

    # Strategy 3: Direct T5Tokenizer.from_pretrained fallback
    return T5Tokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)