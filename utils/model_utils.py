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


def patch_transformers_convert_to_native_format():
    """
    Fix KeyError: 0 in transformers >= 4.49 (PR #44452 / issue #44451)
    where convert_to_native_format crashes on dictionary vocabularies:
      File ".../tokenization_utils_tokenizers.py", line 127, in convert_to_native_format
        if vocab and isinstance(vocab[0], (list, tuple)):
    """
    def _wrap(fn):
        def _inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except KeyError:
                return kwargs if kwargs else (args[0] if args else {})
        return _inner

    # 1. Patch TokenizersBackend / tokenization_utils_tokenizers
    try:
        import transformers.tokenization_utils_tokenizers as tut
        for attr in dir(tut):
            val = getattr(tut, attr)
            if hasattr(val, "convert_to_native_format"):
                orig = getattr(val, "convert_to_native_format")
                try:
                    setattr(val, "convert_to_native_format", classmethod(_wrap(orig)))
                except Exception:
                    pass
            elif attr == "convert_to_native_format" and callable(val):
                try:
                    setattr(tut, attr, _wrap(val))
                except Exception:
                    pass
    except Exception:
        pass

    # 2. Patch PreTrainedTokenizerBase / tokenization_utils_base
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if hasattr(PreTrainedTokenizerBase, "convert_to_native_format"):
            orig = getattr(PreTrainedTokenizerBase, "convert_to_native_format")
            try:
                setattr(PreTrainedTokenizerBase, "convert_to_native_format", classmethod(_wrap(orig)))
            except Exception:
                pass
    except Exception:
        pass

    # 3. Patch T5Tokenizer if present
    try:
        from transformers.models.t5.tokenization_t5 import T5Tokenizer
        if hasattr(T5Tokenizer, "convert_to_native_format"):
            orig = getattr(T5Tokenizer, "convert_to_native_format")
            try:
                setattr(T5Tokenizer, "convert_to_native_format", classmethod(_wrap(orig)))
            except Exception:
                pass
    except Exception:
        pass


def safe_load_tokenizer(model_name_or_path, local_files_only=False, use_fast=False):
    """
    Robust tokenizer loader that handles transformers >= 4.49 KeyError: 0
    with multiple fallback strategies.
    """
    patch_transformers_convert_to_native_format()
    from transformers import AutoTokenizer

    # Attempt 1: AutoTokenizer with specified use_fast
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only, use_fast=use_fast)
    except KeyError as e:
        print(f"⚠️ AutoTokenizer(use_fast={use_fast}) hit KeyError: {e}. Retrying with fast={not use_fast}...")
    except Exception as e:
        print(f"⚠️ AutoTokenizer(use_fast={use_fast}) failed ({e}). Retrying with fast={not use_fast}...")

    # Attempt 2: AutoTokenizer with inverted use_fast
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only, use_fast=(not use_fast))
    except Exception as e:
        print(f"⚠️ AutoTokenizer(use_fast={not use_fast}) failed ({e}). Trying T5Tokenizer directly...")

    # Attempt 3: Direct T5Tokenizer
    try:
        from transformers import T5Tokenizer
        return T5Tokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only, legacy=False)
    except Exception:
        from transformers import T5Tokenizer
        return T5Tokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)