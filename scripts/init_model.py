"""
scripts/init_model.py
Download weights and initialize OpenViVQAModel. Run: python scripts/init_model.py
"""
#!/usr/bin/env python
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gc
import torch
from configs.model_config import OpenViVQAConfig
from models.openvivqa_model import OpenViVQAModel
from utils.model_utils import safe_download_weights, print_trainable_params

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output():
        pass

def main():
    repos_to_download = [
        "openai/clip-vit-base-patch16",
        "VietAI/vit5-base"
    ]

    try:
        safe_download_weights(repos_to_download)
    except RuntimeError as e:
        print(e)

    print("🧹 Cleaning memory before loading...")
    gc.collect()
    torch.cuda.empty_cache()

    print("🚀 Initializing OpenViVQAModel...")

    config = OpenViVQAConfig()

    try:
        model = OpenViVQAModel(config)
        model.train()

        if torch.cuda.is_available():
            print("⚡ Moving model to CUDA...")
            model.cuda()

            gc.collect()
            torch.cuda.empty_cache()

        print("✅ Model loaded successfully on " + str(model.device))
        print_trainable_params(model)

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ LỖI OOM (Out Of Memory)! Thử giảm batch size hoặc kiểm tra lại VRAM.")
            print(e)
        else:
            print("❌ LỖI KHỞI TẠO MODEL:", e)

if __name__ == "__main__":
    main()