"""
scripts/init_model.py
Download weights and initialize OpenViVQAModel. Run: python scripts/init_model.py
"""

#!/usr/bin/env python
import gc
import torch
from configs.model_config import OpenViVQAConfig
from models.openvivqa_model import OpenViVQAModel
from utils.model_utils import safe_download_weights, print_trainable_params

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

    clear_output()
    print("✅ Model loaded successfully on " + str(model.device))
    print_trainable_params(model)

except RuntimeError as e:
    if "out of memory" in str(e):
        print("❌ LỖI OOM (Out Of Memory)! Thử giảm batch size hoặc kiểm tra lại VRAM.")
        print(e)
    else:
        print("❌ LỖI KHỞI TẠO MODEL:", e)