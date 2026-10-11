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
    # Dựng đúng cấu hình sẽ huấn luyện (cùng biến môi trường mà run_pipeline chuyển cho
    # finetune) để bảng tham số chỉ đếm các module thực sự dùng.
    _stage = os.environ.get("STAGE", "finetune").strip().lower()
    config.pretrain = _stage == "pretrain"
    if not config.pretrain:
        config.use_twc = False
    for _env, _attr in [("ABLATION_USE_QACLIP", "ablation_use_qaclip"),
                        ("ABLATION_USE_VS", "ablation_use_vs"),
                        ("ABLATION_USE_OCR", "ablation_use_ocr"),
                        ("ABLATION_USE_OCR_INPUT", "ablation_use_ocr_input"),
                        ("ABLATION_USE_IMAGE", "ablation_use_image")]:
        _v = os.environ.get(_env, "").strip().lower()
        setattr(config, _attr, _v not in ("false", "0", "no") if _v else True)

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