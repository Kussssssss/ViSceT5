"""
scripts/env_report.py
Ghi lại môi trường chạy (GPU, CUDA, torch, transformers, cờ TF32, commit) ra JSON để so
giữa các máy (Vast / Colab / Kaggle).

  python scripts/env_report.py --repo_dir /content/ViSceT5_old --out env.json
"""
import argparse
import json
import platform
import subprocess
from importlib import metadata


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", default=".")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "packages": {p: _version(p) for p in ("transformers", "accelerate", "huggingface_hub", "tokenizers",
                                             "safetensors", "numpy", "pandas", "Pillow", "sentencepiece")},
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": p.name, "capability": f"{p.major}.{p.minor}",
                       "memory_gb": round(p.total_memory / 2**30, 1), "count": torch.cuda.device_count()}
    try:
        info["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.strip()
    except OSError:
        info["driver"] = None
    try:
        info["commit"] = subprocess.run(["git", "-C", a.repo_dir, "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()
    except OSError:
        info["commit"] = None

    print(json.dumps(info, indent=2, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
