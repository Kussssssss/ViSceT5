"""
scripts/hf_io.py
Tải model tốt nhất / upload kết quả predict của một cấu hình lên HF Hub.

Chạy bằng tiến trình riêng (`!python scripts/hf_io.py ...` trong Colab) thay vì import trong
kernel: setup.sh cài lại huggingface_hub sau khi kernel đã import bản có sẵn của Colab, nên
import thêm trong kernel trộn hai phiên bản (ImportError: reset_sessions ...).

  python scripts/hf_io.py download_best --repo USER/REPO --dir OUTPUT_DIR
  python scripts/hf_io.py upload_predictions --repo USER/REPO --dir OUTPUT_DIR --files a.csv b.csv
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download

# Model tốt nhất nằm ở thư mục gốc của repo (trainer.save_model sau load_best_model_at_end);
# chỉ lấy file cần cho inference, không tải checkpoint-* (kèm optimizer, ~15GB).
INFERENCE_FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "special_tokens_map.json", "spiece.model", "tokenizer.json",
    "tokenizer_config.json", "added_tokens.json", "image_processor/*",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["download_best", "upload_predictions"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--path_in_repo", default="predictions")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or None
    model_path = os.path.join(a.dir, "model.safetensors")

    if a.action == "download_best":
        if os.path.exists(model_path):
            print(f"✅ Đã có model trong {a.dir} → không tải lại.")
            return
        print(f"⬇️ Tải model tốt nhất (thư mục gốc) từ {a.repo} → {a.dir}")
        snapshot_download(repo_id=a.repo, repo_type="model", local_dir=a.dir, token=token,
                          allow_patterns=INFERENCE_FILES)
        if not os.path.exists(model_path):
            sys.exit(f"❌ Repo {a.repo} không có model.safetensors ở thư mục gốc.")
        print(f"✅ {sorted(os.listdir(a.dir))}")
        return

    if not token:
        sys.exit("❌ Thiếu HF_TOKEN → không upload được; hãy tải file về trước khi ngắt phiên.")
    api = HfApi(token=token)
    for f in a.files:
        p = os.path.join(a.dir, f)
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"{a.path_in_repo}/{f}",
                            repo_id=a.repo, repo_type="model")
            print(f"↑ {a.repo}/{a.path_in_repo}/{f}")


if __name__ == "__main__":
    main()
