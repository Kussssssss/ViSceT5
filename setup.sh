#!/bin/bash
# setup.sh — Environment setup for OpenViVQA on Kaggle/Colab/Ubuntu

set -e

# apt fail-fast: một số host Vast không ra được archive.ubuntu.com → apt treo ở
# "Waiting for headers" theo timeout mặc định (~120s × retry). Hạ timeout + 1 retry
# để mirror hỏng bị bỏ qua nhanh; '|| true' để set -e không giết run khi update
# trả nonzero do Ign (danh sách cache/mirror khác vẫn đủ cài).
APT_OPTS="-o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20"

# Java cho pycocoevalcap (BLEU/CIDEr). CHỈ cài JRE HEADLESS + --no-install-recommends:
# openjdk-17-JDK (full) kéo cả deps GUI (dconf/gsettings...) ~132MB → trên host mạng
# yếu 1 file fail là hỏng cả install. JRE headless ~1/10 dung lượng, không deps GUI.
# Bỏ git-lfs: repo clone bằng git thường, checkpoint tải qua huggingface_hub (không cần LFS).
# EM/F1 KHÔNG cần java (metrics.py tự tắt mềm BLEU/CIDEr nếu java vắng) → cài lỗi cũng chạy được.
apt-get $APT_OPTS update -y || true
apt-get $APT_OPTS install -y --no-install-recommends openjdk-17-jre-headless || {
    echo "⚠️ [setup] cài java lỗi (mirror yếu) — thử lại 1 lần rồi BỎ QUA (BLEU/CIDEr tắt, EM/F1 vẫn chạy)."
    apt-get $APT_OPTS update -y --fix-missing || true
    apt-get $APT_OPTS install -y --no-install-recommends --fix-missing openjdk-17-jre-headless || true
}

export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"

# Python deps
pip uninstall -y transformers peft accelerate 2>/dev/null || true
pip install --quiet -r requirements.txt
pip install --quiet git+https://github.com/salaniz/pycocoevalcap

echo "✅ Setup complete"
