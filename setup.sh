#!/bin/bash
# setup.sh — Environment setup for OpenViVQA on Kaggle/Colab/Ubuntu

set -e

# apt fail-fast: một số host Vast không ra được archive.ubuntu.com → apt treo ở
# "Waiting for headers" theo timeout mặc định (~120s × retry). Hạ timeout + 1 retry
# để mirror hỏng bị bỏ qua nhanh; '|| true' để set -e không giết run khi update
# trả nonzero do Ign (danh sách cache/mirror khác vẫn đủ cài).
APT_OPTS="-o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20"

# Java (for pycocoevalcap CIDEr scorer)
apt-get $APT_OPTS update -y || true
apt-get $APT_OPTS install -y git-lfs openjdk-17-jdk || {
    echo "⚠️ [setup] apt install lỗi (mirror?) — thử lại 1 lần rồi bỏ qua (CIDEr/BLEU sẽ tắt, EM/F1 vẫn chạy)."
    apt-get $APT_OPTS update -y || true
    apt-get $APT_OPTS install -y git-lfs openjdk-17-jdk || true
}

export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"

# Python deps
pip uninstall -y transformers peft accelerate 2>/dev/null || true
pip install --quiet -r requirements.txt
pip install --quiet git+https://github.com/salaniz/pycocoevalcap

echo "✅ Setup complete"
