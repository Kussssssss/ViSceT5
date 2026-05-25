#!/bin/bash
# setup.sh — Environment setup for OpenViVQA on Kaggle/Colab/Ubuntu

set -e

# Java (for pycocoevalcap CIDEr scorer)
apt-get install -y git-lfs openjdk-17-jdk

export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"

# Python deps
pip uninstall -y transformers peft accelerate 2>/dev/null || true
pip install --quiet -r requirements.txt
pip install --quiet git+https://github.com/salaniz/pycocoevalcap

echo "✅ Setup complete"
