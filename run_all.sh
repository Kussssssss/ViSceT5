#!/bin/bash
# run_all.sh — Vast.ai / generic Linux launcher for ViSceT5
#
# Usage:
#   bash run_all.sh                 # full pretrain (default)
#   bash run_all.sh mock            # quick MOCK/SMOKE pretrain (fast sanity test)
#   bash run_all.sh finetune        # full finetune
#   bash run_all.sh finetune mock   # mock finetune
#   PREDICT_HF_REPO=<repo> bash run_all.sh predict   # sinh submission_{dev,test}.csv
#
# Args are order-independent. Environment overrides are also respected and take
# precedence if already exported: STAGE, MOCK_TEST, HF_TOKEN, HF_REPO,
# VAST_CONTAINERLABEL, CONTAINER_API_KEY (see run_pipeline.py).
set -e

# ---- resolve STAGE / MOCK_TEST from args (env wins if already set) ----
STAGE="${STAGE:-pretrain}"
MOCK_TEST="${MOCK_TEST:-false}"
for a in "$@"; do
  case "$(echo "$a" | tr '[:upper:]' '[:lower:]')" in
    pretrain|finetune|predict)    STAGE="$a" ;;
    mock|mocktest|smoke|smoke_test) MOCK_TEST="true" ;;
    full)                          MOCK_TEST="false" ;;
    *) echo "⚠️  Unknown arg '$a' (ignored). Use: [pretrain|finetune|predict] [mock]";;
  esac
done
export STAGE MOCK_TEST
echo "▶ [Vast.ai] STAGE=$STAGE | MOCK_TEST=$MOCK_TEST"

# ---- environment bootstrap ----
# fail-fast apt (xem setup.sh): tránh treo ở "Waiting for headers" khi host Vast
# không ra được archive.ubuntu.com. python3/git thường có sẵn trên image pytorch.
APT_OPTS="-o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20"
apt-get $APT_OPTS update || true
apt-get $APT_OPTS install -y python3-venv git || true
cd /workspace

if [ -d "/workspace/ViSceT5" ]; then
    cd /workspace/ViSceT5
else
    git clone https://github.com/Kussssssss/ViSceT5.git
    cd /workspace/ViSceT5
fi
# Các thay đổi (grounded-cloze, PRETRAIN_HF_REPO, ...) nằm ở nhánh exp/pretrain-gen-all,
# KHÔNG phải main. Bắt buộc checkout đúng nhánh.
BRANCH="${REPO_BRANCH:-exp/pretrain-gen-all}"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"
git log --oneline -1

if [ ! -d "/workspace/myenv" ]; then
    python3 -m venv --system-site-packages /workspace/myenv
fi

source /workspace/myenv/bin/activate

chmod +x setup.sh
./setup.sh

# ---- launch (background so the run survives SSH disconnects) ----
nohup python3 run_pipeline.py > train_execution.log 2>&1 &
echo "✅ Launched in background (STAGE=$STAGE, MOCK_TEST=$MOCK_TEST)."
echo "   Tail logs:  tail -f /workspace/ViSceT5/train_execution.log"
