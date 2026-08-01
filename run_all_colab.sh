#!/bin/bash
# run_all_colab.sh — Google Colab launcher for ViSceT5
#
# Usage (in a Colab cell, GPU runtime enabled):
#   !bash run_all_colab.sh                 # full pretrain (default)
#   !bash run_all_colab.sh mock            # quick MOCK/SMOKE pretrain (fast sanity test)
#   !bash run_all_colab.sh finetune        # full finetune
#   !bash run_all_colab.sh finetune mock   # mock finetune
#
# Args are order-independent. To set secrets, export before calling, e.g.:
#   !HF_TOKEN=xxx HF_REPO=user/repo bash run_all_colab.sh mock
#
# Differences vs run_all.sh (Vast.ai): runs under /content, uses Colab's system
# Python (NO venv — keeps the preinstalled CUDA build of torch intact), and runs
# in the FOREGROUND with `tee` so logs stream live into the cell and to a file.
set -e

# ---- resolve STAGE / MOCK_TEST from args (env wins if already set) ----
STAGE="${STAGE:-pretrain}"
MOCK_TEST="${MOCK_TEST:-false}"
for a in "$@"; do
  case "$(echo "$a" | tr '[:upper:]' '[:lower:]')" in
    pretrain|finetune)            STAGE="$a" ;;
    mock|mocktest|smoke|smoke_test) MOCK_TEST="true" ;;
    full)                          MOCK_TEST="false" ;;
    *) echo "⚠️  Unknown arg '$a' (ignored). Use: [pretrain|finetune] [mock]";;
  esac
done
export STAGE MOCK_TEST
# Colab: show only the trainer's tqdm progress bar, suppress per-step debug logs.
export VISCET5_PROGRESS_ONLY=1
echo "▶ [Colab] STAGE=$STAGE | MOCK_TEST=$MOCK_TEST | progress-bar only"

# ---- environment bootstrap ----
# GIT_BRANCH: nhánh cần chạy (mặc định 'main'). Ví dụ finetune bản đang thử nghiệm:
#   !GIT_BRANCH=exp/pretrain-gen-all STAGE=finetune ... bash run_all_colab.sh finetune
GIT_BRANCH="${GIT_BRANCH:-main}"
cd /content

if [ -d "/content/ViSceT5" ]; then
    cd /content/ViSceT5
    git fetch origin --quiet
    git checkout "$GIT_BRANCH"
    git pull origin "$GIT_BRANCH"
else
    git clone https://github.com/Kussssssss/ViSceT5.git
    cd /content/ViSceT5
    git checkout "$GIT_BRANCH"
fi
echo "▶ [Colab] GIT_BRANCH=$GIT_BRANCH ($(git rev-parse --short HEAD))"

# setup.sh installs Java (for CIDEr) + pip deps. torch is pinned with '>=' so
# Colab's preinstalled CUDA torch is kept (not downgraded/reinstalled).
chmod +x setup.sh
./setup.sh

# ---- launch (foreground + tee: logs both stream into the cell and are saved) ----
python3 run_pipeline.py 2>&1 | tee train_execution.log
echo "✅ Finished (STAGE=$STAGE, MOCK_TEST=$MOCK_TEST). Log saved to /content/ViSceT5/train_execution.log"
