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
echo "▶ [Colab] STAGE=$STAGE | MOCK_TEST=$MOCK_TEST"

# ---- environment bootstrap ----
cd /content

if [ -d "/content/ViSceT5" ]; then
    cd /content/ViSceT5
    git pull
else
    git clone https://github.com/Kussssssss/ViSceT5.git
    cd /content/ViSceT5
fi

# setup.sh installs Java (for CIDEr) + pip deps. torch is pinned with '>=' so
# Colab's preinstalled CUDA torch is kept (not downgraded/reinstalled).
chmod +x setup.sh
./setup.sh

# ---- launch (foreground + tee: logs both stream into the cell and are saved) ----
python3 run_pipeline.py 2>&1 | tee train_execution.log
echo "✅ Finished (STAGE=$STAGE, MOCK_TEST=$MOCK_TEST). Log saved to /content/ViSceT5/train_execution.log"
