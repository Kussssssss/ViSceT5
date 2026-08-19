#!/bin/bash
# run_all_kaggle.sh — Kaggle Notebook launcher for ViSceT5 (OpenViVQA)
#
# Usage (in a Kaggle Notebook code cell with GPU P100 / T4 x2):
#   !bash run_all_kaggle.sh                 # full pretrain on branch exp/pretrain-gen-all
#   !bash run_all_kaggle.sh mock            # quick smoke test (sanity check)
#   !bash run_all_kaggle.sh finetune        # full finetune
#   !bash run_all_kaggle.sh finetune mock   # mock finetune
#
# To pass secrets & resume configs:
#   !STAGE=finetune HF_TOKEN=hf_xxx HF_REPO=username/viscet5-finetune bash run_all_kaggle.sh
#
set -e

# ---- resolve STAGE / MOCK_TEST from args (env wins if already set) ----
STAGE="${STAGE:-pretrain}"
MOCK_TEST="${MOCK_TEST:-false}"
for a in "$@"; do
  case "$(echo "$a" | tr '[:upper:]' '[:lower:]')" in
    pretrain|finetune|predict)      STAGE="$a" ;;
    mock|mocktest|smoke|smoke_test) MOCK_TEST="true" ;;
    full)                            MOCK_TEST="false" ;;
    *) echo "⚠️  Unknown arg '$a' (ignored). Use: [pretrain|finetune|predict] [mock]";;
  esac
done
export STAGE MOCK_TEST
export VISCET5_PROGRESS_ONLY=1
echo "▶ [Kaggle] STAGE=$STAGE | MOCK_TEST=$MOCK_TEST"

# ---- Branch & Directory Setup ----
GIT_BRANCH="${GIT_BRANCH:-exp/pretrain-gen-all}"
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${REPO_DIR:-}" ]; then
    TARGET_DIR="$REPO_DIR"
elif [ -f "$_SELF_DIR/run_pipeline.py" ]; then
    TARGET_DIR="$_SELF_DIR"
else
    TARGET_DIR="/kaggle/working/openvivqa"
fi

echo "▶ [Kaggle] Target Branch: $GIT_BRANCH | Target Dir: $TARGET_DIR"

if [ -d "$TARGET_DIR/.git" ]; then
    cd "$TARGET_DIR"
    git fetch origin --quiet
    git checkout "$GIT_BRANCH"
    if [ "${NO_PULL:-0}" != "1" ]; then
        git pull origin "$GIT_BRANCH"
    fi
else
    mkdir -p "$(dirname "$TARGET_DIR")"
    git clone https://github.com/Kussssssss/ViSceT5.git "$TARGET_DIR"
    cd "$TARGET_DIR"
    git checkout "$GIT_BRANCH"
fi

echo "▶ [Kaggle] Current Branch: $(git branch --show-current) ($(git rev-parse --short HEAD))"

# ---- Environment Setup ----
chmod +x setup.sh
./setup.sh

# ---- Launch Pipeline with Live Logging ----
python3 run_pipeline.py 2>&1 | tee train_execution.log
echo "✅ Finished (STAGE=$STAGE, MOCK_TEST=$MOCK_TEST). Log saved to $TARGET_DIR/train_execution.log"
