#!/bin/bash
# Colab launcher. Usage: bash run_all_colab.sh [finetune|predict] [mock|full]
# Configured through environment variables (see run_pipeline.py); GIT_BRANCH picks the branch.
set -euo pipefail
MOCK_TEST="${MOCK_TEST:-false}"
STAGE="${STAGE:-finetune}"
for argument in "$@"; do
    case "$argument" in
        mock|smoke) MOCK_TEST="true" ;;
        full) MOCK_TEST="false" ;;
        finetune|predict) STAGE="$argument" ;;
        *) echo "Unknown argument: $argument. Use finetune|predict and mock|full." >&2; exit 2 ;;
    esac
done
export MOCK_TEST STAGE
# stdout goes through tee (a pipe): without this Python buffers prints in 8KB blocks.
export PYTHONUNBUFFERED=1
GIT_BRANCH="${GIT_BRANCH:-exp/twc-ocr-centric}"
cd /content
if [ -d /content/ViSceT5 ]; then
    cd /content/ViSceT5
    git fetch -q origin
    git checkout -q "$GIT_BRANCH"
    git pull -q origin "$GIT_BRANCH"
else
    git clone -q -b "$GIT_BRANCH" https://github.com/Kussssssss/ViSceT5.git
    cd /content/ViSceT5
fi
echo "▶ branch $GIT_BRANCH ($(git rev-parse --short HEAD)) | STAGE=$STAGE | MOCK_TEST=$MOCK_TEST"
bash setup.sh
python3 run_pipeline.py 2>&1 | tee train_execution.log
