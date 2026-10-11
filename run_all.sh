#!/bin/bash
# Direct VQA training. Usage: bash run_all.sh [mock|full]
set -euo pipefail
MOCK_TEST="${MOCK_TEST:-false}"
for argument in "$@"; do
    case "$argument" in
        mock|smoke) MOCK_TEST="true" ;;
        full) MOCK_TEST="false" ;;
        finetune) ;; # Accepted for existing VQA launch commands.
        *) echo "Unknown argument: $argument. Use mock or full." >&2; exit 2 ;;
    esac
done
export MOCK_TEST
apt-get update && apt-get install -y python3-venv git
cd /workspace
if [ -d /workspace/ViSceT5 ]; then
    cd /workspace/ViSceT5
    git pull
else
    git clone https://github.com/Kussssssss/ViSceT5.git
    cd /workspace/ViSceT5
fi
if [ ! -d /workspace/myenv ]; then
    python3 -m venv --system-site-packages /workspace/myenv
fi
source /workspace/myenv/bin/activate
bash setup.sh
nohup python3 run_pipeline.py > train_execution.log 2>&1 &
echo "Launched VQA training. Tail train_execution.log for progress."
