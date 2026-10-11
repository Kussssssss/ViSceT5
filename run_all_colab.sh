#!/bin/bash
# Direct VQA training. Usage: bash run_all_colab.sh [mock|full]
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
cd /content
if [ -d /content/ViSceT5 ]; then
    cd /content/ViSceT5
    git pull
else
    git clone https://github.com/Kussssssss/ViSceT5.git
    cd /content/ViSceT5
fi
bash setup.sh
python3 run_pipeline.py 2>&1 | tee train_execution.log
