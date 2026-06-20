#!/bin/bash
set -e

apt-get update && apt-get install -y python3-venv git
cd /workspace

if [ -d "/workspace/ViSceT5" ]; then
    cd /workspace/ViSceT5
    git pull
else
    git clone https://github.com/Kussssssss/ViSceT5.git
    cd /workspace/ViSceT5
fi

if [ ! -d "/workspace/myenv" ]; then
    python3 -m venv --system-site-packages /workspace/myenv
fi

source /workspace/myenv/bin/activate

chmod +x setup.sh
./setup.sh

nohup python3 run_pipeline.py > train_execution.log 2>&1 &
