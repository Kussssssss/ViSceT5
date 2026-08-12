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
#
# Ablation finetune (mặc định = tất cả true trong configs/finetune.yaml):
#   ABLATION_USE_QACLIP / ABLATION_USE_VS / ABLATION_USE_OCR / ABLATION_USE_OCR_AUG
# Khác:
#   REPO_DIR=<path>   chạy trên thư mục repo cụ thể (mặc định: thư mục chứa file này,
#                     hoặc /workspace/ViSceT5 nếu chạy từ nơi chưa có repo)
#   REPO_BRANCH=<br>  nhánh cần checkout (mặc định exp/pretrain-gen-all)
#   NO_PULL=1         KHÔNG git pull (giữ nguyên commit đang checkout — để test bản cũ)
#   SKIP_SANITY=1     bỏ qua bước kiểm tra khởi tạo model
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

# ---- CHỐT 1: bắt biến môi trường bị DÍNH ----
# Copy lệnh export hay dính non-breaking space → bash gộp 2 biến làm một, ví dụ
# NUM_TRAIN_EPOCHS="5<nbsp>CLAMP_VISION=1". Khi đó biến thứ hai KHÔNG hề được set và
# biến thứ nhất mang giá trị rác → crash khó hiểu. Không biến nào dưới đây được chứa '='.
for _v in STAGE MOCK_TEST NUM_TRAIN_EPOCHS DATASET_NAME HF_REPO \
          ABLATION_USE_QACLIP ABLATION_USE_VS ABLATION_USE_OCR ABLATION_USE_OCR_AUG \
          LOSS_ABLATION_MODE CLAMP_VISION DETERMINISTIC SMOKE_TRAIN_SAMPLES SMOKE_MAX_STEPS; do
  eval "_val=\${$_v-}"
  case "$_val" in
    *=*) echo "🛑 [env] Biến $_v='$_val' bị DÍNH (thường do non-breaking space khi copy)."
         echo "        Mỗi biến phải export trên MỘT dòng riêng, dùng dấu cách thường."
         exit 1 ;;
  esac
done

# ---- environment bootstrap ----
# fail-fast apt (xem setup.sh): tránh treo ở "Waiting for headers" khi host Vast
# không ra được archive.ubuntu.com. python3/git thường có sẵn trên image pytorch.
APT_OPTS="-o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20"
apt-get $APT_OPTS update || true
apt-get $APT_OPTS install -y python3-venv git || true

# ---- CHỐT 2: chạy đúng thư mục repo ----
# Trước đây đường dẫn bị hardcode /workspace/ViSceT5 → chạy từ thư mục khác (vd
# ViSceT5-ver2) vẫn nhảy về repo cũ và train nhầm code. Giờ ưu tiên: REPO_DIR > thư mục
# chứa chính file này > /workspace/ViSceT5 (clone nếu chưa có).
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${REPO_DIR:-}" ]; then
    :
elif [ -f "$_SELF_DIR/run_pipeline.py" ]; then
    REPO_DIR="$_SELF_DIR"
else
    REPO_DIR="/workspace/ViSceT5"
fi
if [ ! -d "$REPO_DIR/.git" ]; then
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone https://github.com/Kussssssss/ViSceT5.git "$REPO_DIR"
fi
cd "$REPO_DIR"

BRANCH="${REPO_BRANCH:-exp/pretrain-gen-all}"
git fetch origin
git checkout "$BRANCH"
if [ "${NO_PULL:-0}" = "1" ]; then
    echo "ℹ️  NO_PULL=1 → giữ nguyên commit đang checkout."
else
    git pull origin "$BRANCH"
fi
git log --oneline -1

if [ ! -d "/workspace/myenv" ]; then
    python3 -m venv --system-site-packages /workspace/myenv
fi
source /workspace/myenv/bin/activate

chmod +x setup.sh
./setup.sh

# ---- tóm tắt cấu hình (soi nhanh trước khi tốn GPU) ----
echo ""
echo "────────────── CẤU HÌNH ──────────────"
echo "  repo      : $REPO_DIR ($BRANCH)"
echo "  stage     : $STAGE | mock=$MOCK_TEST"
echo "  dataset   : ${DATASET_NAME:-<yaml default>} | epochs=${NUM_TRAIN_EPOCHS:-<yaml>}"
if [ "$STAGE" = "finetune" ]; then
echo "  ablation  : qaclip=${ABLATION_USE_QACLIP:-true} vs=${ABLATION_USE_VS:-true} ocr=${ABLATION_USE_OCR:-true} ocr_aug=${ABLATION_USE_OCR_AUG:-true}"
fi
echo "  hf_repo   : ${HF_REPO:-<không push>}"
echo "──────────────────────────────────────"

# ---- CHỐT 3: model phải khởi tạo SẠCH trước khi train ----
# QA-ViT adapter dùng nn.Linear(bias=False); HF _fast_init từng để chúng CHƯA khởi tạo
# (torch.empty = rác, có lúc NaN) → img_tokens NaN 100% → loss NaN ngay step 0. Lỗi này
# KHÔNG tất định nên phải chặn ngay tại đây thay vì phát hiện sau vài giờ train.
if [ "${SKIP_SANITY:-0}" != "1" ] && [ "$STAGE" != "predict" ]; then
  echo "🔎 Kiểm tra khởi tạo model..."
  python3 - <<'PY' || { echo "🛑 DỪNG: model có tham số non-finite ngay khi khởi tạo."; exit 1; }
import os, sys, warnings; warnings.filterwarnings("ignore")
import torch
from configs.model_config import OpenViVQAConfig
from models.openvivqa_model import OpenViVQAModel

def _flag(name, default=True):
    v = os.environ.get(name, "").strip().lower()
    if not v: return default
    return v in ("1", "true", "yes", "on")

torch.manual_seed(42)
c = OpenViVQAConfig(); c.pretrain = False
# Dựng ĐÚNG cấu hình sắp train (không phải mặc định): ablation_use_vs quyết định
# visual_search có tồn tại hay không, nên số tham số in ra dưới đây mới là số thật.
c.ablation_use_qaclip = _flag("ABLATION_USE_QACLIP")
c.ablation_use_vs     = _flag("ABLATION_USE_VS")
c.ablation_use_ocr    = _flag("ABLATION_USE_OCR")
c.ablation_use_ocr_aug= _flag("ABLATION_USE_OCR_AUG")
m = OpenViVQAModel(c)
bad = [n for n, p in m.named_parameters() if not torch.isfinite(p).all()]
print(f"   modules: qaclip={c.ablation_use_qaclip} vs={c.ablation_use_vs} "
      f"ocr={c.ablation_use_ocr} ocr_aug={c.ablation_use_ocr_aug}")
print(f"   visual_search: {'CÓ' if hasattr(m, 'visual_search') else 'ĐÃ GỠ'} | "
      f"tổng tham số = {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
print(f"   non-finite params = {len(bad)}")
if bad:
    print("   vd:", bad[:3]); sys.exit(1)
PY
fi

# ---- launch (background so the run survives SSH disconnects) ----
LOG="${LOG_FILE:-$REPO_DIR/train_execution.log}"
nohup python3 run_pipeline.py > "$LOG" 2>&1 &
echo "✅ Launched in background (STAGE=$STAGE, MOCK_TEST=$MOCK_TEST)."
echo "   Tail logs:  tail -f $LOG"
