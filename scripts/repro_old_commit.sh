#!/bin/bash
# repro_old_commit.sh — chạy lại pipeline của một commit CŨ (mặc định 7b99d60, commit đã
# sinh run Vast `ViSceT5-ft-vitextvqa-only_consformer`) trên Colab, KHÔNG sửa code commit đó.
#
#   bash scripts/repro_old_commit.sh predict          # STAGE=predict (cần PREDICT_CKPT_DIR)
#   bash scripts/repro_old_commit.sh finetune         # finetune đầy đủ (HF_REPO để push/resume)
#   bash scripts/repro_old_commit.sh finetune mock    # smoke test vài bước
#
# Env: REPRO_COMMIT (7b99d60), REPRO_DIR (/content/ViSceT5_old), REPRO_LOG (file log, ghi nối),
# cùng mọi biến run_pipeline.py của commit đó đọc (ABLATION_USE_*, PREDICT_*, HF_TOKEN, HF_REPO).
#
# Commit cũ được clone ở thư mục RIÊNG (detached HEAD) để không đụng repo đang dùng.
# Khác biệt duy nhất so với lúc chạy trên Vast: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 — PyTorch ≥2.6
# đổi mặc định torch.load(weights_only=True) làm resume của transformers 4.45 nổ ở
# rng_state.pth; biến này khôi phục hành vi cũ, không ảnh hưởng tính toán.
set -e
set -o pipefail

STAGE="${STAGE:-finetune}"
MOCK_TEST="${MOCK_TEST:-false}"
for a in "$@"; do
  case "$(echo "$a" | tr '[:upper:]' '[:lower:]')" in
    finetune|predict) STAGE="$a" ;;
    mock|smoke)       MOCK_TEST="true" ;;
    *) echo "⚠️  Bỏ qua tham số '$a' (dùng: finetune|predict [mock])" ;;
  esac
done

COMMIT="${REPRO_COMMIT:-7b99d60}"
DIR="${REPRO_DIR:-/content/ViSceT5_old}"
LOG="${REPRO_LOG:-$DIR/repro_${STAGE}.log}"

if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/Kussssssss/ViSceT5.git "$DIR"
fi
cd "$DIR"
git fetch origin --quiet
git checkout --quiet --detach "$COMMIT"
echo "▶ [repro] commit $(git rev-parse --short HEAD) | STAGE=$STAGE | MOCK_TEST=$MOCK_TEST | log=$LOG"

chmod +x setup.sh
./setup.sh

export STAGE MOCK_TEST
export VISCET5_PROGRESS_ONLY=1 PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$(dirname "$LOG")"
python3 run_pipeline.py 2>&1 | tee -a "$LOG"
echo "✅ [repro] xong STAGE=$STAGE (commit $COMMIT)"
