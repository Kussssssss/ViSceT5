#!/usr/bin/env python
"""
scripts/diag_vision_grad.py  —  CHẨN ĐOÁN (read-only) nguồn grad_norm=inf.

Chạy trên đúng GPU tái hiện lỗi (A100). KHÔNG train, KHÔNG sửa gì. Nó dựng model
FULL (scratch, đúng config finetune), lấy 1 batch, forward+backward MỘT lần, rồi in:
  - độ lớn |img_hs|, |patch_scores| (output QA-CLIP vision) — có chạm 1e4 (=đã tràn
    rồi bị clamp) hay bình thường (~vài chục)?
  - grad_norm tổng (tính float64 để KHÔNG bị tràn) — có thật sự lớn không?
  - top module theo grad-norm — module nào đóng góp grad khổng lồ?
  - param nào có grad non-finite (nếu có).

Dùng:
    cd /workspace/ViSceT5
    python scripts/diag_vision_grad.py            # mặc định FULL, ViTextVQA
    python scripts/diag_vision_grad.py only_qaclip  # so sánh cấu hình chạy được
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import torch, pandas as pd
from collections import defaultdict

from configs.model_config import OpenViVQAConfig
from configs.ocr_config import DEFAULT_OCR_CONFIG
from configs.base_config import OUTPUT_PATH
from models.openvivqa_model import OpenViVQAModel
from models.modules.ocr_encoder_feature import Vision_Encode_Ocr_Feature
from data.collator import ViT5VQADataCollator
from data.dataset import ViT5VQADataset
from transformers import AutoTokenizer

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# A/B TF32: A100/Ampere bật TF32 mặc định (matmul mantissa ~10-bit). DIAG_NO_TF32=1 để
# ép fp32 THẬT → nếu phân kỳ biến mất khi tắt TF32 thì TF32 chính là nguyên nhân.
if os.environ.get("DIAG_NO_TF32", "0").lower() in ("1", "true", "yes", "on"):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("DIAG_NO_TF32=1 → TF32 OFF (fp32 thật)")
else:
    print(f"TF32 matmul allowed = {torch.backends.cuda.matmul.allow_tf32} (mặc định GPU)")

# batch từ CSV cache (giống lúc train). Thử vài vị trí file.
_csv = None
for p in [os.path.join(OUTPUT_PATH, "merged_train.csv"),
          os.path.join(OUTPUT_PATH, "merged_train_ViTextVQA.csv"),
          "output/merged_train.csv"]:
    if os.path.exists(p):
        _csv = p; break
assert _csv, "Không tìm thấy merged_train CSV — chạy sau khi đã prepare dataset."
_bs = int(os.environ.get("DIAG_BATCH", "4"))
df = pd.read_csv(_csv).head(_bs).reset_index(drop=True)
print(f"device={dev} | mode={MODE} | csv={_csv} | batch={len(df)}")

torch.manual_seed(42)
c = OpenViVQAConfig(); c.pretrain = False
full = (MODE == "full")
c.ablation_use_qaclip = True
c.ablation_use_vs     = full
c.ablation_use_ocr    = full
c.ablation_use_ocr_aug = full
m = OpenViVQAModel(c); m.pretrain = False; m.config.use_twc = False
m = m.to(dev); m.train()
tok = AutoTokenizer.from_pretrained("VietAI/vit5-base")
voc = Vision_Encode_Ocr_Feature(DEFAULT_OCR_CONFIG)
col = ViT5VQADataCollator(tokenizer=tok, image_processor=m.image_processor, ocr_encoder=voc,
    config=m.config, term_vocab_path="configs/data/term_vocab.txt",
    viet_vocab_path="configs/data/viet_vocab.txt", eng_vocab_path="", dataframe=df, pretrain=False)
if hasattr(col, "set_mode"): col.set_mode(pretrain=False, mask_prob=0.0)
col.use_ocr_aug_finetune = full
def to_dev(bb):
    return {k: (v.to(dev) if torch.is_tensor(v)
                else ([{kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in d.items()} for d in v]
                      if k == "ocr_info" else v)) for k, v in bb.items()}

ds = ViT5VQADataset(df)
b = to_dev(col([ds[i] for i in range(len(df))]))

# hook vision output
cap = {}
_orig = m._encode_image
def wrap(*a, **k):
    out = _orig(*a, **k)
    try:
        it = out["img_tokens"].detach().float()
        ps = out.get("patch_scores")
        cap["img_tokens_absmax"] = float(it.abs().max())
        if ps is not None:
            cap["patch_scores_absmax"] = float(ps.detach().float().abs().max())
    except Exception:
        pass
    return out
m._encode_image = wrap

def grad_report(mm):
    """Trả (grad_norm float64, top-module dict, max|g| (name,val), có non-finite?)."""
    mod_sq = defaultdict(float); total_sq = 0.0; nonfinite = []
    gmax = 0.0; gmax_name = ""
    for n, p in mm.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            nonfinite.append(n); continue
        s = float((g.double() ** 2).sum()); total_sq += s
        top = n.split(".")[0] + ("." + n.split(".")[1] if n.startswith(("vit5", "qa_clip")) and len(n.split(".")) > 1 else "")
        mod_sq[top] += s
        gm = float(g.abs().max())
        if gm > gmax: gmax, gmax_name = gm, n
    return math.sqrt(total_sq), mod_sq, (gmax_name, gmax), nonfinite

# ── QUÉT NHIỀU BATCH: tìm batch làm img_tokens/grad_norm vọt lên (data-dependent?) ──
NB = int(os.environ.get("DIAG_NBATCH", "20"))
dfull = pd.read_csv(_csv).head(NB * _bs).reset_index(drop=True)
dsN = ViT5VQADataset(dfull)
print(f"\n== SCAN {NB} batch (bs={_bs}) ==")
print(f"{'batch':>5} {'loss':>9} {'|img_tok|':>10} {'|patch|':>9} {'grad_norm':>12}  nonfinite")
worst = (-1.0, None)
for bi in range(NB):
    items = [dsN[bi * _bs + j] for j in range(_bs)]
    bb = col(items)
    bb = {k: (v.to(dev) if torch.is_tensor(v)
              else ([{kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in d.items()} for d in v]
                    if k == "ocr_info" else v)) for k, v in bb.items()}
    m.zero_grad(set_to_none=True); cap.clear()
    o = m(**bb); l = o["loss"]; l.backward()
    gn, msq, (gname, gval), nf = grad_report(m)
    print(f"{bi:>5} {float(l):>9.3f} {cap.get('img_tokens_absmax', float('nan')):>10.4g} "
          f"{cap.get('patch_scores_absmax', float('nan')):>9.4g} {gn:>12.4g}  {len(nf)}")
    if gn > worst[0] or nf:
        worst = (gn if not nf else float('inf'), (bi, msq, (gname, gval), nf))
print("\n== BATCH TỆ NHẤT (grad_norm lớn nhất / có non-finite) ==")
bi, msq, (gname, gval), nf = worst[1]
print(f"batch {bi}: grad_norm={worst[0]:.4g} | max|g|={gval:.4g} @ {gname} | nonfinite={len(nf)}")
if nf: print("  nonfinite params:", nf[:8])
print("  top modules:")
for k, v in sorted(msq.items(), key=lambda x: -x[1])[:10]:
    print(f"    {math.sqrt(v):>12.4g}  {k}")

# ── MINI-TRAIN: tái hiện phân kỳ (AdamW + clip 1.0 y hệt trainer) ──
STEPS = int(os.environ.get("DIAG_STEPS", "0"))
if STEPS > 0:
    lr = float(os.environ.get("DIAG_LR", "3e-5"))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    print(f"\n== MINI-TRAIN {STEPS} steps (lr={lr}, clip=1.0) — xem grad_norm trôi tới inf? ==")
    print(f"{'step':>4} {'loss':>9} {'|img_tok|':>10} {'gnorm_f64':>12} {'vs_norm':>11} {'enc_norm':>10} {'gnorm_f32(clip)':>16}")
    for si in range(STEPS):
        items = [dsN[(si * _bs + j) % len(dsN)] for j in range(_bs)]
        bb = to_dev(col(items))
        opt.zero_grad(set_to_none=True); cap.clear()
        o = m(**bb); l = o["loss"]; l.backward()
        gn, msq2, _, nf = grad_report(m)
        vs = math.sqrt(msq2.get("visual_search", 0.0))
        enc = math.sqrt(msq2.get("vit5.encoder", 0.0))
        gn32 = torch.nn.utils.clip_grad_norm_(
            [p for p in m.parameters() if p.requires_grad and p.grad is not None], 1.0)
        opt.step()
        print(f"{si:>4} {float(l):>9.3f} {cap.get('img_tokens_absmax', float('nan')):>10.4g} "
              f"{gn:>12.4g} {vs:>11.4g} {enc:>10.4g} {float(gn32):>16.4g}")
        if not math.isfinite(float(gn32)):
            print(f"  → grad_norm(float32) = inf tại step {si} → clip triệt tiêu update từ đây (kẹt).")
