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
ds = ViT5VQADataset(df)
b = col([ds[i] for i in range(len(df))])
b = {k: (v.to(dev) if torch.is_tensor(v)
         else ([{kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in d.items()} for d in v]
               if k == "ocr_info" else v)) for k, v in b.items()}

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

out = m(**b); loss = out["loss"]
print(f"loss = {float(loss):.4f} (finite={bool(torch.isfinite(loss).all())})")
loss.backward()

print(f"|img_tokens| max = {cap.get('img_tokens_absmax')}")
print(f"|patch_scores| max = {cap.get('patch_scores_absmax')}")

# grad norm tính bằng float64 (không tràn) + gom theo module top-level
mod_sq = defaultdict(float); total_sq = 0.0; nonfinite = []
gmax = 0.0; gmax_name = ""
for n, p in m.named_parameters():
    if not p.requires_grad or p.grad is None:
        continue
    g = p.grad.detach()
    if not torch.isfinite(g).all():
        nonfinite.append(n)
        continue
    s = float((g.double() ** 2).sum())
    total_sq += s
    top = n.split(".")[0] + ("." + n.split(".")[1] if n.startswith(("vit5", "qa_clip")) and len(n.split(".")) > 1 else "")
    mod_sq[top] += s
    gm = float(g.abs().max())
    if gm > gmax: gmax, gmax_name = gm, n

print(f"\nTOTAL grad_norm (float64) = {math.sqrt(total_sq):.4g}")
print(f"max |grad| element = {gmax:.4g}  @ {gmax_name}")
if nonfinite:
    print(f"NON-FINITE grad params ({len(nonfinite)}): {nonfinite[:8]}")
print("\nTop modules by grad-norm:")
for k, v in sorted(mod_sq.items(), key=lambda x: -x[1])[:12]:
    print(f"  {math.sqrt(v):>12.4g}  {k}")
