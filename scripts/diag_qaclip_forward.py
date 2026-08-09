#!/usr/bin/env python
"""
scripts/diag_qaclip_forward.py — READ-ONLY: tìm module ĐẦU TIÊN trong QA-CLIP có output
non-finite, trên đúng batch đã lỗi. KHÔNG sửa model, KHÔNG train.

Tái hiện y hệt verify_all: seed 42 -> build only_qaclip -> batch đầu của thứ tự shuffle 42.
Gắn forward-hook lên MỌI submodule của qa_clip (+ đo biên độ txt_hidden_states đưa vào),
in theo THỨ TỰ THỰC THI: module nào lần đầu cho NaN/inf, và biên độ ngay trước đó.

Dùng:
    python scripts/diag_qaclip_forward.py 2>&1 | tee /workspace/diag_qaclip.log
Env: CONFIG=only_qaclip|full   BS=4   NPOOL=4096   BATCH=0
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import torch, pandas as pd

from configs.model_config import OpenViVQAConfig
from configs.ocr_config import DEFAULT_OCR_CONFIG
from configs.base_config import OUTPUT_PATH
from models.openvivqa_model import OpenViVQAModel
from models.modules.ocr_encoder_feature import Vision_Encode_Ocr_Feature
from data.collator import ViT5VQADataCollator
from data.dataset import ViT5VQADataset
from transformers import AutoTokenizer

CONFIG = os.environ.get("CONFIG", "only_qaclip")
BS     = int(os.environ.get("BS", "4"))
NPOOL  = int(os.environ.get("NPOOL", "4096"))
BATCH  = int(os.environ.get("BATCH", "0"))
DEV    = "cuda" if torch.cuda.is_available() else "cpu"
FULL   = (CONFIG == "full")

_csv = None
for p in [os.path.join(OUTPUT_PATH, "merged_train.csv"),
          os.path.join(OUTPUT_PATH, "merged_train_ViTextVQA.csv"), "output/merged_train.csv"]:
    if os.path.exists(p): _csv = p; break
assert _csv, "Không thấy merged_train CSV."
df = pd.read_csv(_csv).head(NPOOL).reset_index(drop=True)

torch.manual_seed(42)                      # GIỐNG verify_all.build()
c = OpenViVQAConfig(); c.pretrain = False
c.ablation_use_qaclip = True; c.ablation_use_vs = FULL
c.ablation_use_ocr = FULL;    c.ablation_use_ocr_aug = FULL
m = OpenViVQAModel(c); m.pretrain = False; m.config.use_twc = False
m = m.to(DEV); m.train()
tok = AutoTokenizer.from_pretrained("VietAI/vit5-base")
voc = Vision_Encode_Ocr_Feature(DEFAULT_OCR_CONFIG)
col = ViT5VQADataCollator(tokenizer=tok, image_processor=m.image_processor, ocr_encoder=voc,
    config=m.config, term_vocab_path="configs/data/term_vocab.txt",
    viet_vocab_path="configs/data/viet_vocab.txt", eng_vocab_path="", dataframe=df, pretrain=False)
col.set_mode(pretrain=False, mask_prob=0.0); col.use_ocr_aug_finetune = FULL
ds = ViT5VQADataset(df)
g = torch.Generator(); g.manual_seed(42)
ORDER = torch.randperm(len(ds), generator=g).tolist()
idx = ORDER[BATCH * BS:(BATCH + 1) * BS]
print(f"config={CONFIG} | device={DEV} | batch={BATCH} rows={idx}")

b = col([ds[i] for i in idx])
b = {k: (v.to(DEV) if torch.is_tensor(v)
         else ([{kk: (vv.to(DEV) if torch.is_tensor(vv) else vv) for kk, vv in d.items()} for d in v]
               if k == "ocr_info" else v)) for k, v in b.items()}

# ── hook: ghi lại biên độ + phát hiện non-finite theo THỨ TỰ THỰC THI ──
log, first_bad, step = [], [None], [0]

def stat(t):
    if not torch.is_tensor(t) or not t.is_floating_point(): return None
    f = t.detach().float()
    fin = torch.isfinite(f)
    return dict(shape=tuple(f.shape), nan=int(torch.isnan(f).sum()), inf=int(torch.isinf(f).sum()),
                absmax=(float(f[fin].abs().max()) if fin.any() else float("nan")))

def mk(name):
    def hook(mod, inp, out):
        step[0] += 1
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for oi, o in enumerate(outs):
            s = stat(o)
            if s is None: continue
            log.append((step[0], name, oi, s))
            if (s["nan"] or s["inf"]) and first_bad[0] is None:
                ins = [stat(x) for x in (inp if isinstance(inp, tuple) else (inp,))]
                first_bad[0] = (step[0], name, oi, s, [x for x in ins if x])
    return hook

hs = [m.qa_clip.register_forward_hook(mk("qa_clip"))]
for n, mod in m.qa_clip.named_modules():
    if n: hs.append(mod.register_forward_hook(mk("qa_clip." + n)))

with torch.no_grad():
    out = m(**b)
for h in hs: h.remove()

loss = out.get("loss")
print(f"loss = {loss} | finite = {bool(torch.isfinite(loss).all()) if loss is not None else None}")

print("\n=== MODULE ĐẦU TIÊN CÓ OUTPUT NON-FINITE ===")
if first_bad[0] is None:
    print("Không module nào trong qa_clip cho non-finite ở forward này.")
else:
    st, name, oi, s, ins = first_bad[0]
    print(f"#{st}  {name}  (output[{oi}])  nan={s['nan']} inf={s['inf']} absmax={s['absmax']:.4g} shape={s['shape']}")
    print("  INPUT của nó:")
    for k, x in enumerate(ins):
        print(f"    in[{k}]: nan={x['nan']} inf={x['inf']} absmax={x['absmax']:.4g} shape={x['shape']}")
    print("\n  5 module NGAY TRƯỚC đó (xem biên độ leo thang):")
    for e in [x for x in log if x[0] < st][-5:]:
        print(f"    #{e[0]}  {e[1]}[{e[2]}]  absmax={e[3]['absmax']:.4g}  nan={e[3]['nan']} inf={e[3]['inf']}")

print("\n=== TOP 15 module theo biên độ (|absmax| lớn nhất) ===")
for e in sorted([x for x in log if x[3]["absmax"] == x[3]["absmax"]], key=lambda x: -x[3]["absmax"])[:15]:
    print(f"  {e[3]['absmax']:>14.4g}   #{e[0]:<5} {e[1]}[{e[2]}]")
