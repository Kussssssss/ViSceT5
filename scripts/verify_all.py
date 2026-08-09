#!/usr/bin/env python
"""
scripts/verify_all.py — XÁC MINH TRỌN GÓI (read-only, không train, không sửa gì).

Chạy MỘT lần trên máy đang lỗi, trả lời hết:
  P1. Môi trường: GPU, torch, TF32, commit, biến env tồn đọng.
  P2. Cấu hình nào sinh grad non-finite? (quét batch theo thứ tự SHUFFLE giống trainer)
  P3. det/rec có phải nguyên nhân? -> A/B ngay trong CÙNG model bằng cách zero-hoá
      ocr_lite_det_proj/rec_proj (khi đó det_emb=rec_emb=0 => công thức y hệt bản
      TRƯỚC khi thêm det/rec). Cùng seed, cùng batch, cùng trọng số => A/B tuyệt đối sạch.
  P4. NaN hay inf? param nào? module nào?
  P5. Op nào sinh NaN trong backward -> torch.autograd.detect_anomaly (chỉ chạy trên
      đúng batch đã lỗi, nên nhanh).
  P6. Bảng tổng kết + KẾT LUẬN tự động.

Dùng:
    cd /workspace/<repo> && source /workspace/myenv/bin/activate
    python scripts/verify_all.py 2>&1 | tee /workspace/verify_all.log

Env tuỳ chọn: BS=4  NPOOL=4096  NB=40  CONFIGS=full,only_qaclip,...
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, math, traceback
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

BS      = int(os.environ.get("BS", "4"))
NPOOL   = int(os.environ.get("NPOOL", "4096"))
NB      = int(os.environ.get("NB", "40"))
DEV     = "cuda" if torch.cuda.is_available() else "cpu"
SEP     = "=" * 78

# (tên, use_qaclip, use_vs, use_ocr, use_ocr_aug)
ALL_CFG = [
    ("full",         True,  True,  True,  True),
    ("only_qaclip",  True,  False, False, False),
    ("only_vs",      False, True,  False, False),
    ("only_ocr",     False, False, True,  False),
    ("only_ocr_aug", False, False, False, True),
    ("baseline_off", False, False, False, False),
]
_want = os.environ.get("CONFIGS", "").strip()
if _want:
    _keep = {s.strip() for s in _want.split(",") if s.strip()}
    ALL_CFG = [x for x in ALL_CFG if x[0] in _keep]

# ─────────────────────────── P1. MÔI TRƯỜNG ───────────────────────────
print(SEP); print("P1. MÔI TRƯỜNG"); print(SEP)
print(f"torch        : {torch.__version__}")
print(f"device       : {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV == "cuda" else ""))
if DEV == "cuda":
    print(f"TF32 matmul  : {torch.backends.cuda.matmul.allow_tf32} | cudnn: {torch.backends.cudnn.allow_tf32}")
try:
    import subprocess
    _c = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
    # -uno: BỎ file chưa track (chỉ quan tâm file ĐÃ track bị sửa cục bộ — thứ có thể
    # làm code chạy khác với commit đang ghi ở trên).
    _d = subprocess.run(["git", "status", "--porcelain", "-uno"], capture_output=True, text=True).stdout.strip()
    print(f"git HEAD     : {_c}")
    print(f"git dirty    : {'CÓ (file ĐÃ TRACK bị sửa cục bộ!)' if _d else 'không'}")
    if _d: print("   " + _d.replace("\n", "\n   "))
except Exception as e:
    print(f"git          : (không đọc được: {e})")
_envs = ["ABLATION_USE_QACLIP", "ABLATION_USE_VS", "ABLATION_USE_OCR", "ABLATION_USE_OCR_AUG",
         "LOSS_ABLATION_MODE", "CLAMP_VISION", "OCR_LITE_DETREC", "DETERMINISTIC", "ANOMALY",
         "MODEL_NAME_OR_PATH", "PRETRAIN_HF_REPO", "PRETRAIN_HF_CKPT", "RESUME_FROM_CHECKPOINT",
         "RESUME_CHECKPOINT_ID", "RESUME_FROM_HF", "HF_REPO", "SMOKE_TRAIN_SAMPLES", "SMOKE_MAX_STEPS"]
_set = {k: os.environ[k] for k in _envs if os.environ.get(k, "").strip()}
print(f"env tồn đọng : {_set if _set else '(không có)'}")

# dữ liệu
_csv = None
for p in [os.path.join(OUTPUT_PATH, "merged_train.csv"),
          os.path.join(OUTPUT_PATH, "merged_train_ViTextVQA.csv"), "output/merged_train.csv"]:
    if os.path.exists(p): _csv = p; break
assert _csv, "Không thấy merged_train CSV — chạy prepare dataset trước."
df = pd.read_csv(_csv).head(NPOOL).reset_index(drop=True)
print(f"csv          : {_csv} | pool={len(df)} | BS={BS} | quét tối đa {NB} batch")

tok = AutoTokenizer.from_pretrained("VietAI/vit5-base")
voc = Vision_Encode_Ocr_Feature(DEFAULT_OCR_CONFIG)
ds  = ViT5VQADataset(df)
g = torch.Generator(); g.manual_seed(42)              # thứ tự GIỐNG trainer
ORDER = torch.randperm(len(ds), generator=g).tolist()


def build(cfg, detrec=True):
    _, q, v, o, a = cfg
    torch.manual_seed(42)                              # cùng seed => cùng trọng số
    c = OpenViVQAConfig(); c.pretrain = False
    c.ablation_use_qaclip = q; c.ablation_use_vs = v
    c.ablation_use_ocr = o;    c.ablation_use_ocr_aug = a
    m = OpenViVQAModel(c); m.pretrain = False; m.config.use_twc = False
    if not detrec and hasattr(m, "ocr_lite_det_proj"):
        with torch.no_grad():                          # det_emb = rec_emb = 0  <=> bản CŨ
            m.ocr_lite_det_proj.weight.zero_(); m.ocr_lite_det_proj.bias.zero_()
            m.ocr_lite_rec_proj.weight.zero_(); m.ocr_lite_rec_proj.bias.zero_()
    m = m.to(DEV); m.train()
    col = ViT5VQADataCollator(tokenizer=tok, image_processor=m.image_processor, ocr_encoder=voc,
        config=m.config, term_vocab_path="configs/data/term_vocab.txt",
        viet_vocab_path="configs/data/viet_vocab.txt", eng_vocab_path="", dataframe=df, pretrain=False)
    col.set_mode(pretrain=False, mask_prob=0.0); col.use_ocr_aug_finetune = a
    return m, col


def to_dev(b):
    return {k: (v.to(DEV) if torch.is_tensor(v)
                else ([{kk: (vv.to(DEV) if torch.is_tensor(vv) else vv) for kk, vv in d.items()} for d in v]
                      if k == "ocr_info" else v)) for k, v in b.items()}


def bad_grads(m):
    out, tn, ti = [], 0, 0
    for n, p in m.named_parameters():
        if p.grad is None or torch.isfinite(p.grad).all():
            continue
        cn = int(torch.isnan(p.grad).sum()); ci = int(torch.isinf(p.grad).sum())
        tn += cn; ti += ci; out.append((n, cn, ci))
    return out, tn, ti


HOOKS = os.environ.get("HOOKS", "0") == "1"   # HOOKS=1: bắt module đầu tiên non-finite


def scan(m, col, nb):
    """Quét tới khi gặp batch xấu. Trả (batch_idx, rows, info) hoặc None."""
    # Gắn hook TRONG CHÍNH chuỗi này (chạy riêng lẻ KHÔNG tái hiện được vì NaN ở
    # ngưỡng biên, phụ thuộc trạng thái RNG tích luỹ) → mới bắt đúng lúc nó xảy ra.
    _log, _first, _step = [], [None], [0]
    _hs = []
    if HOOKS:
        def _stat(t):
            if not torch.is_tensor(t) or not t.is_floating_point(): return None
            f = t.detach().float(); fin = torch.isfinite(f)
            return dict(nan=int(torch.isnan(f).sum()), inf=int(torch.isinf(f).sum()),
                        absmax=(float(f[fin].abs().max()) if fin.any() else float("nan")),
                        shape=tuple(f.shape))
        def _mk(name):
            def h(mod, inp, out):
                _step[0] += 1
                for oi, o in enumerate(out if isinstance(out, (tuple, list)) else (out,)):
                    s = _stat(o)
                    if s is None: continue
                    _log.append((_step[0], name, oi, s))
                    if (s["nan"] or s["inf"]) and _first[0] is None:
                        ins = [_stat(x) for x in (inp if isinstance(inp, tuple) else (inp,))]
                        _first[0] = (_step[0], name, oi, s, [x for x in ins if x])
            return h
        for n, mod in m.named_modules():
            if n: _hs.append(mod.register_forward_hook(_mk(n)))

    for bi in range(nb):
        idx = ORDER[bi * BS:(bi + 1) * BS]
        if len(idx) < BS: break
        b = to_dev(col([ds[i] for i in idx]))
        m.zero_grad(set_to_none=True)
        if HOOKS: _log.clear(); _first[0] = None; _step[0] = 0
        out = m(**b); loss = out["loss"]; loss.backward()
        bp, tn, ti = bad_grads(m)
        if bp and HOOKS:
            print(f"\n    --- MODULE ĐẦU TIÊN NON-FINITE (batch {bi}) ---")
            if _first[0] is None:
                print("      forward SẠCH ở mọi module → NaN sinh trong BACKWARD.")
                _rk = [x for x in _log if x[3]['absmax'] == x[3]['absmax'] and x[3]['absmax'] < 1e30]
                for e in sorted(_rk, key=lambda x: -x[3]['absmax'])[:8]:
                    print(f"      absmax={e[3]['absmax']:>12.4g}  {e[1]}[{e[2]}]")
            else:
                st, nm, oi, s, ins = _first[0]
                print(f"      #{st} {nm}[{oi}] nan={s['nan']} inf={s['inf']} absmax={s['absmax']:.4g} shape={s['shape']}")
                for k, x in enumerate(ins):
                    print(f"        in[{k}]: nan={x['nan']} inf={x['inf']} absmax={x['absmax']:.4g}")
                for e in [x for x in _log if x[0] < st][-5:]:
                    print(f"        trước: #{e[0]} {e[1]}[{e[2]}] absmax={e[3]['absmax']:.4g}")
            for h in _hs: h.remove()
        if bp:
            mods = defaultdict(int)
            for n, _, _ in bp: mods[n.split(".")[0]] += 1
            for h in _hs: h.remove()
            return bi, idx, dict(loss=float(loss), finite=bool(torch.isfinite(loss).all()),
                                 nparam=len(bp), nan=tn, inf=ti, mods=dict(mods), first=bp[0])
    for h in _hs: h.remove()
    return None


# ─────────────── P2+P3+P4. MA TRẬN CẤU HÌNH × det/rec ON/OFF ───────────────
print("\n" + SEP); print("P2/P3/P4. QUÉT CẤU HÌNH  (det/rec ON rồi OFF)"); print(SEP)
results = {}
for cfg in ALL_CFG:
    name = cfg[0]
    row = {}
    for detrec in (True, False):
        tag = "ON " if detrec else "OFF"
        try:
            m, col = build(cfg, detrec=detrec)
            r = scan(m, col, NB)
            row["on" if detrec else "off"] = r
            if r is None:
                print(f"[{name:<13}] det/rec={tag} -> SẠCH ({NB} batch)")
            else:
                bi, idx, info = r
                kind = "NaN" if (info['nan'] and not info['inf']) else ("inf" if info['inf'] else "?")
                print(f"[{name:<13}] det/rec={tag} -> XẤU @batch {bi} rows={idx} | loại={kind} "
                      f"nan={info['nan']} inf={info['inf']} | {info['nparam']} param | mods={info['mods']}")
                print(f"{'':<16} param đầu: {info['first'][0]} (nan={info['first'][1]}, inf={info['first'][2]}) "
                      f"| loss={info['loss']:.3f} finite={info['finite']}")
            del m, col; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"[{name:<13}] det/rec={tag} -> OOM (bỏ qua; giảm BS)")
            row["on" if detrec else "off"] = "OOM"; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{name:<13}] det/rec={tag} -> LỖI: {type(e).__name__}: {e}")
            row["on" if detrec else "off"] = "ERR"; torch.cuda.empty_cache()
    results[name] = row

# ─────────────── P5. DETECT_ANOMALY trên batch lỗi đầu tiên ───────────────
print("\n" + SEP); print("P5. DETECT_ANOMALY (tìm op sinh NaN trong backward)"); print(SEP)
target = None
for cfg in ALL_CFG:
    r = results.get(cfg[0], {}).get("on")
    if isinstance(r, tuple):
        target = (cfg, r[0], r[1]); break
if target is None:
    print("Không có cấu hình nào lỗi -> bỏ qua P5.")
else:
    cfg, bi, idx = target
    print(f"Chạy lại cấu hình '{cfg[0]}' batch {bi} (rows={idx}) với detect_anomaly...")
    try:
        torch.autograd.set_detect_anomaly(True)
        m, col = build(cfg, detrec=True)
        b = to_dev(col([ds[i] for i in idx]))
        m.zero_grad(set_to_none=True)
        # QUAN TRỌNG: traceback của lời gọi FORWARD được phát ra dưới dạng UserWarning
        # NGAY TRƯỚC khi raise, KHÔNG nằm trong exception → phải ghi lại warning mới thấy
        # đúng dòng code. (Lần chạy trước bị filterwarnings('ignore') ở đầu file nuốt mất.)
        _err = None
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            try:
                out = m(**b); out["loss"].backward()
            except RuntimeError as e:
                _err = e
        if _err is not None:
            print("*** ANOMALY RAISE ***"); print(str(_err)[:2000])
        else:
            print("KHÔNG raise (có thể là inf, hoặc anomaly không bắt được).")
        for _wi in _w:
            _msg = str(_wi.message)
            if "Traceback of forward call" in _msg or "anomaly" in _msg.lower():
                print("\n*** TRACEBACK FORWARD (dòng code tạo ra op hỏng) ***")
                print(_msg[:4000])
    except Exception as e:
        print(f"lỗi khác: {type(e).__name__}: {e}"); traceback.print_exc()
    finally:
        torch.autograd.set_detect_anomaly(False)

# ─────────────────────────── P6. TỔNG KẾT ───────────────────────────
print("\n" + SEP); print("P6. TỔNG KẾT"); print(SEP)
print(f"{'config':<15}{'det/rec ON':<16}{'det/rec OFF':<16}")
print("-" * 50)
def _fmt(r):
    if r is None: return "SẠCH"
    if isinstance(r, str): return r
    return f"XẤU@b{r[0]}"
for cfg in ALL_CFG:
    row = results.get(cfg[0], {})
    print(f"{cfg[0]:<15}{_fmt(row.get('on')):<16}{_fmt(row.get('off')):<16}")

print("\nKẾT LUẬN TỰ ĐỘNG:")
_on_bad  = [c[0] for c in ALL_CFG if isinstance(results.get(c[0], {}).get("on"), tuple)]
_off_bad = [c[0] for c in ALL_CFG if isinstance(results.get(c[0], {}).get("off"), tuple)]
if not _on_bad:
    print("  • Không tái hiện lỗi trên máy này với dữ liệu/kịch bản đã quét.")
elif set(_on_bad) == set(_off_bad):
    print("  • det/rec ON và OFF LỖI GIỐNG NHAU  =>  det/rec KHÔNG phải nguyên nhân.")
    print("    Nguyên nhân nằm ở nhánh khác (xem mods/param đầu + traceback P5).")
elif _on_bad and not _off_bad:
    print("  • CHỈ lỗi khi det/rec ON  =>  det/rec ĐÚNG là nguyên nhân. Cần sửa cách tích hợp det/rec.")
else:
    print(f"  • Khác nhau một phần: ON lỗi {_on_bad} | OFF lỗi {_off_bad} — xem chi tiết ở trên.")
if "full" in _on_bad:
    print("  • LƯU Ý: 'full' bật OCR nên KHÔNG chạy code det/rec baseline. 'full' lỗi ở cả 2 cột")
    print("    là bằng chứng độc lập rằng det/rec vô can.")
print(SEP)
