import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure absolute project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import gc
import json
import torch
import numpy as np
import random
import pandas as pd
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    set_seed,
    GenerationConfig,
)

# Project imports
from configs.base_config import configure_env, OUTPUT_PATH, SEED
from configs.arguments import ModelArguments, DataArguments, CustomTrainingArguments
from configs.model_config import OpenViVQAConfig
from configs.ocr_config import DEFAULT_OCR_CONFIG
from models.openvivqa_model import OpenViVQAModel
from models.modules.ocr_encoder_feature import Vision_Encode_Ocr_Feature
from data.dataset_hub import DatasetHubLoader
from data.dataset import ViT5VQADataset
from data.collator import ViT5VQADataCollator
from training.pretraine_loss import ViT5PretrainLoss, GlobalPretrainAccuracy
from training.metrics import TaskSpecificTrainer, simple_pretrain_aggregator
from utils.io_utils import download_and_extract_checkpoint

# --- Torch >= 2.6 compat for resume ---------------------------------------------
# PyTorch 2.6 flipped torch.load's default to weights_only=True. HF Trainer's
# resume path does torch.load(rng_state.pth / optimizer.pt) without that arg, and
# those files hold numpy objects (RNG state) that weights_only=True rejects
# (numpy._core.multiarray._reconstruct not allowlisted). Our own checkpoints are
# trusted, so restore the pre-2.6 behavior (weights_only=False). Idempotent.
if not getattr(torch.load, "_viscet5_wo_patch", False):
    _orig_torch_load = torch.load
    def _torch_load_full(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    _torch_load_full._viscet5_wo_patch = True
    torch.load = _torch_load_full

def parse_args_with_yaml_and_cli(parser, args_list=None, default_yaml=None):
    import yaml
    is_jupyter = any("ipykernel" in arg or "colab" in arg for arg in sys.argv) or (len(sys.argv) > 0 and ("ipykernel_launcher" in sys.argv[0] or "colab_kernel_launcher" in sys.argv[0]))
    if args_list is not None:
        args = args_list
    elif is_jupyter:
        print(f"[Jupyter/Colab] Environment detected. Using default config: {default_yaml}")
        args = [default_yaml] if default_yaml is not None else []
    else:
        args = sys.argv[1:]
        
    if len(args) >= 1 and args[0].endswith(".yaml"):
        yaml_file = os.path.abspath(args[0])
        print(f"Loading configuration from YAML: {yaml_file}")
        with open(yaml_file, "r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f) or {}
            
        cli_args = args[1:]
        if len(cli_args) > 0:
            print(f"Applying CLI overrides: {cli_args}")
            option_to_dest = {}
            for action in parser._actions:
                for option_string in action.option_strings:
                    option_to_dest[option_string] = action.dest
            
            explicit_dests = set()
            for arg in cli_args:
                if arg.startswith("-"):
                    opt = arg.split("=")[0]
                    if opt in option_to_dest:
                        explicit_dests.add(option_to_dest[opt])
                    elif opt.startswith("--no-") and opt.replace("--no-", "--") in option_to_dest:
                        explicit_dests.add(option_to_dest[opt.replace("--no-", "--")])
            
            # Temporarily disable action.required to avoid argparse complaining about missing required options
            original_required = {}
            for action in parser._actions:
                original_required[action] = action.required
                action.required = False
            
            try:
                parsed_namespace = parser.parse_args(args=cli_args)
            finally:
                # Restore original required attributes
                for action, req in original_required.items():
                    action.required = req
                    
            for dest in explicit_dests:
                yaml_dict[dest] = getattr(parsed_namespace, dest)
                
        return parser.parse_dict(yaml_dict)
    else:
        return parser.parse_args_into_dataclasses(args=args)

def _tensor_health(t):
    """One-line numeric health report for a tensor (or 'None')."""
    if t is None:
        return "None"
    if not torch.is_tensor(t):
        return f"(not a tensor: {type(t).__name__})"
    f = t.detach().float()
    if f.numel() == 0:
        return f"shape={tuple(t.shape)} (empty)"
    return (f"shape={tuple(t.shape)} nan={bool(torch.isnan(f).any())} inf={bool(torch.isinf(f).any())} "
            f"min={f.min().item():.3g} max={f.max().item():.3g} mean={f.mean().item():.3g}")


def _verify_pretrain_batch(model, data_collator, dataset, loss_fn, acc_fn, device, use_twc):
    """
    Run ONE batch forward + backward and (a) DUMP detailed diagnostics for every
    pretrain component and (b) ASSERT they are computed and numerically sound.
    This is the fast "is the method + code correct, and if not WHERE?" gate
    before committing to a full run.

    Raises RuntimeError listing the failed checks if anything is wrong.
    """
    print("\n" + "=" * 70)
    print("🔬 [VERIFY] Single-batch pretrain method check (with diagnostics)")
    print("=" * 70)

    k = min(8, len(dataset))
    if k < 2:
        print("⚠️ [VERIFY] Need >= 2 samples (ITM pollute needs batch>1); skipping.")
        return

    raw = [dataset[i] for i in range(k)]
    batch = data_collator(raw)
    batch = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in batch.items()}

    model.train()
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    
    # PreSTU SplitOCR direct loss or legacy loss
    if "loss" in out and out["loss"] is not None:
        total = out["loss"]
        out["loss_gen"] = total
    else:
        if batch.get("gen_labels") is not None and batch.get("gen_input_ids") is not None:
            _op = model.pretrain
            model.pretrain = False
            try:
                _gen = model(
                    input_ids=batch.get("gen_input_ids"),
                    attention_mask=batch.get("gen_attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    pil_images=batch.get("pil_images"),
                    ocr_info=batch.get("ocr_info"),
                    ocr_mask_token=batch.get("ocr_mask_token"),
                    ocr_mask_box=batch.get("ocr_mask_box"),
                    labels=batch.get("gen_labels"),
                    twa_ocr_char=batch.get("gen_twa_ocr_char", batch.get("twa_ocr_char")),
                    twa_ocr_char_mask=batch.get("gen_twa_ocr_char_mask", batch.get("twa_ocr_char_mask")),
                    twa_word_ids=batch.get("gen_twa_word_ids", batch.get("twa_word_ids")),
                    ocr_to_word_map=batch.get("gen_ocr_to_word_map", batch.get("ocr_to_word_map")),
                )
            finally:
                model.pretrain = _op
            out["gen_loss"] = _gen.get("loss")
        total = loss_fn(batch, out)

    checks = []
    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # PreSTU SplitOCR check
    chk("[SplitOCR] loss produced & finite", total is not None and bool(torch.isfinite(total).all()),
        f"loss={total.item():.4f}" if total is not None else "None")

    # ── ITC checks (opt-in objective; only when ITC_WEIGHT>0) ──────────────
    _itcw = float(os.environ.get("ITC_WEIGHT", "0") or 0)
    if _itcw > 0:
        _li, _ai = out.get("loss_itc"), out.get("acc_itc")
        _qn = getattr(loss_fn, "_itc_txt_q", None)
        print(f"[diag] ITC: loss={float(_li):.4f} acc={float(_ai):.3f} "
              f"queue={_qn.size(0) if _qn is not None else 0} "
              f"dup_tau={getattr(loss_fn, 'itc_dup_tau', 0)} "
              f"text_pool={getattr(model, '_itc_text_pool', 'embed')} "
              f"text_source={getattr(model, '_itc_text_source', 'question')}")
        chk("[ITC] itc vectors produced",
            out.get("itc_img_vec") is not None and out.get("itc_txt_vec") is not None)
        chk("[ITC] loss_itc finite", _li is not None and bool(torch.isfinite(_li).all()))
        # ── ITC DEEP-DIAG: định vị degenerate (mock v4 từng đo loss ≡ ln(N), grad ≡ 0
        # — chữ ký của logits uniform). In đủ để chỉ mặt: stale module / scale chết /
        # text-vec co cụm / mask rỗng.
        _ivv, _tvv = out.get("itc_img_vec"), out.get("itc_txt_vec")
        if _ivv is not None and _tvv is not None:
            import math as _math
            import inspect as _inspect
            _ivd, _tvd = _ivv.detach().float(), _tvv.detach().float()
            _Bv = _tvd.size(0)
            _eye = torch.eye(_Bv, dtype=torch.bool, device=_tvd.device)
            _ct = (_tvd @ _tvd.t()).masked_fill(_eye, float("nan"))
            _ci = (_ivd @ _ivd.t()).masked_fill(_eye, float("nan"))
            _rawsc = float(model.itc_logit_scale.detach()) if hasattr(model, "itc_logit_scale") else float("nan")
            _expsc = _math.exp(min(_rawsc, 100.0)) if _rawsc == _rawsc else float("nan")
            _lg = _expsc * (_ivd @ _tvd.t())
            _spread = float((_lg.max(1).values - _lg.min(1).values).mean())
            def _rng(m):
                v = m[~torch.isnan(m)]
                return f"[{float(v.min()):.3f},{float(v.max()):.3f}]" if v.numel() else "[-]"
            print(f"[diag] ITC-deep: raw_scale={_rawsc:.3f} (exp={_expsc:.2f}) | "
                  f"cos_txt offdiag {_rng(_ct)} | cos_img offdiag {_rng(_ci)} | "
                  f"row-logit-spread={_spread:.4f} (≈0 = uniform/DEGENERATE)")
            _tk, _tn = out.get("itc_dbg_tokok"), out.get("itc_dbg_txt_norm")
            if _tk is not None:
                print(f"[diag] ITC-ocr: clean-token/sample={[int(x) for x in _tk.tolist()]} | "
                      f"txt_pooled norm={[round(float(x), 2) for x in _tn.tolist()]}")
            _fsrc = _inspect.getsource(type(model).forward)
            _has_ocr_branch = "_itc_text_source" in _fsrc
            print(f"[diag] ITC: model.forward có nhánh ocr: {_has_ocr_branch} "
                  f"(False = MODULE CŨ trong kernel — Runtime>Restart rồi chạy lại!)")
            chk("[ITC] model forward is v4 (ocr-branch present)", _has_ocr_branch)
            chk("[ITC] logits NOT degenerate (row-spread > 1e-3)", _spread > 1e-3,
                f"spread={_spread:.2e}; nếu fail xem cos_txt/cos_img và raw_scale ở trên")

    # ── GEN checks (only when a generative decoder objective is active) ────
    if gen_on:
        gl = out.get("gen_loss")
        chk("[GEN] gen_loss produced", gl is not None)
        chk("[GEN] gen_loss finite & > 0",
            gl is not None and bool(torch.isfinite(gl).all()) and gl.item() > 0,
            f"gen_loss={gl.item():.4f}" if gl is not None else "MISSING")
        chk("[GEN] gen_loss requires grad", gl is not None and bool(gl.requires_grad))

    # ── total + backward + per-submodule gradient localization ─────────────
    chk("[TOTAL] loss finite", bool(torch.isfinite(total).all()))
    chk("[TOTAL] loss requires grad", bool(total.requires_grad))
    total.backward()

    grp = {}  # submodule prefix -> [grad_sq_sum, has_bad]
    for nm_, p in model.named_parameters():
        if p.grad is None:
            continue
        pref = nm_.split(".")[0]
        g = p.grad.detach().float()
        bad = not bool(torch.isfinite(g).all())
        ent = grp.setdefault(pref, [0.0, False])
        ent[0] += float(g.pow(2).sum()) if not bad else 0.0
        ent[1] = ent[1] or bad
    print("\n[diag] per-submodule gradient (||grad|| and NaN/inf flag):")
    for pref in sorted(grp):
        gnorm = grp[pref][0] ** 0.5
        flag = "  ⚠️ NaN/inf!" if grp[pref][1] else ""
        print(f"    {pref:24s} ||g||={gnorm:.4g}{flag}")
    bad_mods = [pref for pref, v in grp.items() if v[1]]
    chk("[GRAD] all gradients finite (no NaN/inf)", len(bad_mods) == 0,
        f"bad submodules: {bad_mods}" if bad_mods else "ok")
    chk("[GRAD] gradients flow into params", len(grp) > 0, f"{len(grp)} submodules got grad")

    # ── metric aggregation path ───────────────────────────────────────────
    try:
        acc = acc_fn.calculate(batch, out)
        acc_t = acc if torch.is_tensor(acc) else torch.tensor(float(acc))
        chk("[METRIC] aggregator runs & finite", bool(torch.isfinite(acc_t).all()),
            f"vec={[round(float(x), 3) for x in acc_t.flatten().tolist()[:8]]}")
    except Exception as e:
        chk("[METRIC] aggregator runs", False, f"raised {type(e).__name__}: {e}")

    model.zero_grad(set_to_none=True)

    print("\n[result]")
    all_ok = True
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  ({detail})" if detail else ""))
        all_ok = all_ok and ok
    print("=" * 70)
    if not all_ok:
        failed = [n for n, ok, _ in checks if not ok]
        raise RuntimeError(
            f"🔬 [VERIFY] Pretrain method check FAILED at: {failed}. "
            f"See [diag] lines above to localize which loss/component is wrong."
        )
    print("🔬 [VERIFY] All pretrain components OK — proceeding to the short training loop.\n")


def _mlm_masked_acc(out, batch, device):
    """MLM token-accuracy on masked (label != -1) & non-polluted positions."""
    logits = out["textcls_scores"].detach()
    tgt = batch["cmb_text_mask_label"].to(logits.device)
    B, L, V = logits.shape
    pollute = batch["tag_pollute"].to(logits.device).float().view(B, 1).expand(B, L)
    mask = (tgt != -1) & (pollute <= 0.5)
    n = int(mask.sum().item())
    if n == 0:
        return float("nan"), 0
    pred = logits.argmax(-1)
    return (pred[mask] == tgt[mask]).float().mean().item(), n


def _debug_split_ocr(model, data_collator, dataset, device, n_show=5):
    """Readable SplitOCR debug: show the SplitOCR prompt (prefix words), the
    gold target words, and the model prediction."""
    print("\n" + "=" * 70)
    print("🔎 [PreSTU SplitOCR DEBUG] Prompt (Prefix) -> Target vs Predicted")
    print("=" * 70)
    k = min(8, len(dataset))
    if k < 1:
        return
    tok = data_collator.tokenizer
    batch = data_collator([dataset[i] for i in range(k)])
    batch = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in batch.items()}
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    logits = out.get("logits")
    if logits is None:
        model.train()
        return
    pred = logits.argmax(-1)
    labels = batch["labels"]
    in_ids = batch["input_ids"]
    shown = 0
    for i in range(labels.size(0)):
        pos = [p for p, t in enumerate(labels[i].tolist()) if t != -100]
        prompt = tok.decode([int(t) for t in in_ids[i].tolist() if int(t) != data_collator.pad_id], skip_special_tokens=True).strip()
        gold = tok.decode([int(labels[i][p]) for p in pos], skip_special_tokens=True).strip()
        prd = tok.decode([int(pred[i][p].item()) for p in pos], skip_special_tokens=True).strip()
        print(f"  [sample {i}] Prompt: {prompt[:90]}")
        print(f"              Target: {gold[:80]}")
        print(f"              Pred  : {prd[:80]}")
        shown += 1
        if shown >= n_show:
            break
    print("=" * 70 + "\n")
    model.train()


def _debug_mlm_predictions(model, data_collator, dataset, device, n_show=5):
    """Human-readable MLM debug: for a few CLEAN samples, show the masked input text
    and, for each masked WORD, the gold token vs the model's prediction (argmax).
    Consecutive masked subwords (a whole-word mask) are grouped so it reads at the
    word level: 'gold=pepsi | pred=pepsi ✓'."""
    print("\n" + "=" * 70)
    print("🔎 [MLM DEBUG] câu bị mask → model dự đoán (word-level)")
    print("=" * 70)
    k = min(8, len(dataset))
    if k < 2:
        print("  skip (need >= 2 samples)"); return
    tok = data_collator.tokenizer
    batch = data_collator([dataset[i] for i in range(k)])
    batch = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in batch.items()}
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    logits = out.get("textcls_scores")
    if logits is None:
        print("  (no textcls_scores)"); model.train(); return
    pred = logits.argmax(-1)                      # (B, L)
    ids = batch["mlm_input_ids"]
    labels = batch["cmb_text_mask_label"]
    pollute = batch["tag_pollute"].view(-1)
    pad_id = data_collator.pad_id
    shown = 0
    for i in range(ids.size(0)):
        if int(pollute[i].item()) == 1:
            continue                              # MLM chỉ tính trên mẫu sạch
        lab = labels[i].tolist()
        mask_pos = [p for p, v in enumerate(lab) if v != -1]
        if not mask_pos:
            continue
        # readable masked input (bỏ pad, giữ <extra_id_0>)
        vis = [int(t) for t in ids[i].tolist() if int(t) != pad_id]
        inp_txt = tok.decode(vis, skip_special_tokens=False)
        # gộp các vị trí mask liên tiếp thành 1 "từ"
        spans, cur = [], [mask_pos[0]]
        for p in mask_pos[1:]:
            if p == cur[-1] + 1:
                cur.append(p)
            else:
                spans.append(cur); cur = [p]
        spans.append(cur)
        print(f"\n[sample {i}] input(masked): {inp_txt[:180]}")
        print(f"     (#mask positions={len(mask_pos)}, gộp thành {len(spans)} từ)")
        for wi, sp in enumerate(spans[:10]):
            gold_ids = [int(lab[p]) for p in sp]
            pred_ids = [int(pred[i][p].item()) for p in sp]
            gold_toks = tok.convert_ids_to_tokens(gold_ids)   # subword tokens (thô)
            pred_toks = tok.convert_ids_to_tokens(pred_ids)
            gold_word = tok.decode(gold_ids).strip()           # gộp lại thành word
            pred_word = tok.decode(pred_ids).strip()
            mark = "✓" if gold_word == pred_word else "✗"
            print(f"     từ {wi} @pos{sp[0]}..{sp[-1]}:")
            print(f"        gold: tokens={gold_toks} → '{gold_word}'")
            print(f"        pred: tokens={pred_toks} → '{pred_word}'  {mark}")
        shown += 1
        if shown >= n_show:
            break
    if shown == 0:
        print("  (không có mẫu sạch nào có vị trí mask trong batch này)")
    print("=" * 70 + "\n")
    model.train()


    if shown == 0:
        print("  (no cloze spans in this batch — no question∩OCR overlap)")
    print("=" * 70 + "\n")
    model.train()


def _ablate_ocr_info(ocr_info):
    """Return a copy of ocr_info with the VISUAL fields (det/rec/box) zeroed — used by
    the OCR-ablation eval. Shapes preserved (only values zeroed) so the forward runs."""
    if ocr_info is None:
        return None
    out = []
    for d in ocr_info:
        d2 = dict(d)
        for k in ("det_features", "rec_features", "boxes", "boxes_word_all"):
            v = d2.get(k)
            if v is None:
                continue
            if torch.is_tensor(v):
                d2[k] = torch.zeros_like(v)
            else:
                import numpy as _np
                d2[k] = _np.zeros_like(v)
        out.append(d2)
    return out


def _debug_ocr_ablation(model, data_collator, dataset, device, n_batch=12, bs=8):
    """GROUNDING EVIDENCE: run the decoder MLM (cloze) forward with the OCR branch ON vs
    ABLATED (char zeroed, word_ids -> pad, det/rec/box zeroed), and compare grounded vs
    random token accuracy. If GROUNDED collapses when OCR is off (while RANDOM barely
    moves) → the grounded words are recovered by READING the OCR, not by the LM prior."""
    print("\n" + "=" * 70)
    print("🧪 [OCR-ABLATION] does the decoder recover GROUNDED words by reading OCR?")
    print("=" * 70)
    tok_pad = data_collator.pad_id
    model.eval()
    orig = getattr(model, "pretrain", False)
    model.pretrain = False

    def _counts(batch, ablate):
        gi, gl = batch.get("gen_input_ids"), batch.get("gen_labels")
        mtype = batch.get("gen_mlm_type")
        if gi is None or gl is None or mtype is None:
            return None
        char = batch.get("gen_twa_ocr_char", batch.get("twa_ocr_char"))
        wid = batch.get("gen_twa_word_ids", batch.get("twa_word_ids"))
        cmask = batch.get("gen_twa_ocr_char_mask", batch.get("twa_ocr_char_mask"))
        o2w = batch.get("gen_ocr_to_word_map", batch.get("ocr_to_word_map"))
        oinfo = batch.get("ocr_info")
        if ablate:
            char = torch.zeros_like(char) if char is not None else None
            wid = torch.full_like(wid, tok_pad) if wid is not None else None
            oinfo = _ablate_ocr_info(oinfo)
        out = model(
            input_ids=gi, attention_mask=batch.get("gen_attention_mask"),
            pixel_values=batch.get("pixel_values"), pil_images=batch.get("pil_images"),
            ocr_info=oinfo, ocr_mask_token=batch.get("ocr_mask_token"),
            ocr_mask_box=batch.get("ocr_mask_box"), labels=gl,
            twa_ocr_char=char, twa_ocr_char_mask=cmask, twa_word_ids=wid, ocr_to_word_map=o2w,
        )
        logits = out.get("logits")
        if logits is None:
            return None
        pred = logits.argmax(-1)
        m = gl != -100
        eq = (pred == gl)
        gm, rm = m & (mtype == 1), m & (mtype == 0)
        return [float((eq & gm).sum()), float(gm.sum()), float((eq & rm).sum()), float(rm.sum())]

    on = [0.0, 0.0, 0.0, 0.0]; off = [0.0, 0.0, 0.0, 0.0]
    n = min(n_batch * bs, len(dataset))
    with torch.no_grad():
        for s in range(0, n, bs):
            idx = list(range(s, min(s + bs, len(dataset))))
            if len(idx) < 2:
                break
            batch = data_collator([dataset[i] for i in idx])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            c_on = _counts(batch, ablate=False)
            c_off = _counts(batch, ablate=True)
            if c_on is None or c_off is None:
                continue
            on = [a + b for a, b in zip(on, c_on)]
            off = [a + b for a, b in zip(off, c_off)]
    model.pretrain = orig
    model.train()

    def _acc(c, i):  # correct/total for grounded (i=0) or random (i=2)
        return (c[i] / c[i + 1]) if c[i + 1] > 0 else 0.0
    g_on, g_off = _acc(on, 0), _acc(off, 0)
    r_on, r_off = _acc(on, 2), _acc(off, 2)
    print(f"  GROUNDED (OCR words): OCR-on={g_on:.3f}  OCR-off={g_off:.3f}  drop={g_on-g_off:+.3f}")
    print(f"  RANDOM   (LM words) : OCR-on={r_on:.3f}  OCR-off={r_off:.3f}  drop={r_on-r_off:+.3f}")
    print("  → large GROUNDED drop + small RANDOM drop = grounded words are read from OCR (grounding real).")
    print("=" * 70 + "\n")


def _diagnose_mlm_crutch(model, data_collator, dataset, device):
    """
    Measure the MLM "copy crutch": how much MLM relies on the OCR-FEATURE branch.
    Compare MLM masked-token accuracy (a) FULL vs (b) with the OCR-feature branch
    ABLATED (twa_word_ids -> pad, so those positions are masked out of attention).
    Small drop => MLM solves masked OCR from the TEXT branch (the crutch), NOT from
    the feature branch => feature branch under-trained. Large drop => MLM genuinely
    needs the feature branch. Run under each mask mode to compare.
    """
    print("\n" + "=" * 70)
    print("🔍 [DIAG] MLM 'copy-crutch' A/B — reliance on the OCR-feature branch")
    print("   (model chưa train ở smoke → nhìn để ĐẢM BẢO code chạy + so tương đối;")
    print("    số DROP có ý nghĩa nhất SAU khi train — xem lại ở eval/full run)")
    print("=" * 70)
    k = min(8, len(dataset))
    if k < 2:
        print("  skip (need >= 2 samples)"); return
    tok = data_collator.tokenizer
    orig_mode = getattr(data_collator, "mlm_mask_mode", "wholeword")
    model.eval()
    for mode in ["wholeword", "subword"]:
        data_collator.mlm_mask_mode = mode
        batch = data_collator([dataset[i] for i in range(k)])
        batch = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in batch.items()}
        with torch.no_grad():
            out_full = model(**batch)
            acc_full, n = _mlm_masked_acc(out_full, batch, device)
            b2 = dict(batch)
            if batch.get("twa_word_ids") is not None:
                b2["twa_word_ids"] = torch.full_like(batch["twa_word_ids"], data_collator.pad_id)
            b2["o2r_labels"] = None; b2["r2o_labels"] = None; b2["twc_group_ids"] = None
            try:
                out_abl = model(**b2)
                acc_abl, _ = _mlm_masked_acc(out_abl, batch, device)
            except Exception as e:
                print(f"  (ablated forward failed: {type(e).__name__}: {e})"); acc_abl = float("nan")
        # decode what got masked in sample 0 (shows whole-word vs subword)
        lab0 = batch["cmb_text_mask_label"][0].tolist()
        masked_ids = [t for t in lab0 if t != -1]
        sample = tok.decode(masked_ids) if masked_ids else "(none)"
        drop = (acc_full - acc_abl) if (acc_full == acc_full and acc_abl == acc_abl) else float("nan")
        print(f"  [{mode:9s}] masked={n:4d} | acc FULL={acc_full:.3f} | acc OCR-ablated={acc_abl:.3f} | DROP={drop:.3f}")
        print(f"             mask-targets(sample0): {sample[:120]}")
    data_collator.mlm_mask_mode = orig_mode
    print("  → kỳ vọng: wholeword DROP lớn hơn subword (buộc dùng feature nhiều hơn)")
    print("=" * 70 + "\n")
    model.train()


def _progress_only_mode():
    """
    True when we should show only the trainer's tqdm progress bar and suppress the
    verbose per-step debug logs — i.e. on Colab. Triggered by env VISCET5_PROGRESS_ONLY
    (set by run_all_colab.sh) or by detecting the google.colab runtime.
    """
    if os.environ.get("VISCET5_PROGRESS_ONLY", "").strip().lower() in ("1", "true", "yes"):
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def main(args_list=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    default_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "pretrain.yaml"))
    model_args, data_args, training_args = parse_args_with_yaml_and_cli(parser, args_list, default_yaml)
    
    set_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)
        
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Prepare Data
    print(f">>> Preparing Dataset: {data_args.dataset_name}")
    from configs.base_config import OUTPUT_PATH
    train_csv = os.path.join(OUTPUT_PATH, "merged_train.csv")
    val_csv = os.path.join(OUTPUT_PATH, "merged_val.csv")
    
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        print(f"ℹ️ Found prepared CSV cache in {OUTPUT_PATH}. Loading directly...")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    else:
        print(f"ℹ️ CSV cache not found. Preparing via Hub...")
        raw_dir = os.path.join(data_args.data_dir, "raw")
        out_dir = os.path.join(data_args.data_dir, "processed")
        hub = DatasetHubLoader(raw_dir, out_dir)
        
        import yaml
        ds_names = [d.strip() for d in data_args.dataset_name.split(",") if d.strip()]
        all_train_dfs, all_val_dfs = [], []
        
        for ds_name in ds_names:
            config_path = f"configs/data/{ds_name}.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    ds_cfg = yaml.safe_load(f)
                
                ds_sec = ds_cfg.get('dataset', {})
                hub.register_dataset(
                    dataset_name=ds_cfg['dataset_name'],
                    task_type="VQA",
                    image_zip_id=ds_cfg.get('image', {}).get('drive_id'),
                    image_dir_override='',
                    ocr_zip_id=ds_cfg.get('ocr', {}).get('drive_id'),
                    ocr_dir_override='',
                    splits={
                        "train":      {"id": ds_sec.get('train', {}).get('drive_id') or ds_sec.get('train', {}).get('dir'), "url": None},
                        "validation": {"id": ds_sec.get('validation', {}).get('drive_id') or ds_sec.get('validation', {}).get('dir'), "url": None},
                        "test":       {"id": ds_sec.get('test', {}).get('drive_id') or ds_sec.get('test', {}).get('dir'), "url": None},
                    }
                )
                print(f"⬇️  Preparing {ds_name} via Hub...")
                hub.prepare(ds_name)
                dfs = hub.load_task(ds_name)
                if len(dfs["train"]) > 0: all_train_dfs.append(dfs["train"])
                if len(dfs["validation"]) > 0: all_val_dfs.append(dfs["validation"])
                print(f"   ✅ {ds_name} Ready: {len(dfs['train'])} train, {len(dfs['validation'])} val.")
            else:
                print(f"⚠️ Dataset config not found at {config_path}. Assuming it's already registered or manually ready.")
        
        if all_train_dfs:
            train_df = pd.concat(all_train_dfs, ignore_index=True).sample(frac=1, random_state=training_args.seed).reset_index(drop=True)
            val_df = pd.concat(all_val_dfs, ignore_index=True).sample(frac=1, random_state=training_args.seed).reset_index(drop=True) if all_val_dfs else pd.DataFrame()
            print(f"🔄 Combined Pretrain Datasets: {len(train_df)} train, {len(val_df)} val samples.")
        else:
            print("❌ No datasets loaded. Please run scripts/prepare_dataset.py first.")
            return

    if training_args.smoke_test:
        n_tr = int(getattr(training_args, "smoke_train_samples", 256))
        n_ev = int(getattr(training_args, "smoke_eval_samples", 64))
        n_steps = int(getattr(training_args, "smoke_max_steps", 60))
        print(f"🚨 SMOKE TEST MODE: train={n_tr}, eval={n_ev}, max_steps={n_steps} "
              f"— exercises the FULL pretrain pipeline fast with per-loss debug logging.")
        train_df = train_df.head(n_tr)
        val_df = val_df.head(n_ev)
        # Real optimizer steps (no accumulation) so the loop + scheduler are genuinely exercised.
        training_args.gradient_accumulation_steps = 1
        # ITM pollute logic needs batch > 1; keep small but >= 2.
        training_args.per_device_train_batch_size = max(2, min(4, int(training_args.per_device_train_batch_size)))
        training_args.per_device_eval_batch_size = max(2, min(4, int(training_args.per_device_eval_batch_size)))
        training_args.max_steps = n_steps
        training_args.num_train_epochs = 1.0
        training_args.warmup_steps = 0
        training_args.warmup_ratio = 0.0
        training_args.logging_steps = max(1, n_steps // 20)
        training_args.eval_steps = max(1, n_steps // 3)
        training_args.save_steps = n_steps
        training_args.save_total_limit = 1
        # Avoid best-model bookkeeping that needs many aligned eval/save steps.
        training_args.load_best_model_at_end = False
        # MOCK: ALWAYS print full debug (per-step Loss(M/I/TWC/GEN) + post-train
        # MLM/GEN debug). No env needed.
        import training.metrics as _M
        _M.DEBUG_TRAIN = True
        _M.LOG_TRAIN_EVERY = training_args.logging_steps

    train_dataset = ViT5VQADataset(train_df)
    val_dataset = ViT5VQADataset(val_df)

    # 2. OCR Encoder
    ocr_config = DEFAULT_OCR_CONFIG
    vision_ocr = Vision_Encode_Ocr_Feature(ocr_config)
    
    # 3. Handle Checkpoint Downloads
    if training_args.resume_checkpoint_id:
        resume_dir = os.path.join(training_args.output_dir, "resume_ckpt")
        download_and_extract_checkpoint(training_args.resume_checkpoint_id, resume_dir)
        training_args.resume_from_checkpoint = resume_dir

    ckpt_to_load = model_args.model_name_or_path

    # 4. Tokenizer & Model
    if ckpt_to_load:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, local_files_only=True)
        # OpenViVQAConfig isn't registered with AutoConfig (model_type 'openvivqa'),
        # so AutoConfig.from_pretrained would raise. Build the custom config directly.
        try:
            config = OpenViVQAConfig.from_pretrained(ckpt_to_load, local_files_only=True)
        except Exception as _e:
            print(f"ℹ️ Could not read config.json ({_e}); using default OpenViVQAConfig().")
            config = OpenViVQAConfig()
    else:
        # Fallback to defaults
        tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        config = OpenViVQAConfig()

    model = OpenViVQAModel(config)
    if ckpt_to_load:
        print(f"\n📥 Loading weights manually from: {ckpt_to_load}")
        safe_path = os.path.join(ckpt_to_load, "model.safetensors")
        bin_path = os.path.join(ckpt_to_load, "pytorch_model.bin")
        state_dict = None
        if os.path.exists(safe_path):
            state_dict = load_file(safe_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        if state_dict:
            new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            res = model.load_state_dict(new_state_dict, strict=False)
            n_model = len(model.state_dict())
            n_loaded = n_model - len(res.missing_keys)
            print(f"✅ Loaded {n_loaded}/{n_model} model tensors from checkpoint "
                  f"({len(new_state_dict)} in ckpt) | missing={len(res.missing_keys)} | "
                  f"unexpected={len(res.unexpected_keys)}")
            if n_loaded < 0.5 * n_model:
                print("🚨 CẢNH BÁO: <50% tensor được nạp — trọng số gần như KHÔNG chuyển sang!")

    # Apply config overrides
    mode = model_args.loss_ablation_mode
    # 'gen_all' = generative decoder pretrain + MLM/ITM/TWC auxiliaries (needs OCR-aug + TWC).
    # 'gen'     = generative decoder pretrain + MLM/ITM only (no TWC/OCR-aug).
    use_twc = mode in ["all", "only_twc_ocr_aug", "gen_all"]
    use_ocr_aug = mode in ["all", "only_twc_ocr_aug", "gen_all"]
    
    model.pretrain = True
    # Marks this model instance as being in the PRETRAIN stage so the forward's
    # numerical guard (fused_seq nan_to_num) activates even inside the gen forward
    # (which flips self.pretrain=False). Finetune never sets this → forward untouched.
    model._pretrain_stage = True
    # Persisted flag: weights are trained UNDER the vision clamp → any later stage
    # (finetune/predict) loading this checkpoint must keep the clamp (part of the
    # learned function). Lives in config so it survives save/load automatically.
    model.config.clamp_vision = True
    model.config.pretrain = True
    model.config.pretrain_ablation_mode = mode
    model.config.use_twc = use_twc
    model.config.use_ocr_aug_finetune = use_ocr_aug
    
    model.to(DEVICE)

    # PRETRAIN-ONLY partial vision unfreeze (representation learning). Done HERE in
    # the training script via requires_grad — NOT inside models/ — so the model
    # architecture stays identical to main and finetune (which rebuilds the model)
    # is completely unaffected. n=0 keeps the frozen backbone (default).
    _vuf = int(getattr(model_args, "vision_unfreeze_last_n", 0))
    if _vuf > 0:
        try:
            _layers = model.qa_clip.vision_model.encoder.layers
            _k = min(_vuf, len(_layers)); _cnt = 0
            for _layer in _layers[-_k:]:
                for _p in _layer.parameters():
                    if not _p.requires_grad:
                        _p.requires_grad = True; _cnt += _p.numel()
            _pln = getattr(model.qa_clip.vision_model, "post_layernorm", None)
            if _pln is not None:
                for _p in _pln.parameters():
                    if not _p.requires_grad:
                        _p.requires_grad = True; _cnt += _p.numel()
            # Tell the forward NOT to no_grad/detach QA-CLIP (grads must flow to the
            # unfrozen layers). NaN root fix + fused_seq guard keep it stable; the
            # training_step qa_clip grad-clip prevents explosion.
            model._vision_trainable = True
            print(f"🧊➡️🔥 [pretrain] vision unfreeze: last {_k}/{len(_layers)} CLIP-vision "
                  f"layers (+post_layernorm) → +{_cnt:,} params trainable. (grads ON)")
        except Exception as _e:
            print(f"⚠️ [pretrain] vision unfreeze skipped ({_e}).")

    # 5. Loss, Metrics, Collator
    pretrain_loss_fn = ViT5PretrainLoss(pretrain_ablation_mode=mode)
    pretrain_acc_fn = GlobalPretrainAccuracy(mode=mode)

    data_collator = ViT5VQADataCollator(
        tokenizer=tokenizer,
        image_processor=model.image_processor,
        ocr_encoder=vision_ocr,
        config=model.config,
        term_vocab_path=data_args.vocab_path,
        viet_vocab_path=data_args.viet_vocab_path,
        eng_vocab_path="",
        dataframe=train_df,
        pretrain=True,
        debug=False,
    )
    if hasattr(data_collator, "set_mode"):
        data_collator.set_mode(pretrain=True, mask_prob=0.15)
    data_collator.pretrain_ablation_mode = mode
    data_collator.use_ocr_aug_pretrain = use_ocr_aug
    data_collator.mlm_mask_mode = str(getattr(model_args, "mlm_mask_mode", "wholeword")).lower().strip()
    data_collator.mlm_ocr_in_text = bool(getattr(model_args, "mlm_ocr_in_text", False))

    # ---- HARD-PRETRAIN knobs (v2, env-gated; defaults keep current behavior) ----
    # Because ViT5 is already pretrained we can afford harder/adversarial pretext than
    # TWA's conservative from-scratch ratios. All opt-in via env so the baseline run
    # (and its checkpoint) is unaffected.
    _adv = os.environ.get("TWC_ADV_PROB", "").strip()
    if _adv:
        data_collator.adv_probability_pretrain = float(_adv)  # ↑ = stronger OCR corruption (harder TWC positives)
    _dupbox = os.environ.get("TWC_DUP_BOX", "").strip()
    if _dupbox:
        data_collator.twc_dup_box = _dupbox not in ("0", "false", "False")  # 0 = drop positional shortcut
    _rand = os.environ.get("MLM_RAND_PROB", "").strip()
    if _rand:
        data_collator.mlm_rand_prob = float(_rand)  # ↑ = more random-masked words (harder MLM)
    # ---- v3 transfer knobs (env-gated; defaults keep v2 behavior) ----
    _gts = os.environ.get("GEN_TARGET_STYLE", "").strip().lower()
    if _gts:
        # 'sentinel' (T5 span-infill) | 'qa' (single grounded span, RAW target =
        # finetune's answer format — fixes the sentinel verbosity transfer gap).
        data_collator.gen_target_style = _gts
    _ipol = os.environ.get("ITM_POLLUTE", "").strip()
    if _ipol:
        # 0 = no OCR swap (use when ITM_WEIGHT=0): image<->OCR matched for 100% samples.
        data_collator.itm_pollute = _ipol not in ("0", "false", "False")
    _itp = os.environ.get("ITC_TEXT_POOL", "").strip().lower()
    if _itp:
        # 'embed' (default, static input embeddings) | 'encoder' (v3: pool the vit5
        # encoder output → real sentence vector, ITC also shapes the text encoder).
        # Pretrain-only attribute (like _vision_trainable); finetune never sets it.
        model._itc_text_pool = _itp
    _its = os.environ.get("ITC_TEXT_SOURCE", "").strip().lower()
    if _its:
        # 'question' (v3) | 'ocr' (v4: image↔chuỗi-OCR — câu hỏi template không đủ
        # thông tin đặc-định-ảnh nên ITC ghim ln(4); chuỗi OCR định danh ảnh duy nhất
        # → học được + ép CLIP mở băng học đọc scene text). Pretrain-only attr.
        model._itc_text_source = _its
    print(f">>> [pretrain] hard-knobs: adv_prob={data_collator.adv_probability_pretrain} "
          f"twc_dup_box={getattr(data_collator,'twc_dup_box',True)} "
          f"mlm_rand_prob={getattr(data_collator,'mlm_rand_prob',0.15)} "
          f"itm_weight={os.environ.get('ITM_WEIGHT','0')} | "
          f"gen_style={getattr(data_collator,'gen_target_style','sentinel')} "
          f"itm_pollute={getattr(data_collator,'itm_pollute',True)} "
          f"itc_queue={os.environ.get('ITC_QUEUE','0')} "
          f"itc_dup_tau={os.environ.get('ITC_DUP_TAU','0')} "
          f"itc_text_pool={getattr(model,'_itc_text_pool','embed')} "
          f"itc_text_source={getattr(model,'_itc_text_source','question')}")

    _cloze = str(mode).lower().strip() in ("gen", "gen_all")
    if _cloze:
        _itmw = os.environ.get("ITM_WEIGHT", "0")
        print(f">>> [pretrain] objective = MLM (decoder span-infill) + TWC"
              f"{' + ITM' if _itmw not in ('0','false','False') else ' (ITM off: dead at chance)'} "
              "| MLM = mask OCR-overlap + random words, distinct <extra_id_i>, decoder regenerates "
              "them (not an encoder head) | question-only encoder")
    else:
        print(f">>> [pretrain] MLM mask mode = {data_collator.mlm_mask_mode} | ocr_in_text = {data_collator.mlm_ocr_in_text} "
              f"({'question+OCR' if data_collator.mlm_ocr_in_text else 'QUESTION-ONLY (nạng giảm)'}) | (legacy encoder-MLM path)")

    # 6. Trainer
    trainer = TaskSpecificTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=simple_pretrain_aggregator,
        pretrain_loss_fn=pretrain_loss_fn,
        pretrain_acc_fn=pretrain_acc_fn,
    )

    # Clean console output: keep only the TRAIN progress bar + eval results.
    # Swap HF's default tqdm ProgressCallback (which also draws an eval bar and
    # prints every train-step loss/lr) for CleanProgressCallback. Only applies to
    # the non-notebook (bash) path where ProgressCallback is active.
    try:
        from transformers.trainer_callback import ProgressCallback, PrinterCallback
        from training.metrics import CleanProgressCallback
        if any(isinstance(cb, ProgressCallback) for cb in trainer.callback_handler.callbacks):
            trainer.remove_callback(ProgressCallback)
            trainer.remove_callback(PrinterCallback)
            trainer.add_callback(CleanProgressCallback())
    except Exception as _e:
        print(f"ℹ️ Could not install CleanProgressCallback ({_e}); using default logging.")

    # Debug policy: MOCK always debugs; FULL run debugs ONLY if TWC_TRAIN_LOG=1
    # (default OFF → only progress bar + val eval, no spam/lag).
    pretrain_debug = bool(training_args.smoke_test) or (
        os.environ.get("TWC_TRAIN_LOG", "").strip().lower() in ("1", "true", "yes"))
    # ALWAYS set the flag (True or False). `training.metrics.DEBUG_TRAIN` is module-level
    # state that PERSISTS across importlib.reload(pretrain) (metrics isn't reloaded), so a
    # prior mock run in the same kernel would otherwise leave it True → full run spams
    # [Pretrain] logs. Forcing it here guarantees full-run = clean unless TWC_TRAIN_LOG=1.
    import training.metrics as _M
    _M.DEBUG_TRAIN = bool(pretrain_debug)
    if pretrain_debug:
        _M.LOG_TRAIN_EVERY = max(1, int(getattr(training_args, "logging_steps", 50)))

    # Fast method-correctness gate: in smoke/mock mode, verify a single batch
    # exercises MLM + ITM + TWC correctly before spending time on the loop.
    if training_args.smoke_test:
        _verify_pretrain_batch(
            model, data_collator, train_dataset,
            pretrain_loss_fn, pretrain_acc_fn, DEVICE, use_twc,
        )

    # ── Upload TỪNG checkpoint NGAY khi lưu ──
    _hf_tok = os.environ.get("HF_TOKEN", "").strip()
    _hf_repo = os.environ.get("HF_REPO", "").strip()
    if _hf_tok and _hf_repo:
        from transformers.trainer_callback import TrainerCallback
        from huggingface_hub import HfApi
        class _PushEachCheckpoint(TrainerCallback):
            def __init__(self, repo, token, out_dir):
                self.api = HfApi(token=token); self.repo = repo; self.out = out_dir
                self.api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
            def on_save(self, args, state, control, **kw):
                ck = os.path.join(self.out, f"checkpoint-{state.global_step}")
                if not os.path.isdir(ck):
                    return
                _model = kw.get("model")
                if _model is not None:
                    _bad = [n for n, p in _model.named_parameters() if not torch.isfinite(p).all()]
                    if _bad:
                        print(f"🛑 [HF] checkpoint-{state.global_step}: model có {len(_bad)} tensor NaN/inf "
                              f"(vd {_bad[0]}) → BỎ push (không nhiễm nguồn resume).")
                        return
                try:
                    _light = os.environ.get("HF_PUSH_OPTIM", "1").lower() in ("0", "false", "no", "off")
                    _ignore = ["optimizer.pt", "rng_state*.pth", "scheduler.pt"] if _light else None
                    self.api.upload_folder(folder_path=ck, path_in_repo=f"checkpoint-{state.global_step}",
                                           repo_id=self.repo, repo_type="model",
                                           ignore_patterns=_ignore)
                    print(f"☁️ [HF] đã push checkpoint-{state.global_step} "
                          f"({'weights-only' if _light else 'ĐẦY ĐỦ/resume-được'}) → {self.repo}")
                except Exception as e:
                    print(f"⚠️ [HF] push checkpoint-{state.global_step} lỗi: {e}")
        try:
            trainer.add_callback(_PushEachCheckpoint(_hf_repo, _hf_tok, training_args.output_dir))
            print(f"☁️ [HF] bật auto-push mỗi checkpoint → {_hf_repo}")
        except Exception as _e:
            print(f"ℹ️ [HF] không bật được auto-push callback ({_e}); vẫn có upload cuối ở run_pipeline.")

    # ── Auto-resume TỪ HF repo ──
    if (not training_args.resume_from_checkpoint and _hf_tok and _hf_repo
            and not training_args.smoke_test
            and os.environ.get("RESUME_FROM_HF", "auto").lower() not in ("0", "false", "no", "off")):
        try:
            from huggingface_hub import list_repo_files, snapshot_download
            from safetensors.torch import load_file as _load_sft
            _files = list_repo_files(_hf_repo, token=_hf_tok)
            _cks = sorted({f.split("/")[0] for f in _files if f.startswith("checkpoint-")},
                          key=lambda x: int(x.split("-")[1]))
            if _cks:
                _latest = _cks[-1]
                _has_optim = f"{_latest}/optimizer.pt" in _files
                print(f"♻️ [HF-resume] repo có {len(_cks)} checkpoint; mới nhất={_latest} "
                      f"(optimizer.pt: {'CÓ' if _has_optim else 'THIẾU'}).")
                if _has_optim:
                    print(f"♻️ [HF-resume] tải checkpoint-* về {training_args.output_dir} ...")
                    snapshot_download(_hf_repo, repo_type="model", token=_hf_tok,
                                      local_dir=training_args.output_dir,
                                      allow_patterns=["checkpoint-*/**"])
                    _resume_dir = os.path.join(training_args.output_dir, _latest)
                    _mp = os.path.join(_resume_dir, "model.safetensors")
                    _corrupt = False
                    if os.path.isfile(_mp):
                        try:
                            _corrupt = any((not torch.isfinite(v).all())
                                           for v in _load_sft(_mp).values())
                        except Exception:
                            _corrupt = False
                    if not os.path.isfile(os.path.join(_resume_dir, "optimizer.pt")):
                        pass
                    elif _corrupt:
                        print(f"🛑 [HF-resume] checkpoint {_latest} chứa trọng số NaN/inf → BỎ resume.")
                    else:
                        training_args.resume_from_checkpoint = _resume_dir
                        print(f"♻️ [HF-resume] RESUME từ {_resume_dir}")
                else:
                    print("⚠️ [HF-resume] checkpoint là weights-only (push cũ) → KHÔNG resume đúng được.")
        except Exception as _e:
            print(f"⚠️ [HF-resume] bỏ qua ({type(_e).__name__}: {_e}); train bình thường.")

    print(">>> Starting Pretrain...")
    if training_args.resume_from_checkpoint:
        from training.metrics import seed_train_metrics_from_checkpoint
        seed_train_metrics_from_checkpoint(training_args.output_dir, training_args.resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        train_result = trainer.train()

    print(">>> Pretrain Finished. Verifying...")
    verify_metrics = trainer.evaluate()

    # Readable debug AFTER training (MOCK always; FULL only if TWC_TRAIN_LOG=1).
    if pretrain_debug:
        _debug_split_ocr(model, data_collator, val_dataset, DEVICE)

    # Save best
    trainer.save_model(training_args.output_dir)
    
    print("✅ Pretrain complete and saved successfully.")

if __name__ == "__main__":
    main()
