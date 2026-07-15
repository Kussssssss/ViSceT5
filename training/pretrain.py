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
    # Grounded-cloze decoder loss via the model's existing finetune path (pretrain
    # flipped off) — same mechanism the trainer uses; keeps the objective in the
    # pretrain method, not in models/. Only fires when the collator emitted gen targets.
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
    total = loss_fn(batch, out)  # stamps loss_mlm / loss_itm / loss_twc / loss_gen into `out`

    tcls = out.get("textcls_scores")
    pcls = out.get("pollutecls_scores")
    cs   = out.get("contrastive_scores")
    o2r  = out.get("o2r_block")
    lm, li, lt = out.get("loss_mlm"), out.get("loss_itm"), out.get("loss_twc")
    lg = out.get("loss_gen")
    gen_on = out.get("gen_loss") is not None

    # ── DIAGNOSTIC DUMP (always printed, so you can SEE where a problem is) ──
    print(f"\n[diag] batch size = {k}")
    print("[diag] forward tensors:")
    print(f"    MLM  textcls_scores   : {_tensor_health(tcls)}")
    print(f"    ITM  pollutecls_scores: {_tensor_health(pcls)}")
    print(f"    TWC  contrastive_scores: {_tensor_health(cs)}")

    print("[diag] loss components:")
    for nm, lv in [("loss_mlm", lm), ("loss_itm", li), ("loss_twc", lt),
                   ("loss_gen", lg), ("TOTAL", total)]:
        if lv is None:
            print(f"    {nm:9s}: None")
        else:
            finite = bool(torch.isfinite(lv).all())
            print(f"    {nm:9s}: {lv.item():.5f}  finite={finite}")

    # MLM detail
    if tcls is not None:
        tgt = batch["cmb_text_mask_label"]
        msk = tgt != -1
        n_masked = int(msk.sum().item())
        if n_masked > 0:
            mlm_acc = (tcls.argmax(-1)[msk] == tgt[msk]).float().mean().item()
            print(f"[diag] MLM: masked_positions={n_masked}, batch_token_acc={mlm_acc:.3f}")
        else:
            print("[diag] MLM: NO masked positions in this batch!")

    # ITM detail
    tp = batch["tag_pollute"]
    print(f"[diag] ITM: tag_pollute dist -> polluted={int((tp==1).sum())}, clean={int((tp==0).sum())}")

    # TWC detail (label distribution + positive/negative logit separation)
    if use_twc and cs is not None and o2r is not None:
        n_pos = int((o2r > 0.5).sum()); n_semi = int(((o2r > 0.0) & (o2r <= 0.5)).sum())
        n_neg = int((o2r == 0.0).sum()); n_ign = int((o2r == -1).sum())
        print(f"[diag] TWC labels: positives(>0.5)={n_pos}, semi(0–0.5]={n_semi}, "
              f"negatives(==0)={n_neg}, ignored(-1)={n_ign}")
        pos_l = cs[o2r > 0.5]; neg_l = cs[o2r == 0.0]
        pm = pos_l.mean().item() if pos_l.numel() else float("nan")
        nm = neg_l.mean().item() if neg_l.numel() else float("nan")
        print(f"[diag] TWC logits: mean(positive)={pm:.3f} vs mean(negative)={nm:.3f} "
              f"(positive should trend higher as training proceeds)")
        ls = getattr(model, "logit_scale", None)
        if ls is not None:
            print(f"[diag] TWC logit_scale(raw)={ls.item():.3f}, exp≈{float(torch.exp(ls.detach())):.2f}")

    checks = []
    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ── MLM checks (only when encoder-head MLM is active; DROPPED in cloze modes) ──
    if not gen_on:
        chk("[MLM] head produced logits", tcls is not None)
        chk("[MLM] has masked positions", int((batch["cmb_text_mask_label"] != -1).sum()) > 0)
        chk("[MLM] loss_mlm finite", lm is not None and bool(torch.isfinite(lm).all()))
    else:
        print("  ℹ️ [MLM] encoder-head MLM dropped in cloze mode — masked-prediction "
              "is done by the decoder (grounded-cloze / span-infill).")

    # ── ITM checks ────────────────────────────────────────────────────────
    chk("[ITM] head produced logits", pcls is not None)
    if not (int((tp == 1).sum()) > 0 and int((tp == 0).sum()) > 0):
        # Non-fatal: pollution is random per batch; a one-sided draw is not a bug.
        print("  ⚠️ [ITM] this batch is one-sided (all polluted or all clean) — "
              "not a code error, just an unlucky random draw.")
    chk("[ITM] loss_itm finite", li is not None and bool(torch.isfinite(li).all()))

    # ── TWC checks ────────────────────────────────────────────────────────
    if use_twc:
        chk("[TWC] contrastive_scores produced", cs is not None,
            f"shape={tuple(cs.shape)}" if cs is not None else "MISSING => TWC branch was skipped!")
        chk("[TWC] o2r_block produced", o2r is not None)
        if cs is not None and o2r is not None:
            chk("[TWC] score/label shapes match", tuple(cs.shape) == tuple(o2r.shape),
                f"scores={tuple(cs.shape)} labels={tuple(o2r.shape)}")
            chk("[TWC] scores finite", bool(torch.isfinite(cs).all()))
            chk("[TWC] labels have positives (>0.5)", bool((o2r > 0.5).any()))
            chk("[TWC] labels have negatives (==0)", bool((o2r == 0.0).any()))
        chk("[TWC] loss_twc finite & > 0", lt is not None and bool(torch.isfinite(lt).all()) and lt.item() > 0)

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


def _debug_gen_cloze(model, data_collator, dataset, device, n_show=5):
    """Readable grounded-cloze debug: show the MASKED question (encoder input), the
    CLEAN target words, and the model output — i.e. whether the decoder recovers the
    OCR-overlapping words by reading the OCR feature branch."""
    print("\n" + "=" * 70)
    print("🔎 [MLM DEBUG] decoder span-infill (MLM): masked question → predicted word (target vs output)")
    print("=" * 70)
    k = min(8, len(dataset))
    if k < 2:
        print("  skip"); return
    tok = data_collator.tokenizer
    batch = data_collator([dataset[i] for i in range(k)])
    batch = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in batch.items()}
    if batch.get("gen_labels") is None or batch.get("gen_input_ids") is None:
        print("  (no gen targets in batch)"); return
    model.eval()
    orig = getattr(model, "pretrain", False)
    model.pretrain = False
    try:
        with torch.no_grad():
            out = model(
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
        model.pretrain = orig
    logits = out.get("logits")
    if logits is None:
        print("  (no logits)"); model.train(); return
    pred = logits.argmax(-1)
    labels = batch["gen_labels"]
    gen_in = batch["gen_input_ids"]
    print("  (masked words removed from the question; decoder must read them from OCR features)")
    shown = 0
    for i in range(labels.size(0)):
        pos = [p for p, t in enumerate(labels[i].tolist()) if t != -100]
        if not pos:
            continue  # no cloze span for this sample
        masked_q = tok.decode([int(t) for t in gen_in[i].tolist() if int(t) != data_collator.pad_id],
                              skip_special_tokens=False).strip()
        gold = tok.decode([int(labels[i][p]) for p in pos], skip_special_tokens=True).strip()
        prd = tok.decode([int(pred[i][p].item()) for p in pos], skip_special_tokens=True).strip()
        print(f"  [sample {i}] masked Q: {masked_q[:100]}")
        print(f"              target  : {gold[:80]}")
        print(f"              output  : {prd[:80]}")
        shown += 1
        if shown >= n_show:
            break
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
        config_path = f"configs/data/{data_args.dataset_name}.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                ds_cfg = yaml.safe_load(f)
            hub.register_dataset(
                dataset_name=ds_cfg['dataset_name'],
                task_type="VQA",
                image_zip_id=ds_cfg['image'].get('drive_id'),
                image_dir_override='',
                ocr_zip_id=ds_cfg['ocr'].get('drive_id'),
                ocr_dir_override='',
                splits={
                    "train":      {"id": ds_cfg['dataset']['train'].get('drive_id') or ds_cfg['dataset']['train'].get('dir'), "url": None},
                    "validation": {"id": ds_cfg['dataset']['validation'].get('drive_id') or ds_cfg['dataset']['validation'].get('dir'), "url": None},
                    "test":       {"id": ds_cfg['dataset']['test'].get('drive_id') or ds_cfg['dataset']['test'].get('dir'), "url": None},
                }
            )
            hub.prepare(data_args.dataset_name)
        else:
            print(f"⚠️ Dataset config not found at {config_path}. Assuming it's already registered or manually ready.")
        
        try:
            dfs = hub.load_task(data_args.dataset_name)
            train_df = dfs["train"]
            val_df = dfs["validation"]
        except Exception as e:
            print(f"❌ Failed to load dataset {data_args.dataset_name} from Hub: {e}")
            print("Please run scripts/prepare_dataset.py first.")
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
    print(f">>> [pretrain] hard-knobs: adv_prob={data_collator.adv_probability_pretrain} "
          f"twc_dup_box={getattr(data_collator,'twc_dup_box',True)} "
          f"mlm_rand_prob={getattr(data_collator,'mlm_rand_prob',0.15)} "
          f"itm_weight={os.environ.get('ITM_WEIGHT','0')}")

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
    # In cloze modes the decoder span-infill IS the masked-prediction; encoder-head
    # MLM is dropped, so only the cloze debug is meaningful.
    if pretrain_debug:
        if _cloze:
            _debug_gen_cloze(model, data_collator, val_dataset, DEVICE)
        else:
            _debug_mlm_predictions(model, data_collator, val_dataset, DEVICE)

    # OCR-ablation grounding evidence — run in cloze modes even on the FULL run (cheap,
    # one-time, ~12 batches) since it is the key proof that grounded acc comes from
    # reading OCR. Disable with env OCR_ABLATION=0.
    if _cloze and os.environ.get("OCR_ABLATION", "1") not in ("0", "false", "False"):
        try:
            _debug_ocr_ablation(model, data_collator, val_dataset, DEVICE)
        except Exception as e:
            print(f"⚠️ [OCR-ABLATION] skipped: {type(e).__name__}: {e}")

    # Save best
    trainer.save_model(training_args.output_dir)
    
    print("✅ Pretrain complete and saved successfully.")

if __name__ == "__main__":
    main()
