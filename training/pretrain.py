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
    total = loss_fn(batch, out)  # stamps loss_mlm / loss_itm / loss_twc into `out`

    tcls = out.get("textcls_scores")
    pcls = out.get("pollutecls_scores")
    cs   = out.get("contrastive_scores")
    o2r  = out.get("o2r_block")
    lm, li, lt = out.get("loss_mlm"), out.get("loss_itm"), out.get("loss_twc")

    # ── DIAGNOSTIC DUMP (always printed, so you can SEE where a problem is) ──
    print(f"\n[diag] batch size = {k}")
    print("[diag] forward tensors:")
    print(f"    MLM  textcls_scores   : {_tensor_health(tcls)}")
    print(f"    ITM  pollutecls_scores: {_tensor_health(pcls)}")
    print(f"    TWC  contrastive_scores: {_tensor_health(cs)}")

    print("[diag] loss components:")
    for nm, lv in [("loss_mlm", lm), ("loss_itm", li), ("loss_twc", lt), ("TOTAL", total)]:
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

    # ── MLM checks ────────────────────────────────────────────────────────
    chk("[MLM] head produced logits", tcls is not None)
    chk("[MLM] has masked positions", int((batch["cmb_text_mask_label"] != -1).sum()) > 0)
    chk("[MLM] loss_mlm finite", lm is not None and bool(torch.isfinite(lm).all()))

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
        if _progress_only_mode():
            # Colab / progress-only: keep the tqdm progress bar clean — do NOT emit
            # the per-step [Pretrain] loss logs (they clutter / break the bar).
            training_args.disable_tqdm = False
            print("ℹ️ [smoke] progress-bar mode (Colab): per-step debug logs disabled.")
        else:
            # Turn ON the built-in per-step per-loss breakdown (Loss M / I / TWC) so a
            # diverging or wrong loss term is immediately visible during the loop.
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
        config = AutoConfig.from_pretrained(ckpt_to_load, local_files_only=True)
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
            model.load_state_dict(new_state_dict, strict=False)

    # Apply config overrides
    mode = model_args.loss_ablation_mode
    use_twc = mode in ["all", "only_twc_ocr_aug"]
    use_ocr_aug = mode in ["all", "only_twc_ocr_aug"]
    
    model.pretrain = True
    model.config.pretrain = True
    model.config.pretrain_ablation_mode = mode
    model.config.use_twc = use_twc
    model.config.use_ocr_aug_finetune = use_ocr_aug
    
    model.to(DEVICE)
    
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

    # Fast method-correctness gate: in smoke/mock mode, verify a single batch
    # exercises MLM + ITM + TWC correctly before spending time on the loop.
    if training_args.smoke_test:
        _verify_pretrain_batch(
            model, data_collator, train_dataset,
            pretrain_loss_fn, pretrain_acc_fn, DEVICE, use_twc,
        )

    print(">>> Starting Pretrain...")
    if training_args.resume_from_checkpoint:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        train_result = trainer.train()

    print(">>> Pretrain Finished. Verifying...")
    verify_metrics = trainer.evaluate()
    
    # Save best
    trainer.save_model(training_args.output_dir)
    
    print("✅ Pretrain complete and saved successfully.")

if __name__ == "__main__":
    main()
