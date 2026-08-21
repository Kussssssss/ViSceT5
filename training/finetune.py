import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

from configs.arguments import ModelArguments, DataArguments, CustomTrainingArguments
from configs.model_config import OpenViVQAConfig
from configs.ocr_config import DEFAULT_OCR_CONFIG
from models.openvivqa_model import OpenViVQAModel
from models.modules.ocr_encoder_feature import Vision_Encode_Ocr_Feature
from data.dataset_hub import DatasetHubLoader
from data.dataset import ViT5VQADataset
from data.collator import ViT5VQADataCollator
from training.metrics import TaskSpecificTrainer, build_compute_metrics_finetune
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

def _patch_torch_load_weights_only():
    """PyTorch >= 2.6 đổi mặc định torch.load(weights_only=True). Transformers 4.45 gọi
    torch.load(rng_file/optimizer.pt/scheduler.pt) KHÔNG truyền weights_only, nên resume
    trên torch 2.6 (Kaggle) nổ ở rng_state.pth (chứa RNG numpy):
      UnpicklingError: Unsupported global: numpy._core.multiarray._reconstruct
    Checkpoint là do CHÍNH ta tạo (tin cậy) → khôi phục hành vi cũ weights_only=False.
    Dùng setdefault để lời gọi nào chủ động đặt weights_only vẫn được tôn trọng."""
    import torch as _torch
    if getattr(_torch.load, "_vsce_patched", False):
        return
    _orig = _torch.load
    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)
    _load._vsce_patched = True
    _torch.load = _load
    print("🩹 [compat] torch.load → weights_only=False (resume trên PyTorch ≥2.6).")


def main(args_list=None):
    _patch_torch_load_weights_only()
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    default_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "finetune.yaml"))
    model_args, data_args, training_args = parse_args_with_yaml_and_cli(parser, args_list, default_yaml)
    
    set_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)

    # ── CHẨN ĐOÁN (ANOMALY=1): bật autograd anomaly detection ──────────────────
    # Dùng khi forward FINITE nhưng backward sinh NaN (vd GradGuard báo grad non-finite
    # mỗi step). PyTorch sẽ RAISE ngay tại op tạo NaN trong backward, kèm traceback trỏ
    # đúng dòng forward đã tạo ra op đó → biết CHÍNH XÁC chỗ hỏng thay vì phải đoán.
    # RẤT CHẬM (~2-5x) → chỉ chạy vài chục step để lấy traceback rồi tắt. Không đổi
    # hành vi train khi không bật.
    if os.environ.get("ANOMALY", "0").lower() in ("1", "true", "yes", "on"):
        torch.autograd.set_detect_anomaly(True)
        print("🔬 [finetune] ANOMALY=1 → autograd anomaly detection BẬT (chậm; chỉ để chẩn đoán). "
              "Chạy tới khi raise rồi gửi traceback.")

    # ── Chế độ TÁI LẬP (opt-in, mặc định TẮT để giữ tốc độ + parity notebook) ──
    # Bật bằng DETERMINISTIC=1: khử phần lớn nhiễu run-to-run trên CÙNG GPU
    # (cudnn deterministic, thuật toán xác định, dataloader 1 worker). Lưu ý: KHÔNG
    # đảm bảo giống hệt khi ĐỔI GPU/CUDA khác; và notebook cũng phải bật y hệt mới so được.
    if os.environ.get("DETERMINISTIC", "0").lower() in ("1", "true", "yes", "on"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as _e:
            print(f"ℹ️ [DETERMINISTIC] use_deterministic_algorithms lỗi ({_e}); vẫn bật cudnn.deterministic.")
        training_args.dataloader_num_workers = 0
        print("🔁 [DETERMINISTIC] bật tái lập: cudnn.deterministic=True, num_workers=0, "
              "CUBLAS_WORKSPACE_CONFIG=:4096:8 (chậm hơn; chỉ giống nhau trên CÙNG GPU).")


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Prepare Data
    print(f">>> Preparing Dataset: {data_args.dataset_name}")
    from configs.base_config import OUTPUT_PATH
    _ds = data_args.dataset_name
    # DATASET-AWARE cache: tên file mang tên dataset để đổi dataset KHÔNG bị cache cũ
    # (vd merged_train.csv của ViTextVQA) che mất → tránh âm thầm train nhầm dữ liệu.
    train_csv = os.path.join(OUTPUT_PATH, f"merged_train_{_ds}.csv")
    val_csv = os.path.join(OUTPUT_PATH, f"merged_val_{_ds}.csv")
    # Back-compat: cache cũ KHÔNG hậu tố là của ViTextVQA — chỉ dùng cho đúng nó.
    _legacy_tr = os.path.join(OUTPUT_PATH, "merged_train.csv")
    _legacy_va = os.path.join(OUTPUT_PATH, "merged_val.csv")
    if (not (os.path.exists(train_csv) and os.path.exists(val_csv))
            and _ds == "ViTextVQA"
            and os.path.exists(_legacy_tr) and os.path.exists(_legacy_va)):
        train_csv, val_csv = _legacy_tr, _legacy_va

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        print(f"ℹ️ Found prepared CSV cache for {_ds}: {os.path.basename(train_csv)}. Loading directly...")
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
            return
        # Ghi cache dataset-aware để lần chạy sau nạp thẳng (không prepare lại).
        try:
            train_df.to_csv(os.path.join(OUTPUT_PATH, f"merged_train_{_ds}.csv"), index=False)
            val_df.to_csv(os.path.join(OUTPUT_PATH, f"merged_val_{_ds}.csv"), index=False)
            print(f"💾 Đã lưu cache: merged_train_{_ds}.csv / merged_val_{_ds}.csv")
        except Exception as _e:
            print(f"ℹ️ Không ghi được cache CSV ({_e}); không sao, sẽ prepare lại lần sau.")

    if training_args.smoke_test:
        # Mặc định GIỮ NGUYÊN 8/4/3 (kiểm tra pipeline siêu nhanh). Có thể phóng to qua env
        # để TÁI HIỆN lỗi chỉ xuất hiện sau vài trăm step (vd phân kỳ dần), mà không phải
        # chạy full 21975 step:  SMOKE_TRAIN_SAMPLES=2048 SMOKE_MAX_STEPS=400
        _s_tr = int(os.environ.get("SMOKE_TRAIN_SAMPLES", "8"))
        _s_ev = int(os.environ.get("SMOKE_EVAL_SAMPLES", "4"))
        _s_st = int(os.environ.get("SMOKE_MAX_STEPS", "3"))
        print(f"🚨 RUNNING IN SMOKE TEST MODE: train={_s_tr} eval={_s_ev} max_steps={_s_st}.")
        train_df = train_df.head(_s_tr)
        val_df = val_df.head(_s_ev)
        training_args.max_steps = _s_st
        training_args.num_train_epochs = 1.0
        training_args.logging_steps = max(1, _s_st // 20)
        training_args.eval_steps = max(1, _s_st)
        training_args.save_steps = max(1, _s_st)

    train_dataset = ViT5VQADataset(train_df)
    val_dataset = ViT5VQADataset(val_df)

    # 2. OCR Encoder
    ocr_config = DEFAULT_OCR_CONFIG
    vision_ocr = Vision_Encode_Ocr_Feature(ocr_config)
    
    # 3. Handle Weights Downloads
    if training_args.pretrain_weights_id:
        PRETRAIN_CKPT_DIR = os.path.join(training_args.output_dir, "pretrain_ckpt_base")
        download_and_extract_checkpoint(training_args.pretrain_weights_id, PRETRAIN_CKPT_DIR)
        model_args.model_name_or_path = PRETRAIN_CKPT_DIR

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
        # Warm-start từ checkpoint PRETRAIN: trọng số vision được train DƯỚI clamp
        # ±1e4 (v4) — finetune không clamp sẽ nổ gradient (đo được: grad_norm=inf
        # mọi step ở v4-A, trong khi v3-A 0/447 inf). Bật cờ cho checkpoint pretrain
        # cũ chưa mang sẵn clamp_vision trong config; scratch (không ckpt) không đổi.
        if bool(getattr(config, "pretrain", False)) and not hasattr(config, "clamp_vision"):
            config.clamp_vision = True
            print("🛡️ [finetune] warm-start từ pretrain-ckpt → bật config.clamp_vision "
                  "(giữ đúng hàm forward mà trọng số đã học)")
    else:
        tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        config = OpenViVQAConfig()

    # FINETUNE: các submodule PRETRAIN-ONLY (ITC heads) được gate theo config.pretrain
    # ngay trong __init__. Ép False TRƯỚC khi dựng model để finetune KHÔNG bao giờ tạo
    # ITC → khởi tạo from-scratch khớp notebook (không ITC), và ckpt pretrain nếu có
    # key itc_* thì chỉ là 'unexpected' (bỏ qua an toàn khi load strict=False).
    # Việc phát hiện clamp_vision cho warm-start ở trên đã đọc cờ pretrain gốc của ckpt.
    config.pretrain = False

    # Cờ ablation phải nằm trong config TRƯỚC khi dựng model. __init__ đọc
    # ablation_use_vs để quyết định có giữ visual_search (ConvNeXt, 27.87M) hay gỡ đi;
    # gán sau khi dựng thì forward vẫn tắt AVF đúng, nhưng module đã nằm sẵn trong
    # state_dict và optimizer — tức phần tối ưu không hề có tác dụng. OpenViVQAConfig
    # không khai báo sẵn các cờ này nên getattr mặc định True, càng phải set tường minh.
    config.pretrain_ablation_mode = model_args.loss_ablation_mode
    config.ablation_use_qaclip = bool(model_args.ablation_use_qaclip)
    config.ablation_use_vs = bool(model_args.ablation_use_vs)
    config.ablation_use_ocr = bool(model_args.ablation_use_ocr)
    config.ablation_use_ocr_aug = bool(model_args.ablation_use_ocr_aug)
    config.use_twc = False  # TWC is disabled during finetuning
    config.use_ocr_aug_finetune = bool(model_args.ablation_use_ocr_aug)

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
            # Tóm tắt số tensor ĐÃ NẠP theo module (để thấy rõ decoder/encoder/qa_clip...)
            from collections import Counter as _Counter
            def _grp(k):
                if k.startswith("vit5.decoder"): return "vit5.decoder"
                if k.startswith("vit5.encoder"): return "vit5.encoder"
                return k.split(".")[0]
            _by = _Counter(_grp(k) for k in new_state_dict)
            print("   ✔️ loaded by module:", dict(sorted(_by.items(), key=lambda x: -x[1])))
            if res.missing_keys:
                print("   ⚠️ missing (not in ckpt) e.g.:", res.missing_keys[:6])
            if res.unexpected_keys:
                print("   ⚠️ unexpected (in ckpt, not in model) e.g.:", res.unexpected_keys[:6])
            if n_loaded < 0.5 * n_model:
                print("🚨 CẢNH BÁO: <50% tensor được nạp — trọng số pretrain gần như KHÔNG chuyển sang! "
                      "Kiểm tra tên key / kiến trúc.")

    # Apply Finetune config overrides
    mode = model_args.loss_ablation_mode
    # ABLATION #4 (finetune): OCR augmentation (correct/noise/keep → related-OCR),
    # ĐỘC LẬP với 3 ablation kiến trúc. True = nối thêm bản OCR đã augment cạnh OCR
    # gốc (gấp đôi token OCR đưa vào Consformer); False = chỉ OCR gốc.
    use_ocr_aug = bool(model_args.ablation_use_ocr_aug)

    # Các cờ đã được set trên `config` TRƯỚC khi dựng model (xem phía trên); model.config
    # chính là object đó nên ở đây chỉ còn xác nhận và in ra.
    model.pretrain = False
    model.config.pretrain = False
    print(f">>> [finetune] ABLATION: qaclip={model.config.ablation_use_qaclip} | "
          f"vs={model.config.ablation_use_vs} | ocr={model.config.ablation_use_ocr} | "
          f"ocr_aug={model.config.ablation_use_ocr_aug}")
    print(f">>> [finetune] visual_search (ConvNeXt) "
          f"{'CÓ dựng' if hasattr(model, 'visual_search') else 'ĐÃ GỠ (AVF tắt)'} | "
          f"tổng tham số {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    model.to(DEVICE)
    
    if hasattr(model, "visual_search") and hasattr(model.visual_search, "vit_processor"):
        model.visual_search.vit_processor = model.image_processor

    gen_max_new = int(getattr(model.config, "generation_max_new_tokens", 56))
    gen_num_beams = int(getattr(model.config, "generation_num_beams", 4))
    model.generation_config = GenerationConfig(
        max_new_tokens=gen_max_new,
        num_beams=gen_num_beams,
        do_sample=False,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # 5. Loss, Metrics, Collator
    data_collator = ViT5VQADataCollator(
        tokenizer=tokenizer,
        image_processor=model.image_processor,
        ocr_encoder=vision_ocr,
        config=model.config,
        term_vocab_path=data_args.vocab_path,
        viet_vocab_path=data_args.viet_vocab_path,
        eng_vocab_path="",
        dataframe=train_df,
        pretrain=False,
        debug=False,
    )
    if hasattr(data_collator, "set_mode"):
        data_collator.set_mode(pretrain=False, mask_prob=0.0)
    data_collator.pretrain_ablation_mode = mode
    data_collator.use_ocr_aug_finetune = use_ocr_aug
        
    compute_metrics_fn = build_compute_metrics_finetune(tokenizer)

    # ── CHỌN CHECKPOINT TỐT NHẤT theo metric ĐÚNG VỚI TỪNG DATASET ──
    # ViTextVQA/ViOCRVQA: đáp án là scene-text ngắn → EM là thước đo mục tiêu.
    # OpenViVQA: đáp án tự do, EM rất thấp và không phản ánh chất lượng → dùng CIDEr.
    # Cả EM và CIDEr đều "càng cao càng tốt" (greater_is_better=True), khác với loss cũ.
    # LƯU Ý: CIDEr chỉ được compute_metrics trả về khi có 'java' (PTBTokenizer). Nếu
    # OpenViVQA mà thiếu java → 'eval_cider' vắng → load_best_model_at_end sẽ lỗi; khi đó
    # tự lùi về 'loss' và cảnh báo. Env METRIC_FOR_BEST_MODEL ghi đè nếu cần.
    import shutil as _shutil
    _ds_l = str(data_args.dataset_name).strip().lower()
    _java_ok = _shutil.which("java") is not None
    _env_metric = os.environ.get("METRIC_FOR_BEST_MODEL", "").strip().lower()
    if _env_metric:
        training_args.metric_for_best_model = _env_metric
        training_args.greater_is_better = _env_metric not in ("loss", "eval_loss")
    elif _ds_l == "openvivqa":
        if _java_ok:
            training_args.metric_for_best_model = "cider"
            training_args.greater_is_better = True
        else:
            training_args.metric_for_best_model = "loss"
            training_args.greater_is_better = False
            print("⚠️ [best-metric] OpenViVQA cần CIDEr nhưng KHÔNG thấy 'java' → "
                  "tạm chọn best theo 'loss'. Cài java (openjdk) để dùng đúng CIDEr.")
    else:  # ViTextVQA, ViOCRVQA và mặc định
        training_args.metric_for_best_model = "em"
        training_args.greater_is_better = True
    print(f">>> [best-metric] dataset={data_args.dataset_name} → "
          f"metric_for_best_model='{training_args.metric_for_best_model}' "
          f"greater_is_better={training_args.greater_is_better} (java={'có' if _java_ok else 'không'})")

    # 6. Trainer
    trainer = TaskSpecificTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )

    # ── Upload TỪNG checkpoint NGAY khi lưu (chống mất dữ liệu khi Vast bị reclaim) ──
    # Bài học: run_pipeline chỉ upload SAU khi train xong (bước 3.5); finetune crash
    # hoặc máy interruptible bị chiếm giữa chừng → mất trắng dù đã set HF_REPO. Callback
    # này push checkpoint mới nhất lên HF ngay mỗi lần Trainer save (dùng HfApi.upload_folder
    # thuần API — KHÔNG cần git-lfs). Gated bằng env HF_TOKEN+HF_REPO; không set thì bỏ qua.
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
                # KHÔNG push checkpoint HỎNG (NaN/inf): tránh làm nguồn resume bị nhiễm độc
                # → lần sau auto-resume nạp lại trạng thái hỏng (death-spiral). Bỏ qua bản này.
                _model = kw.get("model")
                if _model is not None:
                    _bad = [n for n, p in _model.named_parameters() if not torch.isfinite(p).all()]
                    if _bad:
                        print(f"🛑 [HF] checkpoint-{state.global_step}: model có {len(_bad)} tensor NaN/inf "
                              f"(vd {_bad[0]}) → BỎ push (không nhiễm nguồn resume). Cần sửa gốc NaN trước.")
                        return
                try:
                    # MẶC ĐỊNH push ĐẦY ĐỦ (gồm optimizer.pt/scheduler.pt/rng_state/
                    # trainer_state.json) để RESUME ĐÚNG được sau khi Colab ngắt.
                    # Đặt HF_PUSH_OPTIM=0 nếu chỉ cần bản inference nhẹ (sẽ KHÔNG resume được).
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
            print(f"☁️ [HF] bật auto-push mỗi checkpoint → {_hf_repo} (chống mất khi reclaim)")
        except Exception as _e:
            print(f"ℹ️ [HF] không bật được auto-push callback ({_e}); vẫn có upload cuối ở run_pipeline.")

    # ── DỪNG SẠCH theo số epoch / thời gian (cho Kaggle 12h) ──
    # num_train_epochs LUÔN = TỔNG (vd 5) để horizon cosine không đổi. Callback này chỉ
    # dừng SỚM session hiện tại ở ĐÚNG ranh giới epoch (sau khi eval+save đã chạy), rồi
    # session sau resume và đi tiếp tới tổng đó — KHÔNG để Kaggle kill giữa chừng.
    #   STOP_AT_EPOCH=k  → dừng khi vừa hoàn tất epoch thứ k (tuyệt đối; tăng dần 2→4→5).
    #   MAX_TRAIN_HOURS=h→ guard thời gian: quá h giờ thì eval+save+dừng (một eval phụ).
    _stop_ep = os.environ.get("STOP_AT_EPOCH", "").strip()
    _max_hr = os.environ.get("MAX_TRAIN_HOURS", "").strip()
    if _stop_ep or _max_hr:
        import time as _time
        from transformers.trainer_callback import TrainerCallback as _TCB
        class _EpochBudgetStop(_TCB):
            def __init__(self, stop_at, max_sec):
                self.stop_at = int(stop_at) if stop_at else None
                self.max_sec = float(max_sec) * 3600 if max_sec else None
                self.t0 = _time.time()
            def on_epoch_end(self, args, state, control, **kw):
                ep = int(round(state.epoch or 0))
                if self.stop_at is not None and ep >= self.stop_at:
                    # save_strategy="epoch" đã bật should_save ở ranh giới này; ép thêm cho
                    # chắc, rồi dừng. Checkpoint được lưu + push TRƯỚC khi vòng lặp thoát.
                    control.should_save = True
                    control.should_training_stop = True
                    print(f"⏹️ [chunk] hoàn tất epoch {ep} ≥ STOP_AT_EPOCH={self.stop_at} "
                          f"→ lưu checkpoint và DỪNG session (resume tiếp ở session sau).")
                return control
            def on_step_end(self, args, state, control, **kw):
                if self.max_sec is not None and (_time.time() - self.t0) > self.max_sec:
                    # Ép eval để khớp load_best_model_at_end, rồi save + dừng giữa epoch.
                    control.should_evaluate = True
                    control.should_save = True
                    control.should_training_stop = True
                    print(f"⏰ [chunk] quá MAX_TRAIN_HOURS → eval+save+DỪNG ở step {state.global_step}.")
                return control
        trainer.add_callback(_EpochBudgetStop(_stop_ep, _max_hr))
        print(f"⏹️ [chunk] dừng sớm: STOP_AT_EPOCH={_stop_ep or '(none)'} "
              f"MAX_TRAIN_HOURS={_max_hr or '(none)'} | tổng epoch giữ nguyên "
              f"{training_args.num_train_epochs}")

    # ── Auto-resume TỪ HF repo (chống mất tiến độ khi Colab ngắt) ──
    # Nếu đã set HF_TOKEN+HF_REPO và repo đã có checkpoint-* ĐẦY ĐỦ (từ lần chạy trước)
    # mà chưa có resume tường minh → TỰ tải các checkpoint về output_dir rồi resume từ
    # cái mới nhất. Chạy lại notebook = train TIẾP từ epoch dở, KHÔNG train lại từ đầu.
    # Tắt bằng RESUME_FROM_HF=0 (vd muốn train mới trên repo cũ). KHÔNG auto-resume trong
    # smoke_test (tránh smoke nạp checkpoint của run thật).
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
                    # CHỈ tải checkpoint MỚI NHẤT, không tải toàn bộ. Repo HF tích luỹ mọi
                    # epoch (~4.5GB/checkpoint); tải hết vào đĩa Kaggle ~20GB sẽ đầy đĩa
                    # ngay khi resume và crash lúc lưu (No space left / unexpected pos).
                    # Resume chỉ cần đúng checkpoint mới nhất.
                    print(f"♻️ [HF-resume] tải RIÊNG {_latest} về {training_args.output_dir} ...")
                    snapshot_download(_hf_repo, repo_type="model", token=_hf_tok,
                                      local_dir=training_args.output_dir,
                                      allow_patterns=[f"{_latest}/**"])
                    _resume_dir = os.path.join(training_args.output_dir, _latest)
                    # KIỂM TRA checkpoint có NaN/inf không → KHÔNG resume vào trạng thái hỏng
                    # (nếu không sẽ nạp lại trọng số NaN mãi mãi — death-spiral).
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
                        print(f"🛑 [HF-resume] checkpoint {_latest} chứa trọng số NaN/inf → BỎ resume "
                              f"(tránh nạp lại trạng thái hỏng). Hãy XOÁ checkpoint hỏng trên HF (hoặc "
                              f"dùng HF_REPO mới / RESUME_FROM_HF=0) rồi train lại từ đầu.")
                    else:
                        training_args.resume_from_checkpoint = _resume_dir
                        print(f"♻️ [HF-resume] RESUME từ {_resume_dir}")
                else:
                    print("⚠️ [HF-resume] checkpoint là weights-only (push cũ) → KHÔNG resume "
                          "đúng được. Từ lần này push sẽ ĐẦY ĐỦ (HF_PUSH_OPTIM=1 mặc định).")
        except Exception as _e:
            print(f"⚠️ [HF-resume] bỏ qua ({type(_e).__name__}: {_e}); train bình thường.")

    print(">>> Starting Finetune...")
    if training_args.resume_from_checkpoint:
        from training.metrics import seed_train_metrics_from_checkpoint
        seed_train_metrics_from_checkpoint(training_args.output_dir, training_args.resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        train_result = trainer.train()

    print(">>> Finetune Finished. Verifying...")
    verify_metrics = trainer.evaluate()
    
    # Save best
    trainer.save_model(training_args.output_dir)
    
    print("✅ Finetune complete and saved successfully.")

if __name__ == "__main__":
    main()
