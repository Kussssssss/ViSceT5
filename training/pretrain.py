import os
import gc
import json
import shutil
import zipfile
import random
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Subset
from safetensors.torch import load_file

from transformers import (
    Seq2SeqTrainingArguments,
    set_seed,
)

# =========================================================
# 1. CONFIGURATION (ĐIỀN THAM SỐ TẠI ĐÂY)
# =========================================================
LOSS_ABLATION_MODE = "all"  # ["only_itm_mlm", "only_twc_ocr_aug", "all"]
ARCH_ABLATION_MODE_RAW = "all"       # Kiến trúc (all, qaclip, qaclip_vs, none)

TARGET_PRETRAIN_EPOCHS = 5
RESUME_PRETRAIN = False

OUTPUT_PATH = "/content/working_dir"
DRIVE_ROOT = "/content/drive/MyDrive/Training_Backup"
SEED = 42

# Tự động mapping cấu hình dựa trên kịch bản Ablation
if LOSS_ABLATION_MODE == "only_itm_mlm":
    PRETRAIN_ABLATION_MODE = "no_twc_ocr_aug"
    USE_TWC = False
    USE_OCR_AUG = False
elif LOSS_ABLATION_MODE == "only_twc_ocr_aug":
    PRETRAIN_ABLATION_MODE = "only_twc_ocr_aug"
    USE_TWC = True
    USE_OCR_AUG = True
else:  # "all"
    PRETRAIN_ABLATION_MODE = "full"
    USE_TWC = True
    USE_OCR_AUG = True

# Tự động sinh tên thư mục để không bị trùng lặp giữa các kịch bản
EXPERIMENT_NAME = f"pretrain_{LOSS_ABLATION_MODE}"
RUN_PRETRAIN_DIR = os.path.join(OUTPUT_PATH, f"run_{EXPERIMENT_NAME}")
LOGS_PRETRAIN_DIR = os.path.join(OUTPUT_PATH, f"logs_{EXPERIMENT_NAME}")
PRETRAIN_CKPT_DIR = os.path.join(OUTPUT_PATH, f"ckpt_{EXPERIMENT_NAME}")

DRIVE_BACKUP_DIR = os.path.join(DRIVE_ROOT, EXPERIMENT_NAME)
DRIVE_RUN_PRETRAIN_DIR = os.path.join(DRIVE_BACKUP_DIR, "run_checkpoints")

for d in [RUN_PRETRAIN_DIR, LOGS_PRETRAIN_DIR, PRETRAIN_CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# 2. UTILS & HELPERS
# =========================================================
if "_original_torch_load" not in globals():
    _original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

def normalize_ablation_mode(mode: str) -> str:
    m = str(mode).lower().strip() if mode else "all"
    mapping = {
        "none": "none", "baseline": "none", "no_module": "none", "nomodule": "none",
        "no_qaclip_fusion": "none", "noqaclipfusion": "none",
        "qaclip": "qaclip", "only_qaclip": "qaclip", "no_vs": "qaclip", "novs": "qaclip",
        "qaclip_vs": "qaclip_vs", "qaclip+vs": "qaclip_vs", "qaclipvs": "qaclip_vs",
        "no_ocr": "qaclip_vs", "noocr": "qaclip_vs",
        "all": "all", "full": "all",
    }
    out = mapping.get(m, m)
    if out not in ("none", "qaclip", "qaclip_vs", "all"):
        raise ValueError(f"Unsupported architecture ablation mode: {mode}")
    return out

ARCH_ABLATION_MODE = normalize_ablation_mode(ARCH_ABLATION_MODE_RAW)

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def unzip_folder(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_to)

def drive_available():
    return os.path.isdir("/content/drive/MyDrive")

def get_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir): return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts: return None
    return os.path.join(output_dir, sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1])

def safe_backup_checkpoint(local_ckpt_dir, drive_backup_dir, ckpt_name):
    if not os.path.exists(local_ckpt_dir): return
    os.makedirs(drive_backup_dir, exist_ok=True)
    zip_path = os.path.join(OUTPUT_PATH, f"{ckpt_name}.zip")
    drive_zip = os.path.join(drive_backup_dir, f"{ckpt_name}.zip")
    zip_folder(local_ckpt_dir, zip_path)
    shutil.copy2(zip_path, drive_zip)
    os.remove(zip_path)

def safe_restore_checkpoint(drive_backup_dir, local_run_dir):
    if not os.path.exists(drive_backup_dir): return
    zips = [f for f in os.listdir(drive_backup_dir) if f.startswith("checkpoint-") and f.endswith(".zip")]
    if not zips: return
    latest_zip = sorted(zips, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
    drive_zip_path = os.path.join(drive_backup_dir, latest_zip)
    extract_path = os.path.join(local_run_dir, latest_zip.replace(".zip", ""))
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        unzip_folder(drive_zip_path, extract_path)


# =========================================================
# 3. PREPARE ENVIRONMENT & MODEL CONFIG
# =========================================================
torch.cuda.empty_cache()
gc.collect()

if RESUME_PRETRAIN and drive_available():
    print(">>> Checking Drive backup for resume...")
    safe_restore_checkpoint(DRIVE_RUN_PRETRAIN_DIR, RUN_PRETRAIN_DIR)
    final_zip = os.path.join(DRIVE_BACKUP_DIR, f"{EXPERIMENT_NAME}_final.zip")
    if os.path.exists(final_zip) and not os.listdir(PRETRAIN_CKPT_DIR):
        unzip_folder(final_zip, PRETRAIN_CKPT_DIR)

# Apply configs to Model (Giả định `model` đã được khởi tạo trước đó)
model.pretrain = True
model.config.pretrain = True
model.config.ablation_mode = ARCH_ABLATION_MODE
model.config.pretrain_ablation_mode = PRETRAIN_ABLATION_MODE
model.config.use_twc = USE_TWC
model.config.use_ocr_aug_finetune = USE_OCR_AUG  # Dùng chung cờ cho pretrain/finetune ocr aug

print("===== PRETRAIN CONFIG =====")
print(f"LOSS_ABLATION_MODE : {LOSS_ABLATION_MODE}")
print(f"arch_ablation_mode : {model.config.ablation_mode}")
print(f"pretrain_ablation  : {model.config.pretrain_ablation_mode}")
print(f"use_twc            : {model.config.use_twc}")
print(f"use_ocr_aug        : {model.config.use_ocr_aug_finetune}")
print("===========================\n")


# =========================================================
# 4. INIT LOSS, METRICS, COLLATOR
# =========================================================
pretrain_loss_fn = ViT5PretrainLoss(pretrain_ablation_mode=PRETRAIN_ABLATION_MODE)
pretrain_acc_fn = GlobalPretrainAccuracy(mode=LOSS_ABLATION_MODE)

data_collator_pretrain = ViT5VQADataCollator(
    tokenizer=tokenizer,
    image_processor=model.image_processor,
    ocr_encoder=vision_ocr,
    config=model.config,
    term_vocab_path=VOCAB_PATH,
    viet_vocab_path=VIET_VOCAB_PATH,
    eng_vocab_path="",
    dataframe=train_df,
    pretrain=True,
    debug=False,
)

if hasattr(data_collator_pretrain, "set_mode"):
    data_collator_pretrain.set_mode(pretrain=True, mask_prob=0.15)
data_collator_pretrain.pretrain_ablation_mode = PRETRAIN_ABLATION_MODE
data_collator_pretrain.use_ocr_aug_pretrain = USE_OCR_AUG


# =========================================================
# 5. SANITY CHECK
# =========================================================
_debug_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=False, collate_fn=data_collator_pretrain
)
_debug_batch = next(iter(_debug_loader))

print("===== BATCH SANITY CHECK =====")
print("mlm_input_ids:", _debug_batch["mlm_input_ids"].shape)
print("o2r_labels:", _debug_batch.get("o2r_labels", None) is not None)
print("r2o_labels:", _debug_batch.get("r2o_labels", None) is not None)
print("==============================\n")
del _debug_loader, _debug_batch; gc.collect()


# =========================================================
# 6. TRAINING
# =========================================================
pretrain_args = Seq2SeqTrainingArguments(
    output_dir=RUN_PRETRAIN_DIR,
    logging_dir=LOGS_PRETRAIN_DIR,
    seed=SEED, data_seed=SEED,
    num_train_epochs=TARGET_PRETRAIN_EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4, weight_decay=0.01, warmup_ratio=0.1,
    lr_scheduler_type="cosine", max_grad_norm=1.0,
    logging_steps=10,
    eval_strategy="steps", eval_steps=1050,
    save_strategy="steps", save_steps=1050, save_total_limit=3,
    metric_for_best_model="pretrain_acc", greater_is_better=True, load_best_model_at_end=True,
    dataloader_num_workers=4, dataloader_pin_memory=True,
    remove_unused_columns=False, predict_with_generate=False,
    optim="adamw_torch", disable_tqdm=False, fp16=False,
)

pretrain_trainer = TaskSpecificTrainer(
    model=model.to(DEVICE),
    args=pretrain_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator_pretrain,
    tokenizer=tokenizer,
    compute_metrics=simple_pretrain_aggregator,
)

def continue_pretrain(trainer, target_epochs, resume=True):
    trainer.args.num_train_epochs = target_epochs
    latest_ckpt = get_latest_checkpoint(trainer.args.output_dir) if resume else None
    if latest_ckpt:
        print(f">>> Resuming from checkpoint: {latest_ckpt}")
        return trainer.train(resume_from_checkpoint=latest_ckpt)
    print(">>> Starting pretrain from scratch/current weights")
    return trainer.train()

train_result = continue_pretrain(pretrain_trainer, TARGET_PRETRAIN_EPOCHS, RESUME_PRETRAIN)

print("\n" + "=" * 50)
print(">>> PRETRAIN FINISHED. VERIFYING & SAVING BEST MODEL...")
print("=" * 50)


# =========================================================
# 7. VERIFY & SAVE
# =========================================================
best_ckpt_path = pretrain_trainer.state.best_model_checkpoint
recorded_best_metric = pretrain_trainer.state.best_metric
verify_metrics = pretrain_trainer.evaluate()
verify_acc = verify_metrics.get("eval_pretrain_acc", -1.0)

print(f"\n📊 VERIFICATION: Recorded = {recorded_best_metric} | Verified = {verify_acc}")

# Lưu trạng thái pretrain config gốc vào file config (không ép cứng False như trước)
model.config.pretrain_ablation_mode = PRETRAIN_ABLATION_MODE
model.config.ablation_mode = ARCH_ABLATION_MODE
model.config.use_twc = USE_TWC
model.config.use_ocr_aug_finetune = USE_OCR_AUG

model.save_pretrained(PRETRAIN_CKPT_DIR)
tokenizer.save_pretrained(PRETRAIN_CKPT_DIR)
if hasattr(model, "image_processor") and model.image_processor:
    model.image_processor.save_pretrained(os.path.join(PRETRAIN_CKPT_DIR, "image_processor"))

info_payload = {
    "ablation_mode": ARCH_ABLATION_MODE,
    "pretrain_ablation_mode": PRETRAIN_ABLATION_MODE,
    "use_twc": USE_TWC,
    "use_ocr_aug": USE_OCR_AUG,
    "best_checkpoint_source": best_ckpt_path,
    "recorded_best_acc": recorded_best_metric,
    "verified_best_acc": verify_acc,
}
with open(os.path.join(PRETRAIN_CKPT_DIR, "eval_metrics_verified.json"), "w") as f:
    json.dump(verify_metrics, f, indent=2)
with open(os.path.join(PRETRAIN_CKPT_DIR, "training_info.json"), "w") as f:
    json.dump(info_payload, f, indent=2)

if "get_model_fingerprint" in globals():
    with open(os.path.join(PRETRAIN_CKPT_DIR, "fingerprint_pretrain.json"), "w") as f:
        json.dump(get_model_fingerprint(pretrain_trainer.model), f, indent=2)

# BACKUP
if drive_available():
    print(">>> Backing up Pretrain to Drive...")
    safe_backup_checkpoint(PRETRAIN_CKPT_DIR, DRIVE_BACKUP_DIR, f"{EXPERIMENT_NAME}_final")
    last_run_ckpt = get_latest_checkpoint(RUN_PRETRAIN_DIR)
    if last_run_ckpt:
        safe_backup_checkpoint(last_run_ckpt, DRIVE_RUN_PRETRAIN_DIR, os.path.basename(last_run_ckpt))
    print(">>> Backup Done.")