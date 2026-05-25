"""
training/finetune.py
Full finetune training loop.
"""

import os
import shutil
import json
import torch
import gc
import zipfile
import gdown
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    GenerationConfig,
    Seq2SeqTrainingArguments,
    set_seed,
)

if "_original_torch_load" not in globals():
    _original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

# =====================================================================
# BẢNG ĐIỀU KHIỂN ABLATION
# =====================================================================
USE_QACLIP = True  # True: Dùng Late Fusion QACLIP | False: Dùng CLIP thường
USE_VS     = False # True: Dùng Visual Search crop | False: Dùng Center Crop
USE_OCR    = False # True: Dùng Consformer       | False: Dùng Linear/MLP Baseline

def get_ablation_signature(qa: bool, vs: bool, ocr: bool) -> str:
    qa_str = "T" if qa else "F"
    vs_str = "T" if vs else "F"
    ocr_str = "T" if ocr else "F"
    return f"qa_{qa_str}_vs_{vs_str}_ocr_{ocr_str}"

CURRENT_ABLATION_SIGNATURE = get_ablation_signature(USE_QACLIP, USE_VS, USE_OCR)

# =====================================================================
# CẤU HÌNH DRIVE ID VÀ THÔNG SỐ TRAINING
# =====================================================================
OUTPUT_PATH = "/kaggle/working"

# 1. Nếu có trọng số Pretrain (vạch xuất phát), điền ID vào đây
PRETRAIN_WEIGHTS_DRIVE_ID = "" 

# 2. Cấu hình Resume (Học tiếp)
RESUME_FINETUNE = False
# Nếu checkpoint nằm trên Drive, điền ID file zip (last_checkpoint.zip) vào đây
RESUME_DRIVE_ID = "" 

PRETRAIN_CKPT_DIR = os.path.join(OUTPUT_PATH, "pretrain_ckpt")
RUN_DIR = os.path.join(OUTPUT_PATH, f"run_finetune_{CURRENT_ABLATION_SIGNATURE}")
LOGS_DIR = os.path.join(OUTPUT_PATH, f"logs_{CURRENT_ABLATION_SIGNATURE}")
BEST_DIR = os.path.join(OUTPUT_PATH, f"best_bundle_{CURRENT_ABLATION_SIGNATURE}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TARGET_FINETUNE_EPOCHS = 1

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(PRETRAIN_CKPT_DIR, exist_ok=True)

set_seed(SEED)
torch.cuda.empty_cache()
gc.collect()

# --- Các hàm tiện ích ---
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

def download_and_extract_checkpoint(drive_id, extract_to):
    if not drive_id: return False
    try:
        output_zip = os.path.join(OUTPUT_PATH, "downloaded_ckpt.zip")
        gdown.download(id=drive_id, output=output_zip, quiet=False)
        if os.path.exists(output_zip):
            os.makedirs(extract_to, exist_ok=True)
            unzip_folder(output_zip, extract_to)
            os.remove(output_zip)
            return True
    except Exception:
        pass
    return False

def get_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir): return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts: return None
    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, ckpts[-1])

def read_json_if_exists(path):
    try:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None

# =====================================================================
# BƯỚC 1: TẢI DỮ LIỆU TỪ GOOGLE DRIVE (PRETRAIN HOẶC RESUME)
# =====================================================================

# 1.1 Tải Pretrain (Nếu có)
if not os.listdir(PRETRAIN_CKPT_DIR):
    if PRETRAIN_WEIGHTS_DRIVE_ID:
        print(f">>> 📥 Downloading Pretrain Weights from ID: {PRETRAIN_WEIGHTS_DRIVE_ID}")
        success = download_and_extract_checkpoint(PRETRAIN_WEIGHTS_DRIVE_ID, PRETRAIN_CKPT_DIR)
        if success:
            contents = os.listdir(PRETRAIN_CKPT_DIR)
            subdirs = [d for d in contents if os.path.isdir(os.path.join(PRETRAIN_CKPT_DIR, d))]
            if len(subdirs) == 1 and "config.json" not in contents:
                nested_dir = os.path.join(PRETRAIN_CKPT_DIR, subdirs[0])
                for filename in os.listdir(nested_dir):
                    shutil.move(os.path.join(nested_dir, filename), PRETRAIN_CKPT_DIR)
                os.rmdir(nested_dir)

# 1.2 Tải Checkpoint Resume (Nếu có)
if RESUME_FINETUNE and RESUME_DRIVE_ID:
    # Chỉ tải nếu trong RUN_DIR chưa có checkpoint nào
    if not get_latest_checkpoint(RUN_DIR):
        print(f">>> 📥 Downloading Resume Checkpoint from ID: {RESUME_DRIVE_ID}")
        success = download_and_extract_checkpoint(RESUME_DRIVE_ID, RUN_DIR)
        if success:
            # Sắp xếp lại nếu giải nén bị bọc lồng 1 thư mục cha (không bắt đầu bằng checkpoint-)
            contents = os.listdir(RUN_DIR)
            if len(contents) == 1 and os.path.isdir(os.path.join(RUN_DIR, contents[0])) and not contents[0].startswith("checkpoint-"):
                nested_dir = os.path.join(RUN_DIR, contents[0])
                for filename in os.listdir(nested_dir):
                    shutil.move(os.path.join(nested_dir, filename), RUN_DIR)
                os.rmdir(nested_dir)

latest_finetune_ckpt = get_latest_checkpoint(RUN_DIR) if RESUME_FINETUNE else None
is_resuming = latest_finetune_ckpt is not None

# =====================================================================
# BƯỚC 2: BẢO VỆ CHUẨN KHOA HỌC & NẠP TOKENIZER / TRỌNG SỐ
# =====================================================================
if is_resuming:
    ckpt_cfg = read_json_if_exists(os.path.join(latest_finetune_ckpt, "config.json"))
    if isinstance(ckpt_cfg, dict):
        ckpt_qa = ckpt_cfg.get("ablation_use_qaclip", True)
        ckpt_vs = ckpt_cfg.get("ablation_use_vs", True)
        ckpt_ocr = ckpt_cfg.get("ablation_use_ocr", True)
        
        if (ckpt_qa != USE_QACLIP) or (ckpt_vs != USE_VS) or (ckpt_ocr != USE_OCR):
            raise ValueError(f"❌ XUNG ĐỘT ABLATION! Cấu hình đang resume ({ckpt_qa}, {ckpt_vs}, {ckpt_ocr}) "
                             f"không khớp với thiết lập hiện tại ({USE_QACLIP}, {USE_VS}, {USE_OCR}).")

try:
    if is_resuming:
        print(f"🔄 Resuming Tokenizer from checkpoint: {latest_finetune_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(latest_finetune_ckpt, local_files_only=True)
    else:
        if os.path.exists(PRETRAIN_CKPT_DIR) and os.listdir(PRETRAIN_CKPT_DIR):
            print(f"📦 Loading Tokenizer from local pretrain dir: {PRETRAIN_CKPT_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_CKPT_DIR, local_files_only=True)
        else:
            print(f"☁️ No local pretrain found. Loading Tokenizer from HuggingFace: {model.config.vit5_name}")
            tokenizer = AutoTokenizer.from_pretrained(model.config.vit5_name)
except Exception as e:
    raise ValueError(f"❌ Tokenizer could not be loaded: {e}")

if not is_resuming:
    state_dict = None
    safe_path = os.path.join(PRETRAIN_CKPT_DIR, "model.safetensors")
    bin_path = os.path.join(PRETRAIN_CKPT_DIR, "pytorch_model.bin")

    if os.path.exists(safe_path):
        state_dict = load_file(safe_path)
        print("   -> 📥 Loaded model.safetensors from pretrain folder.")
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        print("   -> 📥 Loaded pytorch_model.bin from pretrain folder.")
        model.load_state_dict(state_dict, strict=False)
    else:
        # VÁ LỖI TẠI ĐÂY: Không raise ValueError nữa, báo hiệu Finetune Raw!
        print("   -> 🌟 KHÔNG CÓ TRỌNG SỐ PRETRAIN CUSTOM. Tiến hành Finetune RAW từ base model!")

    ip_path = os.path.join(PRETRAIN_CKPT_DIR, "image_processor")
    if os.path.isdir(ip_path):
        ip_reload = CLIPImageProcessor.from_pretrained(ip_path)
        model.image_processor = ip_reload
        print("   -> 🖼️ Loaded custom Image Processor.")

model.to(DEVICE)
model.pretrain = False

# BƠM CẤU HÌNH ABLATION VÀO MODEL CONFIG
model.config.ablation_use_qaclip = USE_QACLIP
model.config.ablation_use_vs = USE_VS
model.config.ablation_use_ocr = USE_OCR

if hasattr(model.config, "ablation_mode"):
    delattr(model.config, "ablation_mode")

print("\n========================================================")
print(f"🚀 RUNNING FINETUNE WITH ABLATION SIGNATURE: {CURRENT_ABLATION_SIGNATURE}")
print(f"   [ ] QACLIP Late Fusion : {USE_QACLIP}")
print(f"   [ ] Visual Search Crop : {USE_VS}")
print(f"   [ ] OCR Consformer     : {USE_OCR}")
print(f"   [ ] Is Resuming        : {is_resuming}")
print("========================================================\n")

if hasattr(model, "visual_search") and hasattr(model.visual_search, "vit_processor"):
    model.visual_search.vit_processor = model.image_processor
elif hasattr(model, "visual_search") and hasattr(model.visual_search, "processor"):
    model.visual_search.processor = model.image_processor

gen_max_new = int(getattr(model.config, "generation_max_new_tokens", 27))
gen_num_beams = int(getattr(model.config, "generation_num_beams", 4))
model.generation_config = GenerationConfig(
    max_new_tokens=gen_max_new,
    num_beams=gen_num_beams,
    do_sample=False,
    pad_token_id=model.config.pad_token_id,
    eos_token_id=model.config.eos_token_id,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# =====================================================================
# BƯỚC 3: KHỞI TẠO TRAINER & HUẤN LUYỆN
# =====================================================================
data_collator_ft = ViT5VQADataCollator(
    tokenizer=tokenizer,
    image_processor=model.image_processor,
    ocr_encoder=vision_ocr,
    config=model.config,
    term_vocab_path=VOCAB_PATH,
    viet_vocab_path=VIET_VOCAB_PATH,
    eng_vocab_path="",
    dataframe=train_df,
    pretrain=False,
)

if hasattr(data_collator_ft, "set_mode"):
    data_collator_ft.set_mode(pretrain=False, mask_prob=0.0)

compute_metrics_fn = build_compute_metrics_finetune(tokenizer)

finetune_args = Seq2SeqTrainingArguments(
    output_dir=RUN_DIR,
    logging_dir=LOGS_DIR,
    seed=SEED,
    data_seed=SEED,
    num_train_epochs=TARGET_FINETUNE_EPOCHS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    predict_with_generate=True,
    optim="adamw_torch",
    dataloader_num_workers=4,
    generation_max_length=gen_max_new,
    generation_num_beams=gen_num_beams,
    report_to=None,
    disable_tqdm=False,
    fp16=False,
    gradient_checkpointing=False,
    overwrite_output_dir=True,
)

finetune_trainer = TaskSpecificTrainer(
    model=model,
    args=finetune_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator_ft,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_fn,
)

def continue_finetune(trainer, target_epochs, resume=True):
    trainer.args.num_train_epochs = target_epochs
    latest_ckpt = None
    if resume:
        latest_ckpt = get_latest_checkpoint(trainer.args.output_dir)
    
    if latest_ckpt is not None:
        print(f"▶️ Bắt đầu nối tiếp huấn luyện từ: {latest_ckpt}")
        train_result = trainer.train(resume_from_checkpoint=latest_ckpt)
    else:
        print("▶️ Bắt đầu huấn luyện từ số 0...")
        train_result = trainer.train()
    return train_result

train_result = continue_finetune(finetune_trainer, TARGET_FINETUNE_EPOCHS, RESUME_FINETUNE)

# =====================================================================
# BƯỚC 4: ĐÁNH GIÁ, LƯU KẾT QUẢ & DỌN RÁC Ổ CỨNG
# =====================================================================
best_ckpt_path = finetune_trainer.state.best_model_checkpoint
recorded_best_metric = finetune_trainer.state.best_metric

if best_ckpt_path and os.path.exists(best_ckpt_path):
    loaded_state_dict = None
    safe_path = os.path.join(best_ckpt_path, "model.safetensors")
    bin_path = os.path.join(best_ckpt_path, "pytorch_model.bin")
    if os.path.exists(safe_path):
        loaded_state_dict = load_file(safe_path)
    elif os.path.exists(bin_path):
        loaded_state_dict = torch.load(bin_path, map_location="cpu")
    if loaded_state_dict is not None:
        model.load_state_dict(loaded_state_dict, strict=False)

verify_metrics = finetune_trainer.evaluate()
verify_cider = float(verify_metrics.get("eval_cider", 0.0) or 0.0)
verify_f1 = float(verify_metrics.get("eval_f1", 0.0) or 0.0)
print(f"📊 Verified Score for {CURRENT_ABLATION_SIGNATURE} ->  F1: {verify_f1:.4f} | CIDEr: {verify_cider:.4f}")

model.config.ablation_use_qaclip = USE_QACLIP
model.config.ablation_use_vs = USE_VS
model.config.ablation_use_ocr = USE_OCR

model.save_pretrained(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)
if getattr(model, "image_processor", None) is not None:
    model.image_processor.save_pretrained(os.path.join(BEST_DIR, "image_processor"))

with open(os.path.join(BEST_DIR, "eval_metrics_verified.json"), "w") as f:
    json.dump(verify_metrics, f, indent=2)

with open(os.path.join(BEST_DIR, "training_info.json"), "w") as f:
    json.dump(
        {
            "ablation_signature": CURRENT_ABLATION_SIGNATURE,
            "ablation_setup": {
                "USE_QACLIP": USE_QACLIP,
                "USE_VS": USE_VS,
                "USE_OCR": USE_OCR
            },
            "best_checkpoint_source": best_ckpt_path,
            "recorded_best_metric": recorded_best_metric,
            "verified_f1": verify_f1,
            "verified_cider": verify_cider,
            "epochs_trained": finetune_trainer.state.epoch,
            "resume_used": bool(RESUME_FINETUNE and is_resuming),
        },
        f,
        indent=2,
    )

if "get_model_fingerprint" in globals():
    fp_finetune = get_model_fingerprint(finetune_trainer.model)
    with open(os.path.join(BEST_DIR, "fingerprint_finetune.json"), "w") as f:
        json.dump(fp_finetune, f, indent=2)

# Dọn dẹp Pretrain weight để tiết kiệm dung lượng
if os.path.exists(PRETRAIN_CKPT_DIR):
    try: shutil.rmtree(PRETRAIN_CKPT_DIR)
    except Exception: pass

# Xóa các Checkpoint cũ, chỉ giữ lại Checkpoint mới nhất (Last)
last_ckpt_path = get_latest_checkpoint(RUN_DIR)
if os.path.exists(RUN_DIR) and last_ckpt_path:
    for item in os.listdir(RUN_DIR):
        item_path = os.path.join(RUN_DIR, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-") and item_path != last_ckpt_path:
            try: shutil.rmtree(item_path)
            except Exception: pass

# Đóng gói Bản tốt nhất (Best)
zip_name = f"best_bundle_{CURRENT_ABLATION_SIGNATURE}.zip"
zip_path = os.path.join(OUTPUT_PATH, zip_name)
zip_folder(BEST_DIR, zip_path)
print(f"✅ Created Best Bundle Zip: {zip_path}")

if os.path.exists(BEST_DIR):
    try: shutil.rmtree(BEST_DIR)
    except Exception: pass

# Đóng gói Checkpoint đang chạy dở (Last) để tiện Resume sau này
if last_ckpt_path:
    resume_zip_name = f"last_checkpoint_{CURRENT_ABLATION_SIGNATURE}.zip"
    resume_zip_path = os.path.join(OUTPUT_PATH, resume_zip_name)
    try:
        zip_folder(last_ckpt_path, resume_zip_path)
        print(f"✅ Created Resume Checkpoint Zip: {resume_zip_path}")
    except Exception:
        pass