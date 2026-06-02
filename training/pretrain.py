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
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    set_seed,
    GenerationConfig,
)

# Project imports
from configs.base_config import configure_env, OUTPUT_PATH, SEED, VOCAB_PATH, VIET_VOCAB_PATH, ENG_VOCAB_PATH
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

def main(args_list=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    
    if args_list is not None:
        if len(args_list) == 1 and args_list[0].endswith(".yaml"):
            model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(args_list[0]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)
    else:
        is_jupyter = any("ipykernel" in arg or "colab" in arg for arg in sys.argv) or ("ipykernel_launcher" in sys.argv[0] or "colab_kernel_launcher" in sys.argv[0])
        
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
        elif is_jupyter:
            default_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "pretrain.yaml"))
            print(f"[Jupyter/Colab] Environment detected. Loading default config: {default_yaml}")
            model_args, data_args, training_args = parser.parse_yaml_file(default_yaml)
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)
        
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Prepare Data
    print(f">>> Preparing Dataset: {data_args.dataset_name}")
    raw_dir = os.path.join(data_args.data_dir, "raw")
    out_dir = os.path.join(data_args.data_dir, "processed")
    hub = DatasetHubLoader(raw_dir, out_dir)
    
    # Normally the dataset should be registered before preparing, 
    # assuming we load yaml config or it's pre-registered:
    # However, since this script should run out-of-the-box, we assume prepare_dataset.py 
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

    from data.vocab import setup_augmented_vocab
    print(">>> Setting up augmented vocabulary...")
    augmented_vocab_path = setup_augmented_vocab(
        viet_vocab_path=data_args.viet_vocab_path,
        eng_vocab_path="",
        dataframe=train_df,
        pretrain_vocab_path=data_args.vocab_path,
        output_dir=os.path.join(data_args.data_dir, "processed", "dict")
    )

    data_collator = ViT5VQADataCollator(
        tokenizer=tokenizer,
        image_processor=model.image_processor,
        ocr_encoder=vision_ocr,
        config=model.config,
        term_vocab_path=augmented_vocab_path,
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
