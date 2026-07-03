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

def main(args_list=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    default_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "finetune.yaml"))
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
            return

    if training_args.smoke_test:
        print("🚨 RUNNING IN SMOKE TEST MODE: Truncating dataset and reducing training steps.")
        train_df = train_df.head(8)
        val_df = val_df.head(4)
        training_args.max_steps = 3
        training_args.num_train_epochs = 1.0
        training_args.logging_steps = 1
        training_args.eval_steps = 2
        training_args.save_steps = 3

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
    else:
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

    # Apply Finetune config overrides
    mode = model_args.loss_ablation_mode
    use_ocr_aug = mode in ["all", "only_twc_ocr_aug"]

    model.pretrain = False
    model.config.pretrain = False
    model.config.pretrain_ablation_mode = mode
    model.config.ablation_use_qaclip = model_args.ablation_use_qaclip
    model.config.ablation_use_vs = model_args.ablation_use_vs
    model.config.ablation_use_ocr = model_args.ablation_use_ocr
    model.config.use_twc = False  # TWC is disabled during finetuning
    model.config.use_ocr_aug_finetune = use_ocr_aug
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

    print(">>> Starting Finetune...")
    if training_args.resume_from_checkpoint:
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
