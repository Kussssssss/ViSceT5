#!/usr/bin/env python
from configs.base_config import configure_env, OUTPUT_PATH, SEED
from data.dataset_hub import DatasetHubLoader

import pandas as pd
import os
import argparse
import yaml

def main(args):
    RAW_DATASET_DIR = os.path.join(args.data_dir, "raw")
    OUT_DIR = os.path.join(args.data_dir, "processed")

    hub = DatasetHubLoader(RAW_DATASET_DIR, OUT_DIR)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    vocab_cfg_path = os.path.join(os.path.dirname(args.config), "vocab.yaml")
    if os.path.exists(vocab_cfg_path):
        with open(vocab_cfg_path, 'r', encoding='utf-8') as f:
            vocab_cfg = yaml.safe_load(f) or {}
            
        term_id = vocab_cfg.get("term_vocab_id", "")
        viet_id = vocab_cfg.get("viet_vocab_id", "")
        out_vocab_dir = os.path.dirname(args.config)
        
        from utils.io_utils import download_file
        if term_id:
            print("\n📥 Downloading term_vocab.txt...")
            download_file(term_id, os.path.join(out_vocab_dir, "term_vocab.txt"))
        if viet_id:
            print("\n📥 Downloading viet_vocab.jsonl...")
            download_file(viet_id, os.path.join(out_vocab_dir, "viet_vocab.jsonl"))

    NAME_SET1 = config['dataset_name']

    image_zip_id = config['image']['drive_id']
    ocr_zip_id = config['ocr']['drive_id']
    
    train_id = config['dataset']['train']['drive_id'] or config['dataset']['train']['dir']
    val_id = config['dataset']['validation']['drive_id'] or config['dataset']['validation']['dir']
    test_id = config['dataset']['test']['drive_id'] or config['dataset']['test']['dir']

    # 3. TRUYỀN VÀO HÀM REGISTER
    hub.register_dataset(
        dataset_name=NAME_SET1,
        task_type="VQA",
        image_zip_id=image_zip_id,
        image_dir_override='',
        ocr_zip_id=ocr_zip_id,
        ocr_dir_override='',
        splits={
            "train":      {"id": train_id, "url": None},
            "validation": {"id": val_id, "url": None},
            "test":       {"id": test_id, "url": None},
        },
    )

    print("\n⬇️  Downloading & Extracting Dataset...")

    path_info1 = hub.prepare(NAME_SET1)
    dfs1 = hub.load_task(NAME_SET1)
    print(f"   ✅ {NAME_SET1} Ready: {len(dfs1['train'])} train samples.")

    def merge_and_shuffle(df_list, split_name):
        if not df_list: return pd.DataFrame()

        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"   -> Merged [{split_name.upper()}]: Total {len(combined)} samples.")
        return combined

    print("\n🔄 Merging Dataset...")

    final_train_df = merge_and_shuffle([dfs1["train"]], "train")
    final_val_df = merge_and_shuffle([dfs1["validation"]], "validation")
    final_test_df = merge_and_shuffle([dfs1["test"]], "test")

    print("\n📊 FINAL DATASET STATISTICS:")
    print(f"   - Training Samples:   {len(final_train_df)}")
    print(f"   - Validation Samples: {len(final_val_df)}")
    print(f"   - Test Samples:       {len(final_test_df)}")
    print(f"   - Columns:            {list(final_train_df.columns)}")

    print("\nSample Rows (Mixed):")
    print(final_train_df.head(5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dataset")
    parser.add_argument("--config", type=str, default="configs/data/ViTextVQA.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to store downloaded data")
    args = parser.parse_args()
    main(args)