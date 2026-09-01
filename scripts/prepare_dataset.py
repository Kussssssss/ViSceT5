#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import pandas as pd
import argparse
import yaml

from configs.base_config import configure_env, OUTPUT_PATH, SEED
from data.dataset_hub import DatasetHubLoader

def main(args):
    RAW_DATASET_DIR = os.path.join(args.data_dir, "raw")
    OUT_DIR = os.path.join(args.data_dir, "processed")

    # 2. Parse config paths (supports comma-separated list)
    cfg_paths = [c.strip() for c in args.config.split(",") if c.strip()]
    
    # Download vocab if available in the first config's directory
    first_cfg = cfg_paths[0]
    first_cfg_dir = os.path.dirname(first_cfg) if os.path.exists(first_cfg) else "configs/data"
    vocab_cfg_path = os.path.join(first_cfg_dir, "vocab.yaml")
    if os.path.exists(vocab_cfg_path):
        with open(vocab_cfg_path, 'r', encoding='utf-8') as f:
            vocab_cfg = yaml.safe_load(f) or {}
            
        term_id = vocab_cfg.get("term_vocab_id", "")
        viet_id = vocab_cfg.get("viet_vocab_id", "")
        out_vocab_dir = first_cfg_dir
        
        from utils.io_utils import download_file
        if term_id:
            print("\n📥 Downloading term_vocab.txt...")
            download_file(term_id, os.path.join(out_vocab_dir, "term_vocab.txt"))
        if viet_id:
            print("\n📥 Downloading viet_vocab.txt...")
            download_file(viet_id, os.path.join(out_vocab_dir, "viet_vocab.txt"))

    hub = DatasetHubLoader(RAW_DATASET_DIR, OUT_DIR)
    all_train_dfs, all_val_dfs, all_test_dfs = [], [], []

    for cfg_entry in cfg_paths:
        if not os.path.exists(cfg_entry):
            cand = os.path.join("configs", "data", f"{cfg_entry}.yaml")
            if os.path.exists(cand):
                cfg_entry = cand
            else:
                print(f"⚠️ Config not found: {cfg_entry}; skipping.")
                continue

        with open(cfg_entry, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        NAME_SET = config['dataset_name']
        
        # Resolve image source
        image_dir = config.get('image', {}).get('dir', '')
        image_id = config.get('image', {}).get('drive_id', '')
        image_dir_override = image_dir if (image_dir and os.path.isdir(image_dir)) else ''
        image_zip_id = None if image_dir_override else image_id

        # Resolve ocr source
        ocr_dir = config.get('ocr', {}).get('dir', '')
        ocr_id = config.get('ocr', {}).get('drive_id', '')
        ocr_dir_override = ocr_dir if (ocr_dir and os.path.isdir(ocr_dir)) else ''
        ocr_zip_id = None if ocr_dir_override else ocr_id

        # Resolve train/validation/test splits
        def resolve_split(section):
            if not isinstance(section, dict):
                return ''
            local_dir = section.get('dir', '')
            drive_id = section.get('drive_id', '')
            if local_dir and os.path.exists(local_dir):
                return local_dir
            return drive_id

        ds_sec = config.get('dataset', {})
        train_id = resolve_split(ds_sec.get('train', {}))
        val_id = resolve_split(ds_sec.get('validation', {}))
        test_id = resolve_split(ds_sec.get('test', {}))

        # Register dataset
        hub.register_dataset(
            dataset_name=NAME_SET,
            task_type="VQA",
            image_zip_id=image_zip_id,
            image_dir_override=image_dir_override,
            ocr_zip_id=ocr_zip_id,
            ocr_dir_override=ocr_dir_override,
            splits={
                "train":      {"id": train_id, "url": None},
                "validation": {"id": val_id, "url": None},
                "test":       {"id": test_id, "url": None},
            },
        )

        print(f"\n⬇️  Downloading & Extracting Dataset {NAME_SET}...")
        hub.prepare(NAME_SET)
        dfs = hub.load_task(NAME_SET)
        print(f"   ✅ {NAME_SET} Ready: {len(dfs['train'])} train, {len(dfs['validation'])} val samples.")
        
        if len(dfs["train"]) > 0: all_train_dfs.append(dfs["train"])
        if len(dfs["validation"]) > 0: all_val_dfs.append(dfs["validation"])
        if len(dfs["test"]) > 0: all_test_dfs.append(dfs["test"])

    def merge_and_shuffle(df_list, split_name):
        if not df_list: return pd.DataFrame()
        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   -> Prepared [{split_name.upper()}]: Total {len(combined)} samples.")
        return combined

    print("\n🔄 Preparing final merged dataset splits...")
    final_train_df = merge_and_shuffle(all_train_dfs, "train")
    final_val_df = merge_and_shuffle(all_val_dfs, "validation")
    final_test_df = merge_and_shuffle(all_test_dfs, "test")

    print("\n📊 FINAL MERGED DATASET STATISTICS:")
    print(f"   - Training Samples:   {len(final_train_df)}")
    print(f"   - Validation Samples: {len(final_val_df)}")
    print(f"   - Test Samples:       {len(final_test_df)}")
    print(f"   - Columns:            {list(final_train_df.columns)}")
    print(f"   - Test Samples:       {len(final_test_df)}")
    print(f"   - Columns:            {list(final_train_df.columns)}")

    print("\nSample Rows (Mixed):")
    print(final_train_df.head(5))
    
    # Save to CSV in OUTPUT_PATH
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    final_train_df.to_csv(os.path.join(OUTPUT_PATH, "merged_train.csv"), index=False)
    final_val_df.to_csv(os.path.join(OUTPUT_PATH, "merged_val.csv"), index=False)
    final_test_df.to_csv(os.path.join(OUTPUT_PATH, "merged_test.csv"), index=False)
    print(f"   ✅ Saved prepared dataset CSVs to {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dataset")
    parser.add_argument("--config", type=str, default="configs/data/ViTextVQA.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to store downloaded data")
    args = parser.parse_args()
    main(args)