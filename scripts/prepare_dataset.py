#!/usr/bin/env python
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import os
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import pandas as pd
import yaml

from configs.base_config import configure_env, OUTPUT_PATH, SEED
from configs.base_config import VOCAB_DIR, VOCAB_PATH, VIET_VOCAB_PATH, ENG_VOCAB_PATH
from data.dataset_hub import DatasetHubLoader, _maybe_download

def main(args):
    RAW_DATASET_DIR = os.path.join(OUTPUT_PATH, "raw")
    OUT_DIR = os.path.join(OUTPUT_PATH, "datasets")

    # 1. Download Vocabularies if Vocab.yaml exists
    vocab_cfg_path = "configs/data/Vocab.yaml"
    if os.path.exists(vocab_cfg_path):
        print("\n⬇️  Downloading Vocabularies...")
        with open(vocab_cfg_path, 'r', encoding='utf-8') as f:
            vocab_config = yaml.safe_load(f)
        
        os.makedirs(VOCAB_DIR, exist_ok=True)
        
        def prepare_vocab_file(section_key, default_dest):
            section = vocab_config.get(section_key, {})
            if not section:
                return
            local_dir = section.get('dir', '')
            drive_id = section.get('drive_id', '')
            if local_dir and os.path.exists(local_dir):
                print(f"   -> Using local vocab for {section_key}: {local_dir}")
                shutil.copy2(local_dir, default_dest)
            elif drive_id:
                print(f"   -> Preparing {section_key} from Google Drive ID: {drive_id}")
                _maybe_download(drive_id, None, default_dest)
            else:
                print(f"   ⚠️ No config found for {section_key}, skipping.")

        prepare_vocab_file('term_vocab', VOCAB_PATH)
        prepare_vocab_file('vietnamese', VIET_VOCAB_PATH)
        prepare_vocab_file('english', ENG_VOCAB_PATH)
        print("   ✅ Vocabularies ready.")

    # 2. Load dataset config
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
            print("\n📥 Downloading viet_vocab.txt...")
            download_file(viet_id, os.path.join(out_vocab_dir, "viet_vocab.txt"))

    NAME_SET1 = config['dataset_name']
    
    # Resolve image source
    image_dir = config['image'].get('dir', '')
    image_id = config['image'].get('drive_id', '')
    if image_dir and os.path.isdir(image_dir):
        image_dir_override = image_dir
        image_zip_id = None
    else:
        image_dir_override = ''
        image_zip_id = image_id

    # Resolve ocr source
    ocr_dir = config['ocr'].get('dir', '') if 'ocr' in config else ''
    ocr_id = config['ocr'].get('drive_id', '') if 'ocr' in config else ''
    if ocr_dir and os.path.isdir(ocr_dir):
        ocr_dir_override = ocr_dir
        ocr_zip_id = None
    else:
        ocr_dir_override = ''
        ocr_zip_id = ocr_id

    # Resolve train/validation/test splits
    def resolve_split(section):
        local_dir = section.get('dir', '')
        drive_id = section.get('drive_id', '')
        if local_dir and os.path.exists(local_dir):
            return local_dir
        return drive_id

    train_id = resolve_split(config['dataset']['train'])
    val_id = resolve_split(config['dataset']['validation'])
    test_id = resolve_split(config['dataset']['test'])

    hub = DatasetHubLoader(RAW_DATASET_DIR, OUT_DIR)

    # 3. Register dataset
    hub.register_dataset(
        dataset_name=NAME_SET1,
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

    print(f"\n⬇️  Downloading & Extracting Dataset {NAME_SET1}...")
    path_info1 = hub.prepare(NAME_SET1)
    dfs1 = hub.load_task(NAME_SET1)
    print(f"   ✅ {NAME_SET1} Ready: {len(dfs1['train'])} train samples.")

    def merge_and_shuffle(df_list, split_name):
        if not df_list: return pd.DataFrame()
        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"   -> Prepared [{split_name.upper()}]: Total {len(combined)} samples.")
        return combined

    print("\n🔄 Preparing final dataset splits...")
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