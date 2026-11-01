#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import argparse
import yaml

from configs.base_config import OUTPUT_PATH
from data.dataset_hub import DatasetHubLoader

def main(args):
    RAW_DATASET_DIR = os.path.join(args.data_dir, "raw")
    OUT_DIR = os.path.join(args.data_dir, "processed")

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

    selected_splits = getattr(args, "splits", None) or ("train", "validation", "test")
    split_specs = {name: {"id": resolve_split(config["dataset"][name]), "url": None}
                   for name in selected_splits}

    hub = DatasetHubLoader(RAW_DATASET_DIR, OUT_DIR)

    # 3. Register dataset
    hub.register_dataset(
        dataset_name=NAME_SET1,
        task_type="VQA",
        image_zip_id=image_zip_id,
        image_dir_override=image_dir_override,
        ocr_zip_id=ocr_zip_id,
        ocr_dir_override=ocr_dir_override,
        splits=split_specs,
    )

    print(f"\n⬇️  Downloading & Extracting Dataset {NAME_SET1}...")
    hub.prepare(NAME_SET1)
    dfs1 = hub.load_task(NAME_SET1)
    print(f"   ✅ {NAME_SET1} Ready: {len(dfs1['train'])} train samples.")

    def merge_and_shuffle(df_list, split_name):
        if not df_list: return pd.DataFrame()
        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   -> Prepared [{split_name.upper()}]: Total {len(combined)} samples.")
        return combined

    print("\n🔄 Preparing final dataset splits...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    suffixes = {"train": "train", "validation": "val", "test": "test"}
    for split in selected_splits:
        frame = merge_and_shuffle([dfs1[split]], split)
        frame.to_csv(os.path.join(OUTPUT_PATH, f"merged_{suffixes[split]}.csv"), index=False)
    print(f"   ✅ Saved prepared dataset CSVs to {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dataset")
    parser.add_argument("--config", type=str, default="configs/data/ViTextVQA.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to store downloaded data")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"))
    args = parser.parse_args()
    main(args)
