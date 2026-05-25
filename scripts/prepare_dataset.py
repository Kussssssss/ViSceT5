"""
scripts/prepare_dataset.py
Register and prepare VQA datasets. Run: python scripts/prepare_dataset.py
"""

#!/usr/bin/env python
from configs.base_config import configure_env, OUTPUT_PATH, SEED
from data.dataset_hub import DatasetHubLoader

import pandas as pd
import os

RAW_DATASET_DIR = OUTPUT_PATH + "/raw"
OUT_DIR = OUTPUT_PATH + "/datasets"

hub = DatasetHubLoader(RAW_DATASET_DIR, OUT_DIR)

NAME_SET1 = "ViTextVQA"
hub.register_dataset(
    dataset_name=NAME_SET1,
    task_type="VQA",
    image_zip_id=None,
    image_dir_override='/kaggle/input/vitextvqa-viocrvqa/ViTextVQA_images/st_images',
    ocr_zip_id=None,
    ocr_dir_override="/kaggle/input/vitextvqa-viocrvqa/OCR_ViTextVQA/swintextspotter",
    splits={
        "train":      {"id": "1fKBlPAA2DmEf1Y1MESIUIuqlUeCBTXjE", "url": None},
        "validation": {"id": "18nt6rJQZSf0g6Z7t0skaTuU_Bgc7EHAP", "url": None},
        "test":       {"id": "1S58lXoNadz0j4A4GhJPSc-Glq3ydaC1N", "url": None},
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
display(final_train_df.head(5))