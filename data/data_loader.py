"""
data/data_loader.py
load_dataset_final() — load or build train/val/test DataFrames from CSV cache.
"""

import pandas as pd
import torch
import os
import json
from configs.base_config import OUTPUT_PATH

def load_dataset_final():
    print("\nLoading Prepared Datasets...")
    try:
        t_df = pd.read_csv(os.path.join(OUTPUT_PATH, "merged_train.csv"))
        v_df = pd.read_csv(os.path.join(OUTPUT_PATH, "merged_val.csv"))
        te_df = pd.read_csv(os.path.join(OUTPUT_PATH, "merged_test.csv"))
        print("   ✅ Loaded dataset splits from CSV cache.")
        return t_df, v_df, te_df
    except FileNotFoundError:
        raise RuntimeError(
            "❌ ERROR: Dataset CSV cache not found. "
            "Please run 'python scripts/prepare_dataset.py --config <config_path>' first."
        )

# Load dataset splits at module level
try:
    train_df, val_df, test_df = load_dataset_final()
except Exception as e:
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()