"""
data/data_loader.py
load_dataset_final() — load or build train/val/test DataFrames from CSV cache.
"""

import pandas as pd
import torch
import os
import json

VOCAB_DIR = OUTPUT_PATH + "/vocab"
os.makedirs(VOCAB_DIR, exist_ok=True)

def load_dataset_final():
    print("\nLoading Final Merged Datasets...")

    if 'final_train_df' not in globals():
        print("WARNING: Could not find 'final_train_df' in memory.")
        print("Trying to find CSV cache...")

        try:
            t_df = pd.read_csv(OUTPUT_PATH+"/merged_train.csv")
            v_df = pd.read_csv(OUTPUT_PATH+"/merged_val.csv")
            te_df = pd.read_csv(OUTPUT_PATH+"/merged_test.csv")
            print("Restored data from CSV cache.")
            return t_df, v_df, te_df
        except FileNotFoundError:
            raise RuntimeError("Error: You haven't run the 'Merge datasets' step (DatasetHubLoader). Go back to the previous step.")

    t_df = final_train_df.copy()
    v_df = final_val_df.copy()
    te_df = final_test_df.copy()

    print("Saving merged datasets to CSV backup...")
    t_df.to_csv(OUTPUT_PATH+"/merged_train.csv", index=False)
    v_df.to_csv(OUTPUT_PATH+"/merged_val.csv", index=False)
    te_df.to_csv(OUTPUT_PATH+"/merged_test.csv", index=False)

    print(f"-"*30)
    print(f"Train samples:      {len(t_df)}")
    print(f"Validation samples: {len(v_df)}")
    print(f"Test samples:       {len(te_df)}")
    print(f"-"*30)

    return t_df, v_df, te_df

train_df, val_df, test_df = load_dataset_final()

print("\nSample Check:")
sample = train_df.iloc[0]
print(f"Question: {sample['question']}")
print(f"Image Path: {sample['image_path']}")
if os.path.exists(sample['image_path']):
    print("Image file exists.")
else:
    print("Image file NOT found. Check paths!")