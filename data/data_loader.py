"""Read the CSV splits produced by scripts.prepare_dataset."""

from pathlib import Path

import pandas as pd

from configs.base_config import OUTPUT_PATH


def load_dataset_final(output_path=OUTPUT_PATH):
    directory = Path(output_path)
    try:
        return tuple(
            pd.read_csv(directory / f"merged_{split}.csv")
            for split in ("train", "val", "test")
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Run scripts/prepare_dataset.py before loading the CSV splits."
        ) from error
