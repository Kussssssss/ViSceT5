"""
configs/base_config.py
Global constants and environment setup for OpenViVQA.
"""

import os
import warnings
import logging as _std_logging

import torch
import pandas as pd

from transformers import logging as hf_logging
try:
    from pandas.errors import SettingWithCopyWarning
except ImportError:
    try:
        from pandas.core.common import SettingWithCopyWarning
    except ImportError:
        class SettingWithCopyWarning(UserWarning):
            pass

# ─────────────────────────────────────────────────────────
# Global seeds & paths
# ─────────────────────────────────────────────────────────
SEED: int = 42
STATE: int = 2
OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "./output")


def configure_env(
    output_path: str = OUTPUT_PATH,
    hf_home: str = "./cache",
    cuda_device: str = "0",
    disable_wandb: bool = True,
) -> None:
    """Apply environment variables and suppress noisy logging."""
    os.environ["HF_HOME"] = hf_home
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if disable_wandb:
        os.environ["WANDB_DISABLED"] = "True"

    _std_logging.disable(_std_logging.INFO)
    _std_logging.disable(_std_logging.WARNING)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", SettingWithCopyWarning)

    hf_logging.set_verbosity_error()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)

    global OUTPUT_PATH
    OUTPUT_PATH = output_path
    os.makedirs(output_path, exist_ok=True)


# Convenient device reference
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
