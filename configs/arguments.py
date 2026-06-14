import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import Seq2SeqTrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # Ablation Flags
    ablation_use_qaclip: bool = field(
        default=True,
        metadata={"help": "Whether to use Late Fusion QACLIP"}
    )
    ablation_use_vs: bool = field(
        default=True,
        metadata={"help": "Whether to use Visual Search (crop)"}
    )
    ablation_use_ocr: bool = field(
        default=True,
        metadata={"help": "Whether to use OCR Consformer"}
    )
    loss_ablation_mode: str = field(
        default="all",
        metadata={"help": "Pretrain loss ablation mode: 'all', 'only_itm_mlm', or 'only_twc_ocr_aug'"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default="ViTextVQA",
        metadata={"help": "The name of the dataset to use."}
    )
    data_dir: str = field(
        default="./datasets",
        metadata={"help": "Local path where datasets are stored or will be downloaded."}
    )
    vocab_path: str = field(
        default="configs/data/term_vocab.txt",
        metadata={"help": "Path to the term vocabulary file."}
    )
    viet_vocab_path: str = field(
        default="configs/data/viet_vocab.txt",
        metadata={"help": "Path to the Vietnamese vocabulary file."}
    )

@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    """
    Custom arguments for training to include standard HF arguments plus our custom flags if needed.
    """
    resume_checkpoint_id: Optional[str] = field(
        default=None,
        metadata={"help": "Google Drive ID of a checkpoint zip to download and resume from."}
    )
    pretrain_weights_id: Optional[str] = field(
        default=None,
        metadata={"help": "Google Drive ID of pretrain weights to download."}
    )
    smoke_test: bool = field(
        default=False,
        metadata={"help": "Run a quick check with tiny dataset and minimal steps to verify the pipeline."}
    )
