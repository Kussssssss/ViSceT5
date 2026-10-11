from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path or Hub identifier of an OpenViVQA checkpoint for answer generation."
        },
    )
    # Ablation Flags
    ablation_use_qaclip: bool = field(
        default=True, metadata={"help": "Whether to use Late Fusion QACLIP"}
    )
    ablation_use_vs: bool = field(
        default=True, metadata={"help": "Whether to use Visual Search (crop)"}
    )
    ablation_use_ocr: bool = field(
        default=True, metadata={"help": "Whether to use OCR Consformer"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default="ViTextVQA", metadata={"help": "The name of the dataset to use."}
    )
    data_dir: str = field(
        default="./datasets",
        metadata={
            "help": "Local path where datasets are stored or will be downloaded."
        },
    )


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    """
    Custom arguments for training to include standard HF arguments plus our custom flags if needed.
    """

    resume_checkpoint_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Google Drive ID of a checkpoint zip to download and resume from."
        },
    )
    smoke_test: bool = field(
        default=False,
        metadata={
            "help": "Run a quick check with a small dataset and few steps to verify the pipeline."
        },
    )
    smoke_train_samples: int = field(
        default=8,
        metadata={
            "help": "Number of training samples to keep in smoke/mock test (enough to be meaningful, still fast)."
        },
    )
    smoke_eval_samples: int = field(
        default=4,
        metadata={"help": "Number of eval samples to keep in smoke/mock test."},
    )
    smoke_max_steps: int = field(
        default=3,
        metadata={
            "help": "Number of optimizer steps to run in smoke/mock test (long enough to see loss/metric trends)."
        },
    )
