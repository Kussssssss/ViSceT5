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
    # Module switches
    ablation_use_qaclip: bool = field(
        default=False, metadata={"help": "QA-ViT: the question enters the late CLIP layers through the "
                                         "learned adapter. False = plain frozen CLIP, no adapter."}
    )
    ablation_use_ocr: bool = field(
        default=True, metadata={"help": "Consformer (SceSpaVis) OCR representation. False = text-only OCR."}
    )
    ablation_use_image: bool = field(
        default=True, metadata={"help": "False = no image features at all (CLIP is not built)."}
    )
    use_twc: bool = field(
        default=False, metadata={"help": "TWC (TWA): related OCR copy after the original in the encoder "
                                         "input + contrastive loss on the encoder output."}
    )
    twc_weight: float = field(default=1.0, metadata={"help": "Weight of the TWC loss."})
    twc_adv_probability: float = field(
        default=0.35, metadata={"help": "TWA adv_pro: probability that a word of the related copy is changed."}
    )
    twc_temperature_init: float = field(
        default=0.07, metadata={"help": "Initial (learnable) TWC temperature τ; 0.07 = CLIP/TWA."}
    )
    ocr_feature_norm: str = field(
        default="layernorm", metadata={"help": "OCR stream norms: layernorm (pre-refactor code) | rmsnorm (refactor)."}
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
    term_vocab_path: str = field(
        default="configs/data/term_vocab.txt", metadata={"help": "Term list of the correction lexicon."}
    )
    viet_vocab_path: str = field(
        default="configs/data/viet_vocab.txt", metadata={"help": "Vietnamese dictionary of the correction lexicon."}
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
    new_module_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "LR of randomly initialised modules (ConsFormer, OCR/char embeddings); "
                          "None = learning_rate. M4C/TWA train new layers ~3-10x faster than pretrained ones."},
    )
    scalar_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "LR of scalar parameters (TWC temperature, QA-ViT gates), without weight decay; "
                          "None = learning_rate. Adam moves a scalar by ~lr per step, so 3e-5 barely moves it."},
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
