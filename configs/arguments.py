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
        default=False, metadata={"help": "Feed the question into CLIP (QA-ViT). False = plain CLIP."}
    )
    ablation_use_ocr: bool = field(
        default=True, metadata={"help": "SceSpaVis OCR representation. False = text-only OCR baseline."}
    )
    ablation_use_image: bool = field(
        default=True, metadata={"help": "False = no image features at all (CLIP is not built)."}
    )
    ablation_use_ocr_aug: bool = field(
        default=True, metadata={"help": "Append the corrected (related) OCR copy after the original."}
    )
    use_twc: bool = field(
        default=False, metadata={"help": "Add the TWA token-word contrastive loss during training."}
    )
    twc_weight: float = field(default=1.0, metadata={"help": "Weight of the TWC loss."})
    twc_loss: str = field(default="infonce", metadata={"help": "infonce (softmax, TWA paper) | bce (TWA code)."})
    twc_feature: str = field(default="fused", metadata={"help": "lexical (OCR text+chars) | fused (encoder output)."})
    twc_queue_size: int = field(
        default=16384, metadata={"help": "Cross-batch memory of word features used as extra negatives; 0 = in-batch."}
    )
    twc_queue_start: int = field(
        default=1000, metadata={"help": "Training forwards before the memory is read (features drift fast early)."}
    )
    twc_mask_same_text: bool = field(
        default=True, metadata={"help": "Never use a candidate with the same text as the query as a negative."}
    )
    twc_objective: str = field(
        default="answer", metadata={"help": "answer (question ↔ answer OCR words) | lexical (TWA) | both."}
    )
    twc_answer_weight: float = field(default=0.1, metadata={"help": "Weight of the answer-grounded TWC loss."})
    twc_temperature_init: float = field(
        default=0.07, metadata={"help": "Initial (learnable) TWC temperature τ; 0.07 = CLIP/TWA."}
    )
    ocr_aug_mode: str = field(
        default="separate", metadata={"help": "OCR Augmentation: separate (two ConsFormer passes, full concatenation) | concat (one pass) | "
                                            "lattice (separate passes, only changed words inserted after the original)."}
    )
    ocr_related: str = field(
        default="correct", metadata={"help": "separate/lattice: related copy = correct (deterministic) | augment "
                                             "(TWA keep/correct/noise at twc_adv_probability), same at train and test."}
    )
    twc_view: str = field(
        default="shared", metadata={"help": "shared | separate (decoder reads the correction, TWC its own augmented copy)."}
    )
    twc_adv_probability: float = field(
        default=0.4, metadata={"help": "TWA augmentation probability of each OCR word's related copy."}
    )
    qavit_mode: Optional[str] = field(
        default=None,
        metadata={"help": "QA-ViT in the late 6 CLIP layers (CLIP itself stays frozen): off = plain CLIP; "
                          "no_question = QA-ViT layers with an empty question, only the gated projection "
                          "learns (pre-refactor code with QA-ViT 'off'); question = full QA-ViT. "
                          "None = from ablation_use_qaclip (question / off)."},
    )
    ocr_feature_norm: str = field(
        default="layernorm", metadata={"help": "OCR stream norms: layernorm (pre-refactor code) | rmsnorm (refactor)."}
    )
    twc_related_train: str = field(
        default="correct", metadata={"help": "Related OCR copy at train: correct (as at test) | twa (augmentation)."}
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
        metadata={"help": "LR of randomly initialised modules (ConsFormer, OCR/char embeddings, TWC query); "
                          "None = learning_rate. M4C/TWA train new layers ~3-10x faster than pretrained ones."},
    )
    scalar_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "LR of scalar parameters (TWC temperatures, CLIP adapter gates), without weight decay; "
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
