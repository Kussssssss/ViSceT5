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
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML configuration file"}
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
        metadata={"help": "Whether to use SceSpaVis (Scene Spatial-Visual-Semantic OCR Representation Module: "
                          "Spatial 2D Box + Visual Det/Rec Spotting + Char-level embeddings). "
                          "If False, uses clean Text-Only OCR baseline."}
    )
    ablation_use_ocr_input: bool = field(
        default=True,
        metadata={"help": "If False, DROP all OCR features entirely (text/box/det/rec/char) so the "
                          "fused sequence is [question, image] only. Different from ablation_use_ocr "
                          "(which only switches SceSpaVis<->text-only but still keeps OCR)."}
    )
    ablation_use_ocr_aug: bool = field(
        default=True,
        metadata={"help": "Ablation #4 (finetune): apply OCR augmentation — the OCR-related "
                          "correct/noise/keep step that appends an augmented OCR view next to the "
                          "clean OCR tokens. True = clean+augmented (doubled OCR tokens); "
                          "False = clean OCR only."}
    )
    loss_ablation_mode: str = field(
        default="all",
        metadata={"help": "Pretrain loss ablation mode: 'all', 'only_itm_mlm', 'only_twc_ocr_aug', 'gen_all', 'gen'"}
    )
    vision_unfreeze_last_n: int = field(
        default=0,
        metadata={"help": "Unfreeze last N CLIP vision layers (+post_layernorm) so pretrain learns visual features. 0 = frozen backbone."}
    )
    mlm_mask_mode: str = field(
        default="wholeword",
        metadata={"help": "Pretrain MLM masking granularity: 'wholeword' (mask whole OCR/question word, TWA-faithful) or 'subword' (old BERT per-subword, for A/B ablation)."}
    )
    mlm_ocr_in_text: bool = field(
        default=False,
        metadata={"help": "Include OCR tokens in the MLM encoder text branch. False (default) = question-only (removes OCR-as-text copy crutch, aligns with finetune; OCR learned via gen+TWC). True = old question+OCR."}
    )
    num_bbox_bins: int = field(
        default=200,
        metadata={"help": "Number of discrete coordinate bins for BBox prediction. 200 bins = 0.5% "
                          "of the image side, already finer than the 1/14 = 7.1% granularity of the "
                          "patch grid the model reasons over; 1000 bins spent capacity on precision "
                          "the evidence cannot support."}
    )
    lambda_ground: float = field(
        default=0.5,
        metadata={"help": "Weight of the region-grounding loss (supervises BOTH the T5 relevance map "
                          "that steers the AVF crop AND the QA-CLIP question-guided patch map). This "
                          "is the direct 'where to look' signal that answer-CE alone never provided."}
    )
    lambda_bbox_ce: float = field(
        default=1.0,
        metadata={"help": "Weight of BBox loss relative to text generation loss."}
    )
    pretrain_use_vs: bool = field(
        default=True,
        metadata={"help": "Whether to use Visual Search (AVF) in pretrain. Combined with "
                          "ablation_use_vs (AND) so pretrain can drop AVF without touching "
                          "the shared ablation flag."}
    )
    vs_t5_guided: bool = field(
        default=True,
        metadata={"help": "Pretrain Visual Search heatmap source. True = LATE path: heatmap from "
                          "the T5 encoder's last-layer attention (mask-box queries + prompt -> image "
                          "patches), crop injected back into the encoder image states via AVFFusion. "
                          "False = EARLY path: heatmap from QA-CLIP attention before the T5 encoder."}
    )
    vs_crop_encoder: str = field(
        default="clip",
        metadata={"help": "Backbone that encodes the AVF zoom-in crop. 'clip' (default) reuses the "
                          "model's own CLIP ViT (AnyRes-style second view, shared weights, transfers "
                          "to finetune, and CLIP actually has glyph sensitivity). 'convnext' keeps "
                          "the old separate ConvNeXt-V2 backbone, which was trained for ImageNet "
                          "classification and cannot read text."}
    )
    use_ocr_aug_pretrain: bool = field(
        default=False,
        metadata={"help": "Whether to apply OCR augmentation in pretrain (default False)."}
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
        metadata={"help": "Run a quick check with a small dataset and few steps to verify the pipeline."}
    )
    smoke_train_samples: int = field(
        default=512,
        metadata={"help": "Number of training samples to keep in smoke/mock test (enough to be meaningful, still fast)."}
    )
    smoke_eval_samples: int = field(
        default=128,
        metadata={"help": "Number of eval samples to keep in smoke/mock test."}
    )
    smoke_max_steps: int = field(
        default=100,
        metadata={"help": "Number of optimizer steps to run in smoke/mock test (long enough to see loss/metric trends)."}
    )
