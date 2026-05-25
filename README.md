# OpenViVQA

**Vietnamese Visual Question Answering** — A multimodal model combining ViT5, CLIP, OCR Consformer, and Visual Search.

## Architecture

```
OpenViVQAModel
├── ViT5 (VietAI/vit5-base)            — Encoder-Decoder language backbone
├── QACLIPEncoder                       — CLIP with query-guided (MMCLIPAttention) vision encoder
│   └── InstructCLIPEncoder             — Late-fusion: plain CLIPEncoderLayer for first half,
│       ├── CLIPEncoderLayer            — MMCLIPEncoderLayer for second half
│       └── MMCLIPEncoderLayer
├── VisualSearch                        — ConvNeXtV2-based attention crop
└── OCR Consformer
    ├── OCREncoder (GroupAttention)     — Encodes OCR token sequences with neighbor attention
    ├── SpatialCirclePosition           — 2D distance-aware spatial position embedding
    └── SemanticOCREmbedding            — Fuses bounding box + text representations
```

**Training Objectives (Pretrain)**
| Objective | Description |
|-----------|-------------|
| **MLM** | Masked Language Modelling on question + OCR tokens |
| **TWC** | Token-Word Contrastive loss (aligned from TWA paper) |
| **ITM** | Image-Text Matching / Pollute detection head |

## Repository Structure

```
openvivqa/
├── configs/
│   ├── base_config.py          # SEED, OUTPUT_PATH, configure_env()
│   ├── model_config.py         # OpenViVQAConfig (PretrainedConfig)
│   └── ocr_config.py           # Default OCR encoder config
│
├── data/
│   ├── dataset_hub.py          # DatasetHubLoader — download & prepare datasets
│   ├── dataset.py              # ViT5VQADataset (PyTorch Dataset)
│   ├── collator.py             # ViT5VQADataCollator (TWC+MLM+ITM)
│   ├── vocab.py                # Char vocab, text normalisation, OCR augmentation utils
│   ├── ocr_utils.py            # Vision_Encode_Ocr_Feature, reading-order sort
│   ├── data_loader.py          # load_dataset_final()
│   └── eda.py                  # EDA analyzers
│
├── models/
│   ├── openvivqa_model.py      # OpenViVQAModel (main model)
│   └── modules/
│       ├── attention.py        # LayerNorm, FC, MLP, AoA, SAoA, GAoA, SGAoA
│       ├── ocr_consformer.py   # OCREncoder, GroupAttention
│       ├── ocr_encoder_feature.py  # Vision_Encode_Ocr_Feature
│       ├── ocr_spatial.py      # SpatialCirclePosition, SemanticOCREmbedding
│       ├── qa_clip.py          # QACLIPEncoder, MMCLIPAttention
│       └── visual_search.py    # VisualSearch (ConvNeXtV2)
│
├── training/
│   ├── finetune.py             # Full finetune training loop (Seq2SeqTrainer)
│   ├── metrics.py              # compute_metrics(), BLEU/CIDEr evaluation
│   └── evaluate.py             # Evaluation / prediction pipeline
│
├── utils/
│   ├── misc.py                 # SET_SEED(), pick_consistent_indices()
│   ├── model_utils.py          # print_trainable_params(), safe_download_weights()
│   ├── visualization.py        # OCR box visualization, sample display
│   └── debug_tools.py          # Collator inspection, dummy batch generation
│
└── scripts/
    ├── prepare_dataset.py      # Download & prepare datasets
    └── init_model.py           # Download backbone weights, initialize model
```

## Quick Start

### 1. Setup environment (Linux / Kaggle / Colab)
```bash
bash setup.sh
```

### 2. Prepare dataset
```bash
python scripts/prepare_dataset.py
```

### 3. Initialize & check model
```bash
python scripts/init_model.py
```

### 4. Finetune
```bash
python training/finetune.py
```

## Key Dependencies
| Package | Version |
|---------|---------|
| transformers | 4.45.2 |
| peft | 0.13.1 |
| accelerate | 0.34.2 |
| torch | ≥2.0.0 |

## Notes
- **OCR features** are pre-computed `.npy` files. See `data/ocr_utils.py` for the loading format.
- **Term vocabulary** for TWC augmentation should be placed at `term_vocab_path` in config.
- Ablation switches: set `ablation_use_qaclip`, `ablation_use_vs`, `ablation_use_ocr` in `OpenViVQAConfig` to toggle individual modules.

## Bug Fixes vs Notebook
- `_encode_ocr_features` was accidentally dedented to module scope in the original notebook — fixed as a proper `OpenViVQAModel` method in `models/openvivqa_model.py`.
