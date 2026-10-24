# ViSceT5

Vietnamese scene-text visual question answering with ViT5, question-conditioned CLIP,
and OCR Consformer features.

The training pipeline learns answer generation directly from image/question/answer
pairs. The configured loss combines decoder cross-entropy over answer tokens
(with label smoothing) and the optional TWC loss on ViT5 encoder outputs.

## Training

The current YAML and Colab notebook run the **no OCR augmentation / no TWC**
ablation. Image features and the full OCR ConsFormer (Origin text, characters,
boxes, detection and recognition features) remain enabled. YAML results go to
`output/finetune_no_aug_no_twc` to keep them separate from the augmented run.
To restore the full method, enable `ablation_use_ocr_aug` and `use_twc` in YAML,
or `USE_OCRAUG` and `USE_TWC` in the notebook.

Install dependencies with `bash setup.sh`, then run from the repository root:

```bash
python training/finetune.py configs/finetune.yaml
python training/finetune.py configs/finetune.yaml --smoke_test True
```

Dataset configuration lives in `configs/data/`. Training prepares the configured
dataset when its CSV cache is unavailable. Data preparation scripts, OCR features,
and vocabulary utilities are retained. OCR augmentation requires the term and
Vietnamese dictionary files in `configs/data/`; the lexicon also uses training Q&A.

With `model_name_or_path: null`, the model initializes its ViT5 and CLIP
backbones from their configured weights. Use `--model_name_or_path PATH` to start
from a VQA checkpoint with a fresh optimizer. Use `--resume_from_checkpoint PATH`
to resume a complete Trainer checkpoint with optimizer/scheduler state.

Weights from older checkpoints with Visual Search can be loaded with
`--model_name_or_path PATH`; the removed ConvNeXt weights are unused. Start a fresh
optimizer after this architecture change instead of resuming the old optimizer state.

Model checkpoints and the tokenizer are saved together in `output/finetune`;
the image processor is saved in its `image_processor/` subdirectory.

```bash
python training/evaluate.py --checkpoint output/finetune --split validation
python training/evaluate.py --checkpoint output/finetune --split test
```

Evaluation reports loss, EM, F1, BLEU, and CIDEr for labelled splits. An entirely
unlabelled split can be used for generation; answer-based metrics require references.

## Research modules

The two flags in `configs/finetune.yaml` select the existing ablation baselines:

| Flag set to `false` | Result |
| --- | --- |
| `ablation_use_qaclip` | CLIP without question instruction |
| `ablation_use_ocr` | Lightweight OCR text/character/box features |

The collator filters and normalizes the original OCR entries while retaining the
alignment between text, boxes, detection features, and recognition features.
It produces `ocr_input_ids`, `ocr_attention_mask`, `ocr_to_word_map`,
`ocr_char_ids`, `ocr_char_mask`, and `ocr_word_mask`, alongside image/question inputs
and optional answer `labels`. A batch must be entirely labelled or entirely
unlabelled.

`OpenViVQAModel.encode()` builds the fused encoder memory. `forward()` computes the
answer loss, and `generate()` can reuse that memory. `VQATrainer` performs evaluation
generation from the same encoder result and writes `train_metrics.jsonl`.

When augmentation is enabled, `ocr_aug_mode: separate` runs the complete Origin and Related
sequences independently through the same ConsFormer. It concatenates their text
features, then adds the aligned character/box/detection/recognition features.
Characters are rebuilt for Related; box/det/rec are reused from the original region.
The fused sequence is `[question embeddings, CLIP patches, all Origin features, all Related features]`.
Unchanged Related entries are retained. There is no lattice selection or type embedding.
Separate masks/maps handle different subword counts and padding in each half.

TWC pools the Origin/Related slices of the **same ViT5 encoder output** used by
the VQA decoder to OCR entry level, normalizes them, then computes the contrastive
loss. When enabled, TWC uses BCE, `twc_feature: fused`, `twc_view: shared`.
`ocr_related: augment` uses keep/correct/character-noise with probability **0.6**
at both train and test, reproducibly seeded per sample. No extra ViT5 pass is needed.
Temperature is fixed at `tau=0.07`: it has no
gradient and is excluded from the optimizer, including after loading a checkpoint.
The configured temperature overrides any previously learned checkpoint scale.
Use a fresh optimizer when warm-starting a run that learned temperature.
Legacy `concat` and `lattice` modes remain available to load old experiments.
To warm-start a lattice checkpoint into the new architecture, use a fresh optimizer;
the old type embedding is unused. New checkpoints save their mode for evaluation.
Visual Search, ConvNeXt, ROI crops and crop tokens have been removed.
`models/modules/visual_search.py` is retained only as reference source and is not
imported by the active model or module package. Image and OCR features enter the
ViT5 encoder directly without modality projectors; their dimensions must match
ViT5 `d_model`.
`encode()` returns `(encoder_outputs, attention_mask, ocr_offset)`. Image preprocessing follows
the CLIP backbone's image size, or the `image_size` override in the model config.

## Cloud runners

```bash
bash run_all.sh           # Vast.ai / Linux
bash run_all.sh mock      # smoke run
bash run_all_colab.sh     # Colab
```

The notebook `notebooks/twc_colab.ipynb` provides a VQA/TWC workflow with the same
separate OCR encoding, full concatenation and p=0.6.
`run_pipeline.py` accepts environment overrides `TRAIN_CONFIG`, `NUM_TRAIN_EPOCHS`,
`MODEL_NAME_OR_PATH`, `RESUME_FROM_CHECKPOINT`, `RESUME_CHECKPOINT_ID`, `MOCK_TEST`,
and `SMOKE_*`. Setting both `HF_TOKEN` and `HF_REPO` enables checkpoint upload;
Vast.ai shutdown remains enabled when its instance credentials are provided.

## Verification

```bash
pip install pytest
python -m pytest tests -q
```

Tests use small random backbones and synthetic images/OCR without downloading a
dataset or model weights. They verify collation, ablations, gradients, generation,
checkpoint round trips, and Trainer integration.
