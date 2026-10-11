# ViSceT5

Vietnamese scene-text visual question answering with ViT5, question-conditioned CLIP,
Visual Search (ConvNeXtV2 crops), and OCR Consformer features.

The training pipeline learns answer generation directly from image/question/answer
pairs. The loss is the decoder's cross-entropy over answer tokens (padding is ignored).

## Training

Install dependencies with `bash setup.sh`, then run from the repository root:

```bash
python training/finetune.py configs/finetune.yaml
python training/finetune.py configs/finetune.yaml --smoke_test True
```

Dataset configuration lives in `configs/data/`. Training prepares the configured
dataset when its CSV cache is unavailable. Data preparation scripts, OCR features,
and vocabulary utilities are retained; the VQA collator does not require auxiliary
dictionary files.

With `model_name_or_path: null`, the model initializes its ViT5, CLIP, and ConvNeXtV2
backbones from their configured weights. Use `--model_name_or_path PATH` to start
from a VQA checkpoint with a fresh optimizer. Use `--resume_from_checkpoint PATH`
to resume a complete Trainer checkpoint with optimizer/scheduler state.

Model checkpoints and the tokenizer are saved together in `output/finetune`;
the image processor is saved in its `image_processor/` subdirectory.

```bash
python training/evaluate.py --checkpoint output/finetune --split validation
python training/evaluate.py --checkpoint output/finetune --split test
```

Evaluation reports loss, EM, F1, BLEU, and CIDEr for labelled splits. An entirely
unlabelled split can be used for generation; answer-based metrics require references.

## Research modules

The three flags in `configs/finetune.yaml` select the existing ablation baselines:

| Flag set to `false` | Result |
| --- | --- |
| `ablation_use_qaclip` | CLIP without question instruction |
| `ablation_use_vs` | ConvNeXtV2 features from the full image |
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

## Cloud runners

```bash
bash run_all.sh           # Vast.ai / Linux
bash run_all.sh mock      # smoke run
bash run_all_colab.sh     # Colab
```

The notebook `notebooks/finetune_colab.ipynb` provides a direct VQA workflow.
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
