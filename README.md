# ViSceT5

Vietnamese scene-text visual question answering with ViT5, question-conditioned CLIP,
and OCR Consformer features.

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

With `model_name_or_path: null`, the model initializes its ViT5 and CLIP
backbones from their configured weights. Use `--model_name_or_path PATH` to start
from a VQA checkpoint with a fresh optimizer. Use `--resume_from_checkpoint PATH`
to resume a complete Trainer checkpoint with optimizer/scheduler state.

Weights from older checkpoints with Visual Search can be loaded with
`--model_name_or_path PATH`; the removed ConvNeXt weights are unused. Start a fresh
optimizer after this architecture change instead of resuming the old optimizer state.
On this research branch, baseline checkpoints also initialize the new joint-attention
layers and visual OCR norms; the old separate det/rec/box norm weights are unused.

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

The full OCR branch first sums the independent Linear projections of det, rec and
bbox features **at OCR entry level**, then applies `nn.RMSNorm`. The sequence
`[QA-ViT image patches, fused OCR regions]` passes through three independent joint
self-attention layers. Each uses `x + MHA(RMSNorm(x))`, ViT5's head count, and no
FFN/dropout. OCR padding is excluded from keys/values and zeroed after every layer.
The updated OCR regions pass through T5LayerNorm, then `ocr_to_word_map` expands
them to subwords before adding the original normalized tok/char features. The
image patches and mean patch token both use the updated image features. The
OCR-lite baseline bypasses this interaction and keeps its previous behavior.

Each joint-attention layer now adds a learned geometry bias only to image→OCR
and OCR→image logits. Normalized image patch boxes follow CLIP's row-major grid;
OCR boxes stay at entry level. Six pairwise values (center dx/dy, log width/height
ratios, IoU and query coverage) feed a `Linear(6,32) → ReLU → Linear(32,heads)`
MLP in each layer. Its last Linear starts at zero, preserving the previous
attention behavior when warm-starting from the same weights. Geometry and padding
share an additive attention mask; image→image/OCR→OCR get no geometry bias.
No additional experiment flags are needed: the configured full OCR branch uses it.
The geometry MLPs add 1,860 learnable parameters with 12 heads. Added masks can
change GPU kernel selection and activation memory; GPU VRAM must be measured.

The joint-attention baseline before geometry is commit `baa69d0`. Its checkpoint
can be used with `--model_name_or_path PATH` and a fresh optimizer. Only the new
geometry weights are missing on this warm-start; existing weights are retained.

The collator filters and normalizes the original OCR entries while retaining the
alignment between text, boxes, detection features, and recognition features.
It produces `ocr_input_ids`, `ocr_attention_mask`, `ocr_to_word_map`,
`ocr_char_ids`, `ocr_char_mask`, and `ocr_word_mask`, alongside image/question inputs
and optional answer `labels`. A batch must be entirely labelled or entirely
unlabelled.

`OpenViVQAModel.encode()` builds the fused encoder memory. `forward()` computes the
answer loss, and `generate()` can reuse that memory. `VQATrainer` performs evaluation
generation from the same encoder result and writes `train_metrics.jsonl`.

The fused sequence is `[question embeddings, updated image patches, OCR features, mean updated patch token]`.
Visual Search, ConvNeXt, ROI crops and crop tokens have been removed.
`models/modules/visual_search.py` is retained only as reference source and is not
imported by the active model or module package. Image and OCR features enter the
ViT5 encoder directly without modality projectors; their dimensions must match
ViT5 `d_model`.
`encode()` returns `(encoder_outputs, attention_mask)`. Image preprocessing follows
the CLIP backbone's image size, or the `image_size` override in the model config.

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
