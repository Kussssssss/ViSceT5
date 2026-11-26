# OCR feature ablations

The switches below apply to `ocr_full_features: true`. Text remains enabled and
character features stay aligned to text tokens. Spatial and visual mean bbox and
the detector/recognizer features, respectively; they do not mean CLIP image tokens.

| Scenario | `ocr_use_char` | `ocr_use_bbox` | `ocr_use_visual` | `ocr_align_spatial_visual` |
| --- | --- | --- | --- | --- |
| 1: text | false | false | false | true |
| 2: text + char | true | false | false | true |
| 3: text + char + bbox | true | true | false | true |
| 4: baseline | true | true | true | true |
| 5: positional fusion | true | true | true | false |

Pass the switches as CLI options to `training/finetune.py`, or use the equivalent
uppercase environment variables with `run_pipeline.py`. They are saved in the
model config. Scenario 5 pairs the nth word-level box/det/rec with the nth text
token; extra tokens get zero spatial/visual contributions after projection and
normalization. Character features remain aligned using the word map.

These switches implement the experiments on the current code. They do **not**
reproduce the existing ABL1 runs: the notebooks check out historical commits,
and ViTextVQA ABL1 differs from the other datasets' ABL1 in architecture and
training settings. Do not compare a run from current `main` directly against an
old ABL1 result as a controlled ablation. Reproduction requires porting and
validating the switches on each historical ABL1 commit, or rerunning the
corresponding baseline on the same code and settings as its new ablations.
