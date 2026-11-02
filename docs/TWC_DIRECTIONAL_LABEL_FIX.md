# Directional OCR pair labels

For an original view O and an augmented/related view R, the soft target belongs
to the R participating in the comparison. When two QA samples reuse an image
and OCR text, their R views may have different augmentation outcomes.

Example: O_A = O_B = "sữa", R_A = "sửa" (0.9), R_B = "sữa" (1.0).
The O-to-R targets are [[0.9, 1.0], [0.9, 1.0]], while the R-to-O targets are
[[0.9, 0.9], [1.0, 1.0]]. Previously both directions inherited by column,
giving the wrong cross-sample R-to-O targets.

The fix applies both when constructing labels for repeated original OCR text
inside a sample and when inheriting labels across QA samples in a batch.
It preserves group identity (image path + original text), negative and ignored
pair rules, temperature, objective, architecture, and the restored ablation
switches. It does not add augmented-to-augmented comparisons or entry IDs.

The notebook and launcher now select `tmp2`, so the historical branch reference
does not bypass this correction. TWC-enabled notebook run names include
`r2ofix` to keep their checkpoints separate from earlier runs.
