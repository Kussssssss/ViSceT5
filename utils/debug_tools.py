"""Inspect VQA batches, generated answers and visual-search crops."""

import numpy as np
import torch


def print_parameter_summary(model):
    """Count shared parameters once; learnable means requires_grad=True."""
    total = learnable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            learnable += parameter.numel()
    frozen = total - learnable
    percentage = 100 * learnable / total if total else 0.0
    print(
        "Model parameters:\n"
        f"  Total:     {total:,}\n"
        f"  Frozen:    {frozen:,}\n"
        f"  Learnable: {learnable:,} ({percentage:.2f}%)",
        flush=True,
    )


def inspect_batch(batch, tokenizer):
    for name, value in batch.items():
        if torch.is_tensor(value):
            print(f"{name}: shape={tuple(value.shape)} dtype={value.dtype}")
    print(
        "Questions:",
        tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True),
    )
    if "labels" in batch:
        labels = batch["labels"].masked_fill(
            batch["labels"] == -100, tokenizer.pad_token_id
        )
        print("Answers:", tokenizer.batch_decode(labels, skip_special_tokens=True))
    for index, info in enumerate(batch["ocr_info"]):
        print(f"OCR sample {index}:", info["texts"])


@torch.inference_mode()
def visualize_samples(model, dataset, collator, tokenizer, num_samples=5, seed=42):
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        indices = np.random.default_rng(seed).choice(
            len(dataset),
            size=min(num_samples, len(dataset)),
            replace=False,
        )
        for index in indices:
            sample = dataset[int(index)]
            batch = collator([sample])
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
                if key != "labels"
            }
            output = model(**batch, return_visual_search_debug=True)
            ids = model.generate(
                encoder_outputs=output["encoder_outputs"],
                attention_mask=output["attention_mask"],
            )
            print("Question:", sample["question"])
            print("Reference:", sample.get("answer", ""))
            print(
                "Prediction:", tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            )
            visual = output.get("vs_debug", {})
            if "attn_grids" not in visual:
                continue
            image = batch["pil_images"][0]
            heat = model.visual_search._heat_on_original(image, visual["attn_grids"][0])
            overlay = Image.blend(image, heat, alpha=0.5)
            box = model.visual_search._map_boxes_to_original(
                visual["boxes_224"], [image]
            )[0]
            ImageDraw.Draw(overlay).rectangle(box, outline="lime", width=3)
            figure, axes = plt.subplots(1, 2, figsize=(12, 6))
            for axis, picture, title in zip(
                axes, (image, overlay), ("Image", "Attention + crop")
            ):
                axis.imshow(picture)
                axis.set_title(title)
                axis.axis("off")
            figure.tight_layout()
            plt.show()
            plt.close(figure)
    finally:
        model.train(was_training)
