"""Weight check of the training losses: gradient of each weighted term per parameter group.

The total loss is 0.5 (L_orig + L_rel) + w_g L_ground + w_s L_span + w_c L_ctr (+ w_a L_adv-reg, with the
answer loss then the mean over three views) as in trainer.compute_loss.
What a weight decides is not the loss value but the gradient on the parameters that the answer
loss also trains: AdamW rescales every parameter on its own, so on a parameter that only an
auxiliary loss reaches (dove_projection, the QA-ViT adapter behind a closed gate) the weight has
almost no effect, while on a shared one (ViT5 encoder, OCR embeddings) w * g_aux and g_answer are
added before the rescaling, and one can drown the other. Global gradient clipping (max_grad_norm)
also shrinks the answer gradient when an auxiliary term is large.

For each term t and group G it prints ||w_t grad_G L_t|| / ||grad_G L_answer|| and the cosine of
the two gradients (negative = the term pulls against the answer). Same configuration as a run:
the env variables of the notebook cell 3 (run_pipeline.training_arguments). CKPT = a checkpoint
directory (default: the initial model, step 0). BATCHES = number of train batches (default 8).

Usage on Colab:  CKPT=output/.../checkpoint-5000 BATCHES=8 python scripts/loss_balance.py
"""
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import HfArgumentParser, set_seed  # noqa: E402
from transformers.trainer_pt_utils import LabelSmoother  # noqa: E402

from configs.arguments import CustomTrainingArguments, DataArguments, ModelArguments  # noqa: E402
from data.dataset import ViT5VQADataset  # noqa: E402
from run_pipeline import training_arguments  # noqa: E402
from training.common import (build_collator, build_corrector, load_dataframes,  # noqa: E402
                             load_model_and_tokenizer, parse_args_with_yaml_and_cli)
from training.finetune import ARCHITECTURE_FLAGS  # noqa: E402

GROUPS = (("encoder", ("vit5.encoder", "vit5.shared")),
          ("decoder", ("vit5.decoder", "vit5.lm_head")),
          ("ocr", ("ocr_encoder", "semantic_ocr_embedding", "char_embedding", "char_position_embedding",
                   "ocr_char_layernorm", "ocr_lite_text", "ocr_visual")),
          ("qavit", ("qa_clip",)),
          ("dove_head", ("dove_projection",)))


def group_of(name):
    for group, prefixes in GROUPS:
        if name.startswith(prefixes):
            return group
    return "other"


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parse_args_with_yaml_and_cli(parser, training_arguments())
    set_seed(training_args.seed)
    checkpoint = os.environ.get("CKPT") or None
    flags = {name: getattr(model_args, name) for name in ARCHITECTURE_FLAGS if getattr(model_args, name) is not None}
    model, tokenizer = load_model_and_tokenizer(checkpoint, overrides=flags)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    config = model.config
    frames = load_dataframes(data_args.dataset_name, data_args.data_dir)
    corrector = build_corrector(config, frames["train"], data_args.term_vocab_path,
                                data_args.viet_vocab_path) if config.use_twc else None
    loader = DataLoader(ViT5VQADataset(frames["train"]), batch_size=training_args.per_device_train_batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(training_args.seed),
                        collate_fn=build_collator(model, tokenizer, corrector, training=True))
    smoother = LabelSmoother(epsilon=training_args.label_smoothing_factor) \
        if training_args.label_smoothing_factor > 0 else None
    weights = {"ground_loss": config.qavit_ground_weight, "span_loss": config.consformer_span_weight,
               "dove_loss": config.dove_contrastive_weight, "twc_loss": config.twc_weight,
               "adv_reg": config.dove_adv_weight}
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = sorted({group_of(name) for name, _ in params})
    print("checkpoint:", checkpoint or "initial model", "| batch", training_args.per_device_train_batch_size,
          "| label smoothing", training_args.label_smoothing_factor, "| max_grad_norm", training_args.max_grad_norm)
    print("weights:", {k: v for k, v in weights.items()})
    sums, values, count = {}, {}, 0
    batches = int(os.environ.get("BATCHES", "8"))
    for step, batch in enumerate(loader):
        if step == batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(**batch)
        labels = batch["labels"]
        answer = smoother(outputs, labels) if smoother else outputs["loss"]
        if "view_logits" in outputs:
            view = smoother({"logits": outputs["view_logits"]}, labels) if smoother else outputs["view_loss"]
            if "adv_logits" in outputs:  # as trainer.compute_loss: mean over the passes
                adv = torch.stack([smoother({"logits": x}, labels) if smoother else torch.nn.functional.cross_entropy(
                    x.reshape(-1, x.shape[-1]).float(), labels.reshape(-1), ignore_index=-100)
                    for x in outputs["adv_logits"]])
                adv = [adv.mean()] if config.dove_adv == "pgd" else list(adv)  # as trainer.compute_loss
                answer = torch.stack([answer, view, *adv]).mean()
            else:
                answer = 0.5 * (answer + view)
        terms = {"answer": answer}
        terms.update({key: weights[key] * outputs[key] for key in weights if key in outputs})
        answer_grads, summed = None, None
        for index, (key, term) in enumerate(terms.items()):
            g = torch.autograd.grad(term, [p for _, p in params], retain_graph=index < len(terms) - 1,
                                    allow_unused=True)
            flat = {group: torch.cat([x.flatten().float() for (n, _), x in zip(params, g)
                                      if x is not None and group_of(n) == group] or [torch.zeros(1, device=device)])
                    for group in groups}
            del g
            values[key] = values.get(key, 0.0) + float(term.detach())
            if answer_grads is None:
                answer_grads = flat
                summed = {group: v.clone() for group, v in flat.items()}
            else:
                for group in groups:
                    if summed[group].shape == flat[group].shape:
                        summed[group] += flat[group]
            for group in groups:
                a, v = answer_grads[group], flat[group]
                same = a.shape == v.shape
                entry = sums.setdefault((key, group), [0.0, 0.0, 0.0])
                entry[0] += float(v.norm())
                entry[1] += float(v.norm() / a.norm().clamp_min(1e-12))
                entry[2] += float(torch.nn.functional.cosine_similarity(a, v, dim=0)) if same and key != "answer" else 1.0
            if key != "answer":
                del flat
        total = sum(float(v.pow(2).sum()) for v in summed.values()) ** 0.5
        sums.setdefault(("_total", "all"), [0.0, 0.0, 0.0])[0] += total
        count += 1
        del outputs, answer_grads, summed
    print(f"\n{count} batches. Weighted loss values (mean):")
    for key, value in values.items():
        print(f"  {key:12s} {value / count:.4f}")
    print("\n||w*grad|| per group, ratio to the answer gradient, cosine with it (mean over batches):")
    print(f"  {'term':12s} {'group':10s} {'norm':>10s} {'/answer':>9s} {'cos':>7s}")
    for (key, group), (norm, ratio, cosine) in sorted(sums.items()):
        if key == "_total":
            continue
        if norm / count < 1e-10:
            continue
        print(f"  {key:12s} {group:10s} {norm / count:10.4g} {ratio / count:9.3f} {cosine / count:7.3f}")
    print(f"\nnorm of the total gradient (clipped to max_grad_norm when above it): {sums[('_total', 'all')][0] / count:.4g}")


if __name__ == "__main__":
    main()
