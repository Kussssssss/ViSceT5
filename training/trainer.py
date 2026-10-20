"""Trainer integration for multimodal answer generation."""

import json
from pathlib import Path

import torch
import numpy as np
from transformers import Seq2SeqTrainer

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:
    from numpy.core.multiarray import _reconstruct


# Randomly initialised modules (everything else is pretrained ViT5 / CLIP).
NEW_MODULES = ("ocr_encoder", "semantic_ocr_embedding", "char_embedding", "char_position_embedding",
               "ocr_char_layernorm", "ocr_lite_text_ln", "ocr_lite_text_proj", "twc_answer_query")
SCALARS = ("twc_logit_scale", "twc_answer_logit_scale", "instruction_proj_gate")


class VQATrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer + TWC auxiliary loss.

    eval_data_collator: collator for evaluation/prediction. Training may use TWA
    augmentation for the related OCR copy, but evaluation must see the same
    deterministic correction as inference.
    """

    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator
        self._twc_sum = {}
        self._twc_count = 0

    def _with_eval_collator(self, build, dataset):
        if self.eval_data_collator is None:
            return build(dataset)
        train_collator, self.data_collator = self.data_collator, self.eval_data_collator
        try:
            return build(dataset)
        finally:
            self.data_collator = train_collator

    def get_eval_dataloader(self, eval_dataset=None):
        return self._with_eval_collator(super().get_eval_dataloader, eval_dataset)

    def get_test_dataloader(self, test_dataset):
        return self._with_eval_collator(super().get_test_dataloader, test_dataset)

    def _load_rng_state(self, checkpoint):
        # Transformers 4.45 saves NumPy's RNG array; recent torch restricts loading
        # it by default. Allow only the NumPy types used by that RNG state.
        if not hasattr(torch.serialization, "safe_globals"):
            return super()._load_rng_state(checkpoint)
        allowed = [_reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
        with torch.serialization.safe_globals(allowed):
            return super()._load_rng_state(checkpoint)

    def create_optimizer(self):
        """Three parameter groups (each keeps its own LR under the shared warmup/cosine schedule):
        pretrained (ViT5, CLIP adapter projections) = learning_rate; randomly initialised
        modules = new_module_learning_rate; scalars (TWC temperatures, adapter gates) =
        scalar_learning_rate without weight decay (as CLIP's logit scale)."""
        if self.optimizer is not None:
            return self.optimizer
        args = self.args
        rates = {
            "pretrained": args.learning_rate,
            "new": getattr(args, "new_module_learning_rate", None) or args.learning_rate,
            "scalar": getattr(args, "scalar_learning_rate", None) or args.learning_rate,
        }
        decay = set(self.get_decay_parameter_names(self.model))
        groups = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.split(".")[-1] in SCALARS:
                kind, weight_decay = "scalar", 0.0
            else:
                kind = "new" if name.split(".")[0] in NEW_MODULES else "pretrained"
                weight_decay = args.weight_decay if name in decay else 0.0
            group = groups.setdefault((kind, weight_decay), {"params": [], "lr": rates[kind],
                                                             "weight_decay": weight_decay})
            group["params"].append(parameter)
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(args, self.model)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(list(groups.values()), **optimizer_kwargs)
        if self.is_world_process_zero():
            summary = {f"{k}{'' if wd else ' (no decay)'}": f"{len(g['params'])} tensors @ lr {g['lr']:g}"
                       for (k, wd), g in groups.items()}
            print("[optimizer] parameter groups:", summary)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Training requires answer labels.")
        loss = (
            self.label_smoother(outputs, labels)
            if self.label_smoother
            else outputs["loss"]
        )
        config = getattr(model, "module", model).config
        terms = [(name, weight) for name, weight in (("twc", config.twc_weight),
                                                     ("twc_answer", config.twc_answer_weight))
                 if f"{name}_loss" in outputs]
        for name, weight in terms:
            loss = loss + weight * outputs[f"{name}_loss"]
            if model.training:
                for key in (f"{name}_loss", f"{name}_acc"):
                    self._twc_sum[key] = self._twc_sum.get(key, 0.0) + float(outputs[key].detach())
        if terms and model.training:
            self._twc_count += 1
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        with torch.no_grad(), self.compute_loss_context_manager():
            if labels is None:
                outputs, loss = model(**inputs), None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            if prediction_loss_only:
                return loss, None, None
            if not self.args.predict_with_generate:
                logits = outputs.get("logits")
                return loss, logits.detach() if logits is not None else None, labels

            generation = dict(getattr(self, "_gen_kwargs", {}))
            # Keep the project's convention: generation_max_length caps NEW tokens.
            max_length = generation.pop("max_length", self.args.generation_max_length)
            if "max_new_tokens" not in generation and max_length is not None:
                generation["max_new_tokens"] = max_length
            if self.args.generation_num_beams is not None:
                generation.setdefault("num_beams", self.args.generation_num_beams)
            tokens = model.generate(
                encoder_outputs=outputs["encoder_outputs"],
                attention_mask=outputs["attention_mask"],
                **generation,
            )
        return loss, tokens.detach(), labels.detach() if labels is not None else None

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        # Preserve the existing protection before gradient clipping/optimizer updates.
        bad = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                parameter.grad = torch.nan_to_num(
                    parameter.grad, nan=0.0, posinf=0.0, neginf=0.0
                )
                bad.append(name)
        if bad:
            print(
                f"Non-finite gradients zeroed at step {self.state.global_step}: {bad}"
            )
        return loss

    def log(self, logs):
        if self._twc_count and "loss" in logs:  # training log: mean TWC since last log
            logs.update({k: round(v / self._twc_count, 4) for k, v in self._twc_sum.items()})
            base = getattr(self.model, "module", self.model)
            for name, key in (("twc_logit_scale", "twc_tau"), ("twc_answer_logit_scale", "twc_answer_tau")):
                if hasattr(base, name):  # learned temperature τ = 1 / exp(clamped logit scale)
                    theta = float(getattr(base, name).detach().clamp(-4.6052, 4.6052))
                    logs[key] = round(float(torch.exp(torch.tensor(-theta))), 4)
            self._twc_sum = {}
            self._twc_count = 0
        super().log(logs)
        if self.is_world_process_zero():
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            entry = {**logs, "step": self.state.global_step, "epoch": self.state.epoch}
            with (output_dir / "train_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
