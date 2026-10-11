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


class VQATrainer(Seq2SeqTrainer):
    def _load_rng_state(self, checkpoint):
        # Transformers 4.45 saves NumPy's RNG array; recent torch restricts loading
        # it by default. Allow only the NumPy types used by that RNG state.
        if not hasattr(torch.serialization, "safe_globals"):
            return super()._load_rng_state(checkpoint)
        allowed = [_reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
        with torch.serialization.safe_globals(allowed):
            return super()._load_rng_state(checkpoint)

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
        super().log(logs)
        if self.is_world_process_zero():
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            entry = {**logs, "step": self.state.global_step, "epoch": self.state.epoch}
            with (output_dir / "train_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
