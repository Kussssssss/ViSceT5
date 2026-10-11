"""Profile the real Trainer on one reusable batch, without changing the model.

Example (run each revision in a fresh process):
    python scripts/profile_training_memory.py configs/finetune.yaml \
        --audit_batch_file /content/vqa-audit-batch.pt \
        --output_dir /content/vqa-memory-current

The batch file is created locally on the first run and reused on subsequent runs.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers
from transformers import HfArgumentParser, TrainerCallback, set_seed

from configs.arguments import CustomTrainingArguments, DataArguments, ModelArguments
from data.dataset import ViT5VQADataset
from training.common import (
    build_collator,
    load_dataframes,
    load_model_and_tokenizer,
    parse_args_with_yaml_and_cli,
)
from training.trainer import VQATrainer


@dataclass
class AuditArguments:
    audit_batch_file: str = field(default="./output/vqa-audit-batch.pt")
    audit_steps: int = field(default=3)


def tensor_metadata(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, dict):
        return {key: tensor_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_metadata(item) for item in value]
    return None


def parameter_report(model, optimizer=None):
    included = (
        {id(p) for group in optimizer.param_groups for p in group["params"]}
        if optimizer is not None
        else set()
    )
    groups, rows = {}, {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        state = optimizer.state.get(parameter, {}) if optimizer is not None else {}
        row = {
            **tensor_metadata(parameter),
            "numel": parameter.numel(),
            "parameter_bytes": parameter.numel() * parameter.element_size(),
            "in_optimizer": id(parameter) in included,
            "has_gradient": gradient is not None,
            "gradient_bytes": 0
            if gradient is None
            else gradient.numel() * gradient.element_size(),
            "optimizer_state_bytes": sum(
                t.numel() * t.element_size()
                for t in state.values()
                if isinstance(t, torch.Tensor)
            ),
        }
        rows[name] = row
        group = groups.setdefault(
            name.split(".")[0],
            {
                "total": 0,
                "trainable": 0,
                "with_gradient": 0,
                "in_optimizer": 0,
                "parameter_bytes": 0,
                "gradient_bytes": 0,
                "optimizer_state_bytes": 0,
            },
        )
        group["total"] += row["numel"]
        group["trainable"] += row["numel"] if parameter.requires_grad else 0
        group["with_gradient"] += row["numel"] if gradient is not None else 0
        group["in_optimizer"] += row["numel"] if row["in_optimizer"] else 0
        for key in ("parameter_bytes", "gradient_bytes", "optimizer_state_bytes"):
            group[key] += row[key]
    return {"groups": groups, "parameters": rows}


class MemoryTrace:
    def __init__(self, output_dir, device):
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.events = []
        self.microstep = 0
        self.shapes = {}

    def record(self, phase, **details):
        memory = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            memory = {
                "allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                "device_free_bytes": free,
                "device_total_bytes": total,
            }
        event = {"phase": phase, "microstep": self.microstep, "cuda": memory, **details}
        self.events.append(event)
        # Overwrite this run's trace, preserving every event collected so far.
        (self.directory / "memory_trace.json").write_text(
            json.dumps(self.events, indent=2), encoding="utf-8"
        )
        print("MEMORY_AUDIT", json.dumps(event), flush=True)

    def capture(self, name):
        def hook(module, inputs, kwargs, output):
            if name == "qa_clip":
                output = {
                    "hidden_states": output.last_hidden_state,
                    "attentions": output.attentions,
                }
            elif hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            elif name == "ocr_encoder":
                output = output[0]
            elif name == "visual_search":
                output = output["crop_tokens"]
            self.shapes.setdefault(name, []).append(tensor_metadata(output))

        return hook


class MemoryCallback(TrainerCallback):
    def __init__(self, trace):
        self.trace = trace

    def on_step_begin(self, args, state, control, **kwargs):
        if self.trace.device.type == "cuda":
            torch.cuda.synchronize(self.trace.device)
            torch.cuda.reset_peak_memory_stats(self.trace.device)
        self.trace.record("optimizer_step_begin", optimizer_step=state.global_step + 1)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.trace.record("before_optimizer", optimizer_step=state.global_step + 1)

    def on_optimizer_step(
        self, args, state, control, model=None, optimizer=None, **kwargs
    ):
        self.trace.record("after_optimizer", optimizer_step=state.global_step + 1)
        report = parameter_report(model, optimizer)
        (self.trace.directory / "parameters_after_optimizer.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


class MemoryTrainer(VQATrainer):
    def __init__(self, *args, trace, **kwargs):
        self.trace = trace
        super().__init__(*args, callbacks=[MemoryCallback(trace)], **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        self.trace.shapes = {}
        self.trace.record("before_forward")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        self.trace.record(
            "after_forward", shapes=self.trace.shapes, loss=float(loss.detach())
        )
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        self.trace.microstep += 1
        loss = super().training_step(model, inputs)
        self.trace.record("after_backward")
        return loss


def main(args_list=None):
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments, AuditArguments)
    )
    model_args, data_args, train_args, audit_args = parse_args_with_yaml_and_cli(
        parser, args_list
    )
    if audit_args.audit_steps < 2:
        raise ValueError(
            "Use at least two optimizer steps to include initialized Adam states."
        )
    if train_args.resume_from_checkpoint or train_args.resume_checkpoint_id:
        raise ValueError(
            "Audit fresh optimizer states; use model_name_or_path to load model weights."
        )
    set_seed(train_args.seed)
    model, tokenizer = load_model_and_tokenizer(model_args.model_name_or_path)
    for name in ("ablation_use_qaclip", "ablation_use_vs", "ablation_use_ocr"):
        setattr(model.config, name, getattr(model_args, name))
    batch_file = Path(audit_args.audit_batch_file)
    if batch_file.is_file():
        # This is the locally generated audit batch, including PIL images.
        batch = torch.load(batch_file, map_location="cpu", weights_only=False)
    else:
        frames = load_dataframes(data_args.dataset_name, data_args.data_dir)
        dataset = ViT5VQADataset(frames["train"])
        count = train_args.per_device_train_batch_size
        if len(dataset) < count:
            raise ValueError("The training split is smaller than the requested batch.")
        # Seeded sample selection, then save the complete collated tensors/mappings.
        indices = torch.randperm(len(dataset))[:count].tolist()
        batch = build_collator(model, tokenizer)([dataset[i] for i in indices])
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(batch, batch_file)
    batch_size = batch["input_ids"].shape[0]
    if batch_size != train_args.per_device_train_batch_size:
        raise ValueError("Saved batch size differs from per_device_train_batch_size.")
    if "labels" not in batch:
        raise ValueError("The audit batch must contain answer labels.")

    # Keep precision, optimizer, accumulation and model settings. Bound only this
    # diagnostic run; disable evaluation/checkpoint writes and repeat one batch.
    train_args.max_steps = audit_args.audit_steps
    train_args.eval_strategy = train_args.save_strategy = "no"
    train_args.load_best_model_at_end = False
    train_args.dataloader_num_workers = 0
    train_args.report_to = []
    train_args.disable_tqdm = True
    trace = MemoryTrace(train_args.output_dir, train_args.device)
    length = (
        batch_size * train_args.gradient_accumulation_steps * audit_args.audit_steps
    )
    trainer = MemoryTrainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=list(range(length)),
        data_collator=lambda unused: batch,
        trace=trace,
    )
    if train_args.world_size != 1 or train_args.n_gpu > 1:
        raise ValueError("Run this comparison in a single training process on one GPU.")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "git_revision": revision,
        "pid": os.getpid(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(train_args.device)
        if train_args.device.type == "cuda"
        else None,
        "model_config": model.config.to_dict(),
        "training_args": train_args.to_dict(),
        "backbone_configs": {
            "vit5": model.vit5.config.to_dict(),
            "qa_clip": model.qa_clip.config.to_dict(),
            "convnext": model.visual_search.cnn.config.to_dict(),
        },
        "source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
            ).hexdigest()
            for name in (
                "models/openvivqa_model.py",
                "models/modules/qa_clip.py",
                "models/modules/ocr_consformer.py",
                "models/modules/ocr_spatial.py",
                "models/modules/visual_search.py",
                "data/collator.py",
                "training/trainer.py",
            )
        },
        "batch_sha256": hashlib.sha256(batch_file.read_bytes()).hexdigest(),
        "batch": tensor_metadata(batch),
        "checkpointing": {
            name: module.gradient_checkpointing
            for name, module in model.named_modules()
            if hasattr(module, "gradient_checkpointing")
        },
        "parameters": parameter_report(model),
    }
    (trace.directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    hooks = [
        model.get_submodule(name).register_forward_hook(
            trace.capture(name), with_kwargs=True
        )
        for name in (
            "vit5.encoder",
            "vit5.decoder",
            "qa_clip",
            "visual_search",
            "ocr_encoder",
            "semantic_ocr_embedding",
        )
    ]
    trace.record("model_loaded")
    try:
        trainer.train()
    finally:
        for hook in hooks:
            hook.remove()
    return trace.directory


if __name__ == "__main__":
    main()
