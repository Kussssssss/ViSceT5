"""
training/metrics.py
compute_metrics(), BLEU/CIDEr/ANLS evaluation helpers for Seq2SeqTrainer.
"""

import os
import gc
import shutil
import unicodedata
import random
import collections
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    set_seed,
)
from transformers.trainer_callback import ProgressCallback


class CleanProgressCallback(ProgressCallback):
    """Keep only the TRAIN progress bar + the eval RESULTS.

    Suppresses (a) the eval/prediction progress bar and (b) the per-step train
    loss/lr/grad_norm log spam. Eval metric dicts are still printed on their own
    line above the training bar.
    """
    def on_prediction_step(self, args, state, control, **kwargs):
        # No eval progress bar; announce once so a long eval isn't silent.
        if state.is_world_process_zero and not getattr(self, "_eval_announced", False):
            print("⏳ Running validation...", flush=True)
            self._eval_announced = True

    def on_evaluate(self, args, state, control, **kwargs):
        self._eval_announced = False  # reset for the next eval

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        logs = logs or {}
        # Only surface evaluation results; drop train-step logs (loss/lr/grad_norm).
        if not any(k.startswith("eval_") for k in logs):
            return
        msg = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in logs.items()
        )
        line = "✅ [eval] " + msg
        if getattr(self, "training_bar", None) is not None:
            self.training_bar.write(line)
        else:
            print(line, flush=True)


def seed_train_metrics_from_checkpoint(output_dir, checkpoint_dir):
    """On resume, prepend the checkpoint's HF log_history into the custom
    train_metrics files so they CONTINUE (append) instead of starting empty in a
    fresh session — i.e. resume adds to the history, never overwrites it.
    No-op if the local metrics file already has content (same-session resume)."""
    import os, json
    if not checkpoint_dir:
        return
    ts = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(ts):
        return
    jsonl = os.path.join(output_dir, "train_metrics.jsonl")
    summary = os.path.join(output_dir, "train_metrics_summary.log")
    if os.path.exists(jsonl) and os.path.getsize(jsonl) > 0:
        return  # history already present in this output_dir; append() will continue it
    try:
        hist = json.load(open(ts, encoding="utf-8")).get("log_history", [])
        os.makedirs(output_dir, exist_ok=True)
        with open(jsonl, "a", encoding="utf-8") as f:
            for e in hist:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"# ==== resumed; {len(hist)} prior log entries carried over from "
                    f"{os.path.basename(checkpoint_dir)} ====\n")
        print(f"↩️  Carried over {len(hist)} prior log entries from "
              f"{os.path.basename(checkpoint_dir)} (metrics continuity).")
    except Exception as e:
        print(f"ℹ️ Could not seed train metrics from checkpoint ({e}).")


from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from training.pretraine_loss import (
    ViT5PretrainLoss,
    PreTrainMLMAccuracy,
    PreTrainContraAccuracy,
    PreTrainTWCAccuracy,
    GlobalPretrainAccuracy,
)

pretrain_loss_fn = ViT5PretrainLoss()
pretrain_acc_fn = GlobalPretrainAccuracy(mode="all")

def dbg(*args, **kwargs):

    if globals().get("DEBUG_TRAIN", False):
        print(*args, **kwargs)

def _normalize_txt(x: str) -> str:
    if x is None:
        return ""
    x = unicodedata.normalize("NFC", x)
    x = x.strip().lower()
    x = " ".join(x.split())
    return x


def compute_f1_em(preds: List[str], labels: List[str]):
    f1_scores = []
    em_scores = []

    for p, l in zip(preds, labels):
        # đảm bảo đã được normalize
        p = _normalize_txt(p)
        l = _normalize_txt(l)

        p_toks = p.split()
        l_toks = l.split()

        # EM: trùng y nguyên string sau normalize
        em = 1.0 if p == l else 0.0

        # Trường hợp 1 bên rỗng
        if len(p_toks) == 0 or len(l_toks) == 0:
            f1_scores.append(0.0)
            em_scores.append(em)
            continue

        common = collections.Counter(p_toks) & collections.Counter(l_toks)
        num_same = sum(common.values())

        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(p_toks)
            recall = num_same / len(l_toks)
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)
        em_scores.append(em)

    return float(np.mean(f1_scores)), float(np.mean(em_scores))


def simple_pretrain_aggregator(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    
    if preds.ndim == 1 or preds.shape[-1] == 1:
        accuracies = np.asarray(preds, dtype=np.float32).reshape(-1)
        mean_acc = float(accuracies.mean()) if accuracies.size > 0 else 0.0
        return {"pretrain_acc": mean_acc}
    else:
        mean_vals = np.mean(preds, axis=0)
        result = {
            "pretrain_acc": float(mean_vals[0]),
            "acc_mlm_itm": float(mean_vals[1]),
            "acc_twc": float(mean_vals[2]),
            "loss_mlm": float(mean_vals[3]),
            "loss_itm": float(mean_vals[4]),
            "loss_twc": float(mean_vals[5]),
        }
        if mean_vals.shape[0] >= 8:
            result["twc_pos_recall"] = float(mean_vals[6])
            result["twc_neg_recall"] = float(mean_vals[7])
        if mean_vals.shape[0] >= 9:
            result["loss_gen"] = float(mean_vals[8])
        return result

def build_compute_metrics_finetune(tokenizer_for_metrics):
    bleu_metric = Bleu()
    cider_metric = Cider()
    ptb_tokenizer = PTBTokenizer()

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # lấy logits sinh ra (nếu predictions là tuple)
        if isinstance(predictions, tuple):
            gen_ids = predictions[0]
        else:
            gen_ids = predictions

        if torch.is_tensor(gen_ids):
            gen_ids = gen_ids.detach().cpu().to(torch.long).numpy()
        gen_ids = np.asarray(gen_ids, dtype=np.int64)

        pad_id = tokenizer_for_metrics.pad_token_id or 0
        gen_ids[gen_ids < 0] = pad_id

        if labels is None:
            labels = np.zeros_like(gen_ids)
        elif torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        labels_proc = labels.copy()
        labels_proc[labels_proc == -100] = pad_id

        # decode thô
        pred_texts = tokenizer_for_metrics.batch_decode(
            gen_ids, skip_special_tokens=True
        )
        label_texts = tokenizer_for_metrics.batch_decode(
            labels_proc, skip_special_tokens=True
        )

        # EM/F1 theo ViTextVQA
        f1, em = compute_f1_em(pred_texts, label_texts)

        # Chuẩn hoá cho BLEU / CIDEr
        pred_texts_n = [_normalize_txt(x) for x in pred_texts]
        label_texts_n = [_normalize_txt(x) for x in label_texts]

        gts = {i: [{"caption": label_texts_n[i]}] for i in range(len(label_texts_n))}
        gens = {i: [{"caption": pred_texts_n[i]}] for i in range(len(pred_texts_n))}
        gts_tok = ptb_tokenizer.tokenize(gts)
        gens_tok = ptb_tokenizer.tokenize(gens)

        bleu_scores, _ = bleu_metric.compute_score(gts_tok, gens_tok)
        cider_score, _ = cider_metric.compute_score(gts_tok, gens_tok)

        res = {
            "bleu1": float(bleu_scores[0]),
            "bleu2": float(bleu_scores[1]),
            "bleu3": float(bleu_scores[2]),
            "bleu4": float(bleu_scores[3]),
            "cider": float(cider_score),
            "em": float(em),
            "f1": float(f1),
        }
        dbg(f"[metrics] {res}")
        return res

    return compute_metrics

class TaskSpecificTrainer(Seq2SeqTrainer):
    def __init__(self, *args, pretrain_loss_fn=None, pretrain_acc_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain_loss_fn = pretrain_loss_fn
        self.pretrain_acc_fn = pretrain_acc_fn
        self._running_loss = 0.0
        self._running_acc = 0.0
        self._running_cnt = 0
        self._last_log_step = -1

    def training_step(self, model, inputs):
        # Detect weights that are ALREADY corrupted (means a previous step slipped
        # a bad update through — should not happen once the guard below works).
        corrupted = [n for n, p in model.named_parameters()
                     if not torch.isfinite(p).all()]
        if corrupted:
            print(f"🚨 [Trainer Check] step {self.state.global_step}: NaN/inf WEIGHTS already in {corrupted[:10]} "
                  f"(+{max(0,len(corrupted)-10)} more) — model is corrupted upstream.")

        loss = super().training_step(model, inputs)  # forward + backward (grads now set)

        if not torch.isfinite(loss):
            print(f"🚨 [Trainer Check] Loss is non-finite at step {self.state.global_step}: {loss.item()}")

        # CRITICAL GUARD. transformers 4.38/4.45 have no `_clip_grad_norm` hook and
        # clip gradients via accelerate AFTER this method returns. Their
        # clip_grad_norm_ turns an inf gradient into NaN (clip_coef = max_norm/inf
        # = 0, then inf*0 = NaN), which the optimizer then writes into the weights —
        # corrupting the whole model permanently (every later forward becomes NaN).
        # So sanitize non-finite grads HERE, before the optimizer step.
        bad = {}
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                pref = name.split(".")[0]
                bad[pref] = bad.get(pref, 0) + 1
        if bad:
            print(f"⚠️  [GradGuard] step {self.state.global_step}: sanitized non-finite grads "
                  f"(zeroed) in submodules {bad} — watch which one recurs to find the source.")

        return loss


    def log(self, logs: Dict[str, float]) -> None:
        super().log(logs)
        if self.args.output_dir:
            import json
            import os
            try:
                os.makedirs(self.args.output_dir, exist_ok=True)
                log_file = os.path.join(self.args.output_dir, "train_metrics.jsonl")
                log_entry = {**logs, "step": self.state.global_step, "epoch": self.state.epoch}
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                summary_file = os.path.join(self.args.output_dir, "train_metrics_summary.log")
                items = [f"Step: {self.state.global_step}", f"Epoch: {self.state.epoch:.3f}"]
                for k, v in logs.items():
                    if k in ("step", "epoch"):
                        continue
                    if isinstance(v, float):
                        items.append(f"{k}: {v:.6f}")
                    else:
                        items.append(f"{k}: {v}")
                summary_line = " | ".join(items)
                with open(summary_file, "a", encoding="utf-8") as f:
                    f.write(summary_line + "\n")
            except Exception:
                pass

    def _log_pretrain_metrics(self, loss, inputs, outputs):
        if not globals().get("DEBUG_TRAIN", False):
            return

        acc_fn = self.pretrain_acc_fn if self.pretrain_acc_fn is not None else pretrain_acc_fn

        with torch.no_grad():
            batch_acc = acc_fn.calculate(inputs, outputs)
            if isinstance(batch_acc, torch.Tensor):
                batch_acc_val = batch_acc[0].item() if batch_acc.ndim > 0 else batch_acc.item()
            else:
                batch_acc_val = float(batch_acc)
            loss_val = loss.item()

        self._running_loss += loss_val
        self._running_acc += batch_acc_val
        self._running_cnt += 1

        LOG_TRAIN_EVERY = globals().get("LOG_TRAIN_EVERY", 100)
        step_idx = self.state.global_step + 1

        if self._running_cnt > 0 and step_idx % LOG_TRAIN_EVERY == 0:
            avg_loss = self._running_loss / self._running_cnt
            avg_acc = self._running_acc / self._running_cnt
            current_epoch = self.state.epoch or 0.0

            if isinstance(batch_acc, torch.Tensor) and batch_acc.ndim > 0 and len(batch_acc) >= 6:
                mlm_itm_a, twc_a = batch_acc[1].item(), batch_acc[2].item()
                mlm_l, itm_l, twc_l = batch_acc[3].item(), batch_acc[4].item(), batch_acc[5].item()
                gen_l = batch_acc[8].item() if len(batch_acc) >= 9 else 0.0
                print(
                    f"[Pretrain] step={step_idx} | epoch={current_epoch:.3f} | "
                    f"Total Loss={avg_loss:.4f}, Acc={avg_acc:.4f} | "
                    f"Batch Detail -> Acc(M+I):{mlm_itm_a:.3f}, Acc(TWC):{twc_a:.3f} | "
                    f"Loss(M):{mlm_l:.3f}, Loss(I):{itm_l:.3f}, Loss(TWC):{twc_l:.3f}, Loss(GEN):{gen_l:.3f}"
                )
            else:
                print(
                    f"[Pretrain] step={step_idx} | epoch={current_epoch:.3f} | "
                    f"loss={avg_loss:.4f} | acc={avg_acc:.4f}"
                )
            self._running_loss = 0.0
            self._running_acc = 0.0
            self._running_cnt = 0

    def _pretrain_gen_loss(self, model, inputs):
        """Read-scene-text GENERATIVE loss for pretrain, computed WITHOUT modifying
        models/: reuse the model's EXISTING finetune forward path (question-only
        encoder + gen_labels) by temporarily flipping `pretrain`. This keeps the
        gen objective a pure PRETRAIN-METHOD concern. Returns an in-graph loss
        tensor, or None if the collator didn't emit gen targets (non-gen modes).
        """
        if inputs.get("gen_labels") is None or inputs.get("gen_input_ids") is None:
            return None
        base = model
        for _ in range(4):
            if hasattr(base, "module"):
                base = base.module
            else:
                break
        orig = getattr(base, "pretrain", False)
        base.pretrain = False
        try:
            gen_out = model(
                input_ids=inputs.get("gen_input_ids"),
                attention_mask=inputs.get("gen_attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                pil_images=inputs.get("pil_images"),
                ocr_info=inputs.get("ocr_info"),
                ocr_mask_token=inputs.get("ocr_mask_token"),
                ocr_mask_box=inputs.get("ocr_mask_box"),
                labels=inputs.get("gen_labels"),
                twa_ocr_char=inputs.get("twa_ocr_char"),
                twa_ocr_char_mask=inputs.get("twa_ocr_char_mask"),
                twa_word_ids=inputs.get("twa_word_ids"),
                ocr_to_word_map=inputs.get("ocr_to_word_map"),
            )
        finally:
            base.pretrain = orig
        return gen_out.get("loss") if isinstance(gen_out, dict) else None

    def compute_loss(self, model, inputs, return_outputs=False):
        if "tag_pollute" in inputs and inputs["tag_pollute"].ndim > 1:
            inputs["tag_pollute"] = inputs["tag_pollute"].squeeze(-1)

        outputs = model(**inputs)

        if getattr(model, "pretrain", False):
            loss_fn = self.pretrain_loss_fn if self.pretrain_loss_fn is not None else pretrain_loss_fn
            acc_fn = self.pretrain_acc_fn if self.pretrain_acc_fn is not None else pretrain_acc_fn
            gen_loss = self._pretrain_gen_loss(model, inputs)
            if gen_loss is not None:
                outputs["gen_loss"] = gen_loss
            loss = loss_fn(inputs, outputs)
            self._log_pretrain_metrics(loss, inputs, outputs)

            if globals().get("DEBUG_TRAIN", False) and (self.state.global_step % 500 == 0):
                 if self.state.global_step != self._last_log_step:
                    self._last_log_step = self.state.global_step
                    with torch.no_grad():
                        acc_val = acc_fn.calculate(inputs, outputs)
                        if isinstance(acc_val, torch.Tensor):
                            acc_val = acc_val[0].item() if acc_val.ndim > 0 else acc_val.item()
                        else:
                            acc_val = float(acc_val)
                        print(f"Debug Step {self.state.global_step}: Loss={loss.item():.4f}, Acc={acc_val:.4f}")
        else:
            loss = outputs.get("loss")
            if loss is None:
                logits = outputs.get("logits")
                labels = inputs.get("labels")
                if logits is not None and labels is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )

        return loss if not return_outputs else (loss, outputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            # Chuẩn hoá inputs
            inputs = self._prepare_inputs(inputs)
            if "tag_pollute" in inputs and isinstance(inputs["tag_pollute"], torch.Tensor) and inputs["tag_pollute"].ndim > 1:
                inputs["tag_pollute"] = inputs["tag_pollute"].squeeze(-1)

            # 1) PRETRAIN BRANCH
            if getattr(model, "pretrain", False):
                loss_fn = self.pretrain_loss_fn if self.pretrain_loss_fn is not None else pretrain_loss_fn
                acc_fn = self.pretrain_acc_fn if self.pretrain_acc_fn is not None else pretrain_acc_fn
                outputs = model(**inputs)
                gen_loss = self._pretrain_gen_loss(model, inputs)
                if gen_loss is not None:
                    outputs["gen_loss"] = gen_loss
                loss = loss_fn(inputs, outputs)
                acc_tensor = acc_fn.calculate(inputs, outputs)

                if prediction_loss_only:
                    return (loss.detach(), None, None)

                if not isinstance(acc_tensor, torch.Tensor):
                    acc_tensor = torch.tensor(acc_tensor, device=loss.device)

                preds = acc_tensor.unsqueeze(0).detach()
                labels = inputs.get("tag_pollute")
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach()
                else:
                    labels = None
                return (loss.detach(), preds, labels)

            # 2) FINETUNE BRANCH
            outputs = model(**inputs)
            loss = outputs.get("loss")

            if loss is None:
                logits_ce = outputs.get("logits")
                labels_ce = inputs.get("labels")
                if logits_ce is not None and labels_ce is not None:
                    loss = F.cross_entropy(
                        logits_ce.view(-1, logits_ce.size(-1)),
                        labels_ce.view(-1),
                        ignore_index=-100
                    )
                else:
                    loss = torch.tensor(0.0, device=next(model.parameters()).device)

            if prediction_loss_only and not self.args.predict_with_generate:
                return (loss.detach(), None, None)

            labels = inputs.get("labels")
            if labels is not None:
                labels = labels.detach()

            # 2.a) Finetune, KHÔNG generate
            if not self.args.predict_with_generate:
                logits = outputs.get("logits")
                if logits is not None:
                    logits = logits.detach()
                return (loss.detach(), logits, labels)

            # 2.b) Finetune, CÓ generate (ViTextVQA)
            gen_kwargs = {
                "max_new_tokens": self.args.generation_max_length,
                "num_beams": self.args.generation_num_beams,
                "input_ids": inputs.get("input_ids"),
                "attention_mask": inputs.get("attention_mask"),
                "pixel_values": inputs.get("pixel_values"),
                "ocr_info": inputs.get("ocr_info"),
                "ocr_mask_token": inputs.get("ocr_mask_token"),
                "ocr_mask_box": inputs.get("ocr_mask_box"),
                "twa_ocr_char": inputs.get("twa_ocr_char"),
                "twa_ocr_char_mask": inputs.get("twa_ocr_char_mask"),
                "twa_word_ids": inputs.get("twa_word_ids"),
                "ocr_to_word_map": inputs.get("ocr_to_word_map"),
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            generated_tokens = model.generate(**gen_kwargs)

            if generated_tokens.shape[-1] < self.args.generation_max_length:
                pad_len = self.args.generation_max_length - generated_tokens.shape[-1]
                generated_tokens = F.pad(
                    generated_tokens,
                    (0, pad_len),
                    value=self.tokenizer.pad_token_id,
                )

            if prediction_loss_only:
                return (loss.detach(), None, None)

            return (loss.detach(), generated_tokens.detach(), labels)

def get_model_fingerprint(model) -> Dict[str, float]:
    fps: Dict[str, float] = {}

    total = 0.0
    for p in model.parameters():
        total += p.detach().float().cpu().sum().item()
    fps["__all_params_sum__"] = float(total)

    if hasattr(model, "char_embedding") and hasattr(model.char_embedding, "weight"):
        fps["char_embedding"] = float(model.char_embedding.weight.detach().cpu().sum().item())

    if hasattr(model, "char_position_embedding") and hasattr(model.char_position_embedding, "weight"):
        fps["char_position_embedding"] = float(model.char_position_embedding.weight.detach().cpu().sum().item())

    if hasattr(model, "logit_scale"):
        ls = model.logit_scale
        if isinstance(ls, torch.nn.Parameter):
            ls = ls.data
        fps["logit_scale"] = float(ls.detach().cpu().sum().item())

    if hasattr(model, "ocr_char_layernorm"):
        s = 0.0
        for p in model.ocr_char_layernorm.parameters():
            s += p.detach().float().cpu().sum().item()
        fps["ocr_char_layernorm"] = float(s)

    if hasattr(model, "ocr_encoder"):
        s = 0.0
        for p in model.ocr_encoder.parameters():
            s += p.detach().float().cpu().sum().item()
        fps["ocr_encoder"] = float(s)

    if hasattr(model, "pollute_head"):
        s = 0.0
        for p in model.pollute_head.parameters():
            s += p.detach().float().cpu().sum().item()
        fps["pollute_head"] = float(s)

    if hasattr(model, "qa_clip") and hasattr(model.qa_clip, "vision_model"):
        s = 0.0
        for p in model.qa_clip.vision_model.parameters():
            s += p.detach().float().cpu().sum().item()
        fps["qa_clip.vision_model"] = float(s)

    if hasattr(model, "semantic_ocr_embedding"):
        s = 0.0
        for p in model.semantic_ocr_embedding.parameters():
            s += p.detach().float().cpu().sum().item()
        fps["semantic_ocr_embedding"] = float(s)

    if hasattr(model, "spatial_embedding"):
        s = 0.0
        for p in model.spatial_embedding.parameters():
            s += p.detach().float().cpu().sum().item()
        fps["spatial_embedding"] = float(s)

    if hasattr(model, "visual_search") and hasattr(model.visual_search, "cnn"):
        s = 0.0
        for p in model.visual_search.cnn.parameters():
            s += p.detach().float().cpu().sum().item()
        fps["visual_search.cnn"] = float(s)

    if hasattr(model, "vit5"):
        if hasattr(model.vit5, "decoder"):
            s = 0.0
            for p in model.vit5.decoder.parameters():
                s += p.detach().float().cpu().sum().item()
            fps["vit5.decoder"] = float(s)

        if hasattr(model.vit5, "encoder"):
            s = 0.0
            for p in model.vit5.encoder.parameters():
                s += p.detach().float().cpu().sum().item()
            fps["vit5.encoder"] = float(s)

        if hasattr(model.vit5, "shared"):
            s = model.vit5.shared.weight.detach().float().cpu().sum().item()
            fps["vit5.shared"] = float(s)

    return fps

def print_consistency_check(fp_ref, fp_new, title="CHECKPOINT CONSISTENCY"):
    print("\n" + "=" * 60)
    print(f"🔍 {title}")
    print("=" * 60)
    all_match = True
    print(f"{'COMPONENT':<30} | {'REFERENCE':<12} | {'RELOADED':<12} | {'STATUS'}")
    print("-" * 75)

    for key in fp_ref:
        if key not in fp_new:
            print(f"{key:<30} | {fp_ref[key]:<12.4f} | {'MISSING':<12} | ❌ FAIL")
            all_match = False
            continue

        val1 = fp_ref[key]
        val2 = fp_new[key]
        is_match = np.isclose(val1, val2, atol=1e-5)
        status = "✅ PASS" if is_match else "❌ FAIL"
        print(f"{key:<30} | {val1:<12.4f} | {val2:<12.4f} | {status}")
        if not is_match:
            all_match = False

    print("-" * 75)
    if all_match:
        print(">>> SUCCESS: Checkpoint loaded perfectly! Logic is CONSISTENT.")
    else:
        print(">>> WARNING: Mismatch detected! Check your checkpoint loading logic.")
    print("=" * 60 + "\n")
