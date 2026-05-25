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
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

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
    accuracies = np.asarray(preds, dtype=np.float32).reshape(-1)
    mean_acc = float(accuracies.mean()) if accuracies.size > 0 else 0.0
    return {"pretrain_acc": mean_acc}

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running_loss = 0.0
        self._running_acc = 0.0
        self._running_cnt = 0
        self._last_log_step = -1

    def _log_pretrain_metrics(self, loss, inputs, outputs):
        if not globals().get("DEBUG_TRAIN", False):
            return

        with torch.no_grad():
            batch_acc = pretrain_acc_fn.calculate(inputs, outputs)
            batch_acc_val = batch_acc.item()
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

            print(
                f"[Pretrain] step={step_idx} | epoch={current_epoch:.3f} | "
                f"loss={avg_loss:.4f} | acc={avg_acc:.4f}"
            )
            self._running_loss = 0.0
            self._running_acc = 0.0
            self._running_cnt = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        if "tag_pollute" in inputs and inputs["tag_pollute"].ndim > 1:
            inputs["tag_pollute"] = inputs["tag_pollute"].squeeze(-1)

        outputs = model(**inputs)

        if getattr(model, "pretrain", False):
            loss = pretrain_loss_fn(inputs, outputs)
            self._log_pretrain_metrics(loss, inputs, outputs)

            if globals().get("DEBUG_TRAIN", False) and (self.state.global_step % 500 == 0):
                 if self.state.global_step != self._last_log_step:
                    self._last_log_step = self.state.global_step
                    with torch.no_grad():
                        acc_val = pretrain_acc_fn.calculate(inputs, outputs).item()
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
                outputs = model(**inputs)
                loss = pretrain_loss_fn(inputs, outputs)
                acc_tensor = pretrain_acc_fn.calculate(inputs, outputs)

                if prediction_loss_only:
                    return (loss.detach(), None, None)

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
