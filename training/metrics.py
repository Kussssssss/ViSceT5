"""Answer-generation metrics: EM, token F1, BLEU and CIDEr."""

import collections
import unicodedata
from typing import List

import numpy as np
import torch


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


def java_available():
    import shutil

    return shutil.which("java") is not None


def build_compute_metrics(tokenizer_for_metrics):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    bleu_metric = Bleu()
    cider_metric = Cider()
    # PTBTokenizer needs Java; without it (setup.sh may skip the install) report
    # EM/F1 only instead of crashing every evaluation.
    ptb_tokenizer = PTBTokenizer() if java_available() else None
    if ptb_tokenizer is None:
        print("[metrics] no 'java' on PATH -> BLEU/CIDEr disabled, EM/F1 only.")

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
        if ptb_tokenizer is None:
            return {"em": float(em), "f1": float(f1), "em_f1": 0.5 * (float(em) + float(f1))}

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
            # Tiêu chí chọn checkpoint cho ViTextVQA/ViOCRVQA (xem training/finetune.py).
            "em_f1": 0.5 * (float(em) + float(f1)),
        }
        return res

    return compute_metrics
