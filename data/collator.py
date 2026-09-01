"""
data/collator.py
ViT5VQADataCollator — full TWC+MLM+ITM data collator.
"""
from PIL import ImageOps
import os
import re
import json
import collections
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import editdistance
import pandas as pd
import unicodedata
import numpy as np


from data.vocab import (
    COMBINED_CHARS, _CHAR_UNK_IDX, TONE_RANK,
    _normalize_text, _remove_vietnamese_accents,
    _non_alnum_ratio, _is_repeated_runs, _looks_like_code_garbage, _char_diversity_low,
    get_tone_id, get_word_tone_score, _stable_hash_int,
)
from data.dataset import ViT5VQADataset


# ---------------------------------------------------------------------------
# Grounded-cloze (OCR-grounded masked generation) — decoder pretrain objective.
# Mask the question words that ALSO appear in the OCR (exact match), then have
# the decoder regenerate the CLEAN question word (T5 sentinel format). To recover
# a proper-noun/scene-text word it cannot guess from the language prior, the
# decoder MUST read the OCR feature branch → warms the exact cross-attention
# pathway finetune reuses (attend OCR feature → emit its text). Target is the
# clean human question word, so NO OCR noise is ever learned.
# ---------------------------------------------------------------------------

# Vietnamese function/question words excluded from cloze span selection (they are
# recoverable from the language prior, so masking them does NOT force grounding).
# Keys are accent-stripped + lowercased.
_CLOZE_STOP: Set[str] = set(
    """la va cua co nhung cac nay do o cho voi duoc gi nao bao nhieu khi thi ma hay hoac
    mot cai con nguoi trong tren duoi ngoai day kia ai sao tai vi de da dang se bi do tu den
    ra vao len xuong khong cung nhu the boi rang thuoc bang qua lai chi nhe oi nha vay em anh
    chi ban toi ta ho no chung minh cua_no dau bao_nhieu the_nao""".split()
)

# ACCENTED stop list for the 'qa' cloze style. The accent-STRIPPED test above has
# harmful collisions: 'thuốc'→'thuoc'≈'thuộc', 'bảo'→'bao', 'bán/bàn'→'ban',
# 'nhà'→'nha' — i.e. the very scene-text/answer-like words grounding needs get
# excluded. Testing the accented core against an accented function-word list keeps
# real content words maskable. (sentinel style keeps the old list → resume-safe.)
_CLOZE_STOP_QA: Set[str] = set(
    """là và của có những các này đó ở cho với được gì nào bao nhiêu khi thì mà hay hoặc
    một cái con người trong trên dưới ngoài đây kia ai sao tại vì để đã đang sẽ bị do từ đến
    ra vào lên xuống không cũng như thế bởi rằng thuộc bằng qua lại chỉ nhé ơi vậy em anh
    chị bạn tôi ta họ nó chúng mình đâu""".split()
)

_CLOZE_EDGE_PUNCT = re.compile(r"^[^0-9a-zA-ZÀ-ỹ]+|[^0-9a-zA-ZÀ-ỹ]+$")


def _cloze_core(word: str) -> str:
    """Normalized, punctuation-stripped form used for OCR matching + stopword test."""
    nw = _normalize_text(word, lowercase=True).strip()
    return _CLOZE_EDGE_PUNCT.sub("", nw)


def _build_grounded_cloze(question: str, ocr_norm_tokens, max_spans: int = 8,
                          rand_prob: float = 0.15, seed: int = 0,
                          style: str = "sentinel"):
    """Return (masked_question, target, n_spans) — the SINGLE T5 span-infilling
    objective for the decoder (this replaces the old encoder-head MLM).

    Masking is BIASED toward grounding but always non-empty:
      • GROUNDED spans: content words that appear (exact-normalized) in the OCR →
        the decoder must READ the OCR feature branch to recover them.
      • RANDOM spans: a few other content words (T5-style span corruption) → keeps a
        language-model signal and guarantees ~100% coverage (grounded-only was ~52%).
    Consecutive masked words are MERGED into one span (T5-native): a run gets a single
    <extra_id_i> and the target emits "<extra_id_i> w0 w1 ...". Distinct sentinels per
    span (never one shared <extra_id_0>) so the decoder can address each blank.

    masked_question : question with each masked run replaced by one <extra_id_i>.
    target          : "<extra_id_0> ... <extra_id_1> ... <extra_id_k>" (clean words).
    n_spans         : number of spans (0 only if the question has no content word).
    span_types      : list[bool] per span — True = GROUNDED (contains an OCR-overlap
                      word), False = RANDOM. Used to split accuracy grounded vs random.

    style='qa' (v3, ANSWER-SHAPED cloze): mask exactly ONE span and the target is the
    RAW masked words (no sentinels) — i.e. the decoder emits a short content phrase
    then EOS, byte-identical to finetune's answer format. This kills the
    sentinel-format/verbosity transfer gap (A/B showed over-generation ratio ~1.13
    from sentinel-infill pretrain). Span priority: one GROUNDED run (consecutive
    OCR-overlap words, e.g. a multi-word brand); if the question has no OCR overlap,
    ONE random content word (LM signal). The grounding mechanism is unchanged —
    recovering the word still requires reading the OCR feature branch.
    """
    if not isinstance(question, str):
        return "", "", 0, []
    ocr_set: Set[str] = set()
    for t in ocr_norm_tokens:
        for sub in str(t).split():
            if len(sub) >= 2:
                ocr_set.add(sub)

    words = question.strip().split()
    cores = [_cloze_core(w) for w in words]
    if style == "qa":
        # accented stop test (see _CLOZE_STOP_QA) — no thuốc/bảo/bán collisions
        is_content = [len(c) >= 3 and c not in _CLOZE_STOP_QA for c in cores]
    else:
        is_content = [len(c) >= 3 and _remove_vietnamese_accents(c) not in _CLOZE_STOP for c in cores]

    to_mask = [False] * len(words)
    grounded = [False] * len(words)  # per-word: masked because it is in the OCR
    # 1) grounded: content words present in OCR
    for i in range(len(words)):
        if is_content[i] and cores[i] in ocr_set:
            to_mask[i] = True
            grounded[i] = True
    rng = random.Random(seed & 0xFFFFFFFF)

    if style == "qa":
        # ANSWER-SHAPED single-span cloze (see docstring). Grounded runs first.
        runs, i = [], 0
        while i < len(words):
            if grounded[i]:
                j = i
                while j + 1 < len(words) and grounded[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        if runs:
            s, e = runs[rng.randrange(len(runs))]
            g_span = True
        else:
            cidx = [k for k in range(len(words)) if is_content[k]]
            if not cidx:
                return question.strip(), "", 0, []
            s = e = rng.choice(cidx)
            g_span = False
        tgt_words = [_CLOZE_EDGE_PUNCT.sub("", w) for w in words[s:e + 1]]
        masked = words[:s] + ["<extra_id_0>"] + words[e + 1:]
        # RAW target — no sentinel, so the decoder's output distribution matches
        # finetune answers (short phrase + EOS).
        return " ".join(masked), " ".join(tgt_words), 1, [g_span]

    # 2) random: a few more content words (LM signal + full coverage)
    for i in range(len(words)):
        if (not to_mask[i]) and is_content[i] and rng.random() < rand_prob:
            to_mask[i] = True  # grounded stays False
    # 3) guarantee ≥1 span if any content word exists
    if not any(to_mask):
        cidx = [i for i in range(len(words)) if is_content[i]]
        if cidx:
            to_mask[rng.choice(cidx)] = True
    if not any(to_mask):
        return question.strip(), "", 0, []

    # Build input + target, MERGING consecutive masked words into one span. A merged
    # span is GROUNDED if any of its words is an OCR-overlap word.
    in_parts, tgt_parts, span_types, sid, i = [], [], [], 0, 0
    while i < len(words):
        if to_mask[i] and sid < max_spans:
            run, run_grounded = [], False
            while i < len(words) and to_mask[i]:
                run.append(_CLOZE_EDGE_PUNCT.sub("", words[i]))
                run_grounded = run_grounded or grounded[i]
                i += 1
            sent = f"<extra_id_{sid}>"
            in_parts.append(sent)
            tgt_parts.append(sent + " " + " ".join(run))
            span_types.append(run_grounded)
            sid += 1
        else:
            in_parts.append(words[i])
            i += 1
    if sid == 0:
        return question.strip(), "", 0, []
    target = " ".join(tgt_parts) + f" <extra_id_{sid}>"
    return " ".join(in_parts), target, sid, span_types

class ViT5VQADataCollator:
    def __init__(
        self,
        tokenizer,
        image_processor,
        ocr_encoder,
        config,
        term_vocab_path,
        viet_vocab_path,
        eng_vocab_path,
        dataframe,
        pretrain=True,
        debug=False,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.ocr_encoder = ocr_encoder

        # PHẢI có dòng này trước khi dùng self.cfg
        self.cfg = config

        self.pretrain = bool(pretrain)
        self.debug = bool(debug)

        self.pretrain_ablation_mode = str(
            getattr(self.cfg, "pretrain_ablation_mode", "full")
        ).lower().strip()

        self.use_ocr_aug_pretrain = self.pretrain_ablation_mode in [
            "full",
            "all",
            "only_twc_ocr_aug",
            "gen_all",
        ]

        self.use_ocr_aug_finetune = bool(
            getattr(self.cfg, "use_ocr_aug_finetune", False)
        )

        self.itm_history = []
        self.itm_history_max = 256

        self.seq_max = int(getattr(self.cfg, "ocr_max_scene_text", 180))

        if self.pretrain:
          self.txt_max_len = 128
        else:
          self.txt_max_len = int(getattr(self.cfg, "text_max_input_length", 32))

        # MLM masking granularity for pretrain:
        #   "wholeword" — decide masking at WORD level, then tokenize, so a multi-subword
        #                 OCR word (e.g. 'pepsi'->['▁p','ep','si']) is masked as ONE unit
        #                 (all its subwords) → removes the intra-word subword copy shortcut,
        #                 forcing recovery from OCR-feature/image/context (TWA-faithful).
        #   "subword"   — old per-subword BERT masking (kept for A/B ablation).
        self.mlm_mask_mode = str(getattr(self.cfg, "mlm_mask_mode", "wholeword")).lower().strip()

        # Whether the MLM encoder text branch INCLUDES the OCR tokens (concatenated to
        # the question). False (default) = QUESTION-ONLY → removes the OCR-as-text
        # "copy crutch" AND aligns the encoder input with finetune (which is also
        # question-only). OCR is still learned via the OCR-feature branch (gen +
        # TWC). True = old behaviour (question+OCR), kept for A/B.
        self.mlm_ocr_in_text = bool(getattr(self.cfg, "mlm_ocr_in_text", False))

        # Generative pretext for the decoder = READ-SCENE-TEXT (PreSTU-like): feed the
        # CLEAN OCR char/word + visual features, target = the OCR reading. Warms the
        # decoder + teaches read-from-features. (The old "denoise" correction pretext was
        # removed: its target is the noisy OCR output, indefensible without ground-truth;
        # revisit correction later with a real correct signal — synthetic-GT / multi-OCR.)

        self.tgt_max_len = int(getattr(self.cfg, "text_max_target_length", 56))
        self.char_max_num = int(getattr(self.cfg, "char_max_num", 50))

        self.pad_id = int(getattr(self.tokenizer, "pad_token_id", 0))
        self.eos_id = int(getattr(self.tokenizer, "eos_token_id", 1))
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        self.mask_prob = float(getattr(self.cfg, "pretrain_mask_prob", 0.15 if self.pretrain else 0.0))
        self.mask_seed = int(getattr(self.cfg, "pretrain_mask_seed", 42))

        self.lowercase = bool(getattr(self.cfg, "ocr_lowercase", True))
        self.non_alnum_max = float(getattr(self.cfg, "ocr_max_non_alnum_ratio", 0.6))
        self.min_len_keep = int(getattr(self.cfg, "ocr_min_text_len_keep", 2))
        self.ignore_index = int(getattr(self.cfg, "mlm_ignore_index", -100))
        self.contrastive_ignore = float(getattr(self.cfg, "contrastive_ignore_value", -1.0))

        self.adv_probability_pretrain = float(getattr(self.cfg, "adv_probability_pretrain", getattr(self.cfg, "adv_probability", 0.35)))
        self.adv_probability_finetune = float(getattr(self.cfg, "adv_probability_finetune", getattr(self.cfg, "adv_probability", 1.0)))
        # HARD-TWC knob (v2, default keeps TWA-aligned behavior): duplicate the box onto
        # the augmented half so the augmented token shares its original's position.
        # Setting False zeros the augmented boxes → removes the positional shortcut →
        # TWC must match via TEXT (char) robustness only → a genuinely hard task.
        self.twc_dup_box = True
        # Fraction of (non-OCR-overlap) content words additionally masked at RANDOM in
        # the decoder MLM span-infill (LM signal + coverage). ↑ = harder MLM.
        self.mlm_rand_prob = 0.15
        # v3 knobs (env-gated in pretrain.py; defaults = current behavior):
        #  gen_target_style: 'sentinel' (T5 span-infill, multi-span) | 'qa' (single
        #  grounded-priority span, RAW target -> matches finetune's answer format).
        self.gen_target_style = "sentinel"
        #  itm_pollute: False = disable the ITM OCR-swap (use when ITM_WEIGHT=0) so
        #  image<->OCR stays matched for 100% of samples (cloze/TWC/ITC consistency).
        self.itm_pollute = True
        self.contrastive_label_list = list(getattr(self.cfg, "contrastive_label_list", [0.9, 0.9]))
        self.editlen = int(getattr(self.cfg, "editlen", 2))

        tokenizer_regex = re.compile(r"[\w]+", re.UNICODE)

        self.char_set = {c: idx for idx, c in enumerate(COMBINED_CHARS)}
        self._char_keys = [c for c in COMBINED_CHARS if c not in {"<s>", "</s>", "<unk>", "<pad>"}]

        self.regex_special = re.compile(r"^(http|https|www|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\d+[./-]\d+|\d{3,}|[a-zA-Z]+\d+|\d+[a-zA-Z]+)")
        self.viet_consonants = ["ngh", "ng", "gh", "gi", "kh", "nh", "ph", "qu", "th", "tr", "ch", "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"]

        self.global_vocab = {}
        self.correction_cache = {}
        self.global_base_map = collections.defaultdict(list)

        if term_vocab_path and os.path.exists(term_vocab_path):
             print(f"[DataCollator] Loading global vocab from: {term_vocab_path}")
             with open(term_vocab_path, 'r', encoding='utf-8') as f:
                 for line in f:
                     w = line.strip().lower()
                     if w:
                         self.global_vocab[w] = min(self.global_vocab.get(w, 1), 1)

        answers = dataframe["answer"].dropna().astype(str).tolist()
        questions = dataframe["question"].dropna().astype(str).tolist()
        qa_regex = re.compile(r"[\w]+")
        for text in answers + questions:
            words = qa_regex.findall(text.lower())
            for w in words:
                if w.isdigit(): continue
                if len(w) > 25: continue
                if len(w) == 1: continue
                self.global_vocab[w] = min(self.global_vocab.get(w, 1), 1)

        if os.path.exists(viet_vocab_path):
            with open(viet_vocab_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue

                    raw = entry.get("text", "")
                    if not raw:
                        continue

                    norm = _normalize_text(raw, lowercase=True)
                    if not norm:
                        continue

                    toks = tokenizer_regex.findall(norm)
                    if not toks:
                        continue

                    if len(toks) > 2:
                        continue

                    for w in toks:
                        w = w.strip()
                        if not w: continue
                        if w.isdigit(): continue
                        if len(w) > 25: continue
                        if len(w) == 1: continue

                        self.global_vocab[w] = min(self.global_vocab.get(w, 0), 0)

        if os.path.exists(eng_vocab_path):
            with open(eng_vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip().lower()
                    if w: self.global_vocab[w] = min(self.global_vocab.get(w, 2), 2)

        for w, src_rank in self.global_vocab.items():
            base_form = _remove_vietnamese_accents(w)
            self.global_base_map[base_form].append((w, src_rank))

        print(f"[Collator] : {len(self.global_vocab)}. Base Map Size: {len(self.global_base_map)}")

    @staticmethod
    def _tokenize_simple(s: str) -> List[str]:
        return re.findall(r"[\w]+", s.lower())

    @staticmethod
    def _jaccard_tokens(a: str, b: str) -> float:
        ta = set(ViT5VQADataCollator._tokenize_simple(a))
        tb = set(ViT5VQADataCollator._tokenize_simple(b))
        if not ta or not tb: return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def _load_term_vocab(fname):
        vocab = collections.OrderedDict()
        try:
            with open(fname, "r", encoding="utf-8") as f:
                for i, line in enumerate(f): vocab[line.rstrip("\n")] = i
        except: vocab["[UNK]"] = 0
        return vocab

    def set_mode(self, pretrain, mask_prob=None, mask_seed=None, debug=None):
        self.pretrain = bool(pretrain)
        if mask_prob is not None: self.mask_prob = float(mask_prob)
        if mask_seed is not None: self.mask_seed = int(mask_seed)
        if debug is not None: self.debug = bool(debug)

    def _is_noise_text(self, s, q_tokens_set=None):
        if not s: return True
        if q_tokens_set and s in q_tokens_set:
            return False
        if len(s) == 1 and not s.isdigit(): return True
        if _non_alnum_ratio(s) > self.non_alnum_max: return True
        if _is_repeated_runs(s): return True
        if _looks_like_code_garbage(s): return True
        if _char_diversity_low(s): return True
        if len(s) < self.min_len_keep and not s.isdigit(): return True
        return False

    @staticmethod
    def _resize_with_pad(image, target_height, target_width):
        original_width, original_height = image.size
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", (target_width, target_height), (128, 128, 128))

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        new_image.paste(image, (paste_x, paste_y))

        return new_image, (new_width, new_height), (paste_x, paste_y)

    @staticmethod
    def _adjust_boxes(boxes, active_size, padding, target_size):
        if boxes is None or len(boxes) == 0:
            return boxes

        new_w, new_h = active_size
        pad_x, pad_y = padding
        target_w, target_h = target_size

        new_boxes = boxes.clone() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float)

        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]] * new_w
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]] * new_h

        new_boxes[:, [0, 2]] += pad_x
        new_boxes[:, [1, 3]] += pad_y

        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]] / target_w
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]] / target_h

        new_boxes = new_boxes.clamp(0.0, 1.0)

        return new_boxes

    @staticmethod
    def _box_iou(b1, b2):
        inter_x1 = max(b1[0], b2[0])
        inter_y1 = max(b1[1], b2[1])
        inter_x2 = min(b1[2], b2[2])
        inter_y2 = min(b1[3], b2[3])

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])

        union = b1_area + b2_area - inter_area
        return inter_area / max(union, 1e-6)

    def _filter_texts(self, texts, det, rec, box, question_str=""):
        keep_indices = []
        seen_items = []
        q_tokens_set = set(self._tokenize_simple(question_str)) if question_str else set()

        num_boxes = det.size(0)

        for i, t in enumerate(texts):
            if i >= num_boxes: continue

            s = _normalize_text(t, lowercase=self.lowercase)
            token_ids = self.tokenizer.encode(s, add_special_tokens=False)

            if len(token_ids) > 25: continue
            if self._is_noise_text(s, q_tokens_set): continue

            is_duplicate = False
            curr_box = box[i]

            for prev_s, prev_box in seen_items:
                text_match = (s == prev_s) or (self._jaccard_tokens(s, prev_s) > 0.9)
                if text_match:
                    if self._box_iou(curr_box, prev_box) > 0.5:
                        is_duplicate = True
                        break

            if is_duplicate: continue

            seen_items.append((s, curr_box))
            keep_indices.append(i)

        if not keep_indices:
            device = det.device
            return ["<unk>"], \
                   torch.zeros(1, det.size(1), dtype=det.dtype, device=device), \
                   torch.zeros(1, rec.size(1), dtype=rec.dtype, device=device), \
                   torch.zeros(1, 4, dtype=box.dtype, device=device)

        idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=det.device)

        keep_t = [texts[i] for i in keep_indices]
        keep_det = det[idx_tensor]
        keep_rec = rec[idx_tensor]
        keep_box = box[idx_tensor]

        return keep_t, keep_det, keep_rec, keep_box

    def _whole_word_mask(self, words, seed):
        """Whole-word MLM masking (TWA-faithful): decide masking at the WORD level,
        THEN tokenize, so a multi-subword word ('pepsi'->['▁p','ep','si']) is masked
        as ONE unit (all its subwords) — no intra-word subword to copy from. Returns
        (input_ids, labels, attention_mask), each length self.txt_max_len. 80/10/10
        (mask/random/keep) applied per WORD. Deterministic per `seed`."""
        rng = random.Random(int(seed) & 0x7FFFFFFF)
        max_len = self.txt_max_len
        vocab = int(getattr(self.tokenizer, "vocab_size", 32000))
        ids, labels = [], []
        for w in words:
            if len(ids) >= max_len - 1:
                break
            sub = self.tokenizer.encode(w, add_special_tokens=False)
            if not sub:
                continue
            if rng.random() < self.mask_prob:
                r2 = rng.random()  # one decision for the whole word
                for sid in sub:
                    labels.append(int(sid))
                    if r2 < 0.8:
                        ids.append(self.mask_token_id)          # [MASK]
                    elif r2 < 0.9:
                        ids.append(rng.randrange(vocab))         # random
                    else:
                        ids.append(int(sid))                     # keep
            else:
                for sid in sub:
                    ids.append(int(sid)); labels.append(-1)
        # truncate (leave room for EOS), append EOS, pad
        ids = ids[:max_len - 1] + [self.eos_id]
        labels = labels[:max_len - 1] + [-1]
        attn = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.pad_id); labels.append(-1); attn.append(0)
        # fallback: guarantee ≥1 masked position (else MLM loss has no target)
        if all(l == -1 for l in labels):
            for k in range(len(ids)):
                if attn[k] == 1 and ids[k] not in (self.pad_id, self.eos_id):
                    labels[k] = ids[k]; ids[k] = self.mask_token_id; break
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long))

    def _random_word(self, tokens, tokenizer, mask_prob, gen, pad_id):
        if mask_prob <= 0.0: return tokens.clone(), torch.full_like(tokens, -1)
        device = tokens.device
        t = tokens.clone()
        if t.dim() == 1: B, T, t2 = 1, t.numel(), t.unsqueeze(0)
        else: B, T, t2 = t.size(0), t.size(1), t

        rnd = torch.rand((B, T), device=device, generator=gen) if gen else torch.rand((B, T), device=device)
        specials = [x for x in [getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "unk_token_id", None), self.mask_token_id] if x is not None]
        choose = rnd < mask_prob
        for sid in specials: choose &= t2 != sid

        labels = torch.where(choose, t2, torch.full_like(t2, -1))
        repl_id = self.mask_token_id
        replace_mask = choose & (rnd < (mask_prob * 0.8))
        random_mask = choose & (rnd >= (mask_prob * 0.8)) & (rnd < (mask_prob * 0.9))
        out = torch.where(replace_mask, torch.full_like(t2, repl_id), t2)
        if random_mask.any():
            voc = int(getattr(tokenizer, "vocab_size", 32000))
            rnd_ids = torch.randint(0, voc, (B, T), device=device, generator=gen, dtype=torch.long) if gen else torch.randint(0, voc, (B, T), device=device, dtype=torch.long)
            out = torch.where(random_mask, rnd_ids, out)
        if t.dim() == 1: out, labels = out.squeeze(0), labels.squeeze(0)
        return out, labels

    def _create_adv_word_adr(self, token):
        rng = random.Random()
        chars = list(token)
        if len(chars) < 2: return token
        if len(chars) < 4:
            ins = rng.choice(self._char_keys)
            pos = rng.randint(1, len(chars))
            chars = chars[:pos] + [ins] + chars[pos:]
        else:
            pos = rng.randint(1, len(chars) - 1)
            if rng.random() < 0.5: del chars[pos]
            else: chars[pos] = rng.choice(self._char_keys)
        return "".join(chars)

    def _get_initial_consonant(self, text):
        for cons in self.viet_consonants:
            if text.startswith(cons): return cons
        return ""

    def _is_vietnamese_vowel(self, ch: str) -> bool:
        if not ch: return False
        base = _remove_vietnamese_accents(ch.lower())
        return base in {"a", "e", "i", "o", "u", "y"}

    def _strip_tone_char(self, ch: str) -> str:
        tone_marks = {"\u0301", "\u0300", "\u0309", "\u0303", "\u0323"}
        base = unicodedata.normalize("NFD", ch)
        return unicodedata.normalize("NFC", "".join(c for c in base if c not in tone_marks))

    def _apply_tone_char(self, ch: str, tone_id: int) -> str:
        if tone_id == 0: return ch
        tone_map = {1: "\u0301", 2: "\u0300", 4: "\u0309", 5: "\u0303", 3: "\u0323"}
        mark = tone_map.get(tone_id)
        if mark is None: return ch
        base_wo_tone = self._strip_tone_char(ch)
        return unicodedata.normalize("NFC", unicodedata.normalize("NFD", base_wo_tone) + mark)

    def _generate_tone_shift_candidates(self, token: str) -> List[str]:
        chars = list(token); vowels_idx = []; tone_positions = []; tone_id = 0
        for i, ch in enumerate(chars):
            if self._is_vietnamese_vowel(ch):
                vowels_idx.append(i)
                t = get_tone_id(ch)
                if t != 0: tone_positions.append(i); tone_id = t
        if tone_id == 0 or len(vowels_idx) <= 1 or len(tone_positions) != 1: return []
        tone_pos = tone_positions[0]
        base_chars = chars.copy()
        base_chars[tone_pos] = self._strip_tone_char(base_chars[tone_pos])
        candidates = []
        for idx in vowels_idx:
            if idx == tone_pos: continue
            new_chars = base_chars.copy()
            new_chars[idx] = self._apply_tone_char(new_chars[idx], tone_id)
            candidates.append("".join(new_chars))
        return candidates

    def _generate_tone_change_candidates(self, token: str) -> List[str]:
        chars = list(token); vowels_idx = []
        for i, ch in enumerate(chars):
            if self._is_vietnamese_vowel(ch): vowels_idx.append(i)
        if not vowels_idx: return []
        current_tone_idx = -1
        for idx in vowels_idx:
            if get_tone_id(chars[idx]) != 0: current_tone_idx = idx; break
        target_indices = [current_tone_idx] if current_tone_idx != -1 else vowels_idx
        possible_tones = [0, 1, 2, 3, 4, 5]
        candidates = []
        for idx in target_indices:
            base_char = self._strip_tone_char(chars[idx])
            for t_id in possible_tones:
                if t_id == get_tone_id(chars[idx]): continue
                new_chars = chars.copy()
                new_chars[idx] = self._apply_tone_char(base_char, t_id)
                candidates.append("".join(new_chars))
        return candidates

    def _find_related_word(self, token, editlen):
        token = token.lower().strip()
        if not token or token[0].isdigit() or token[-1].isdigit(): return None
        if token in self.correction_cache: return self.correction_cache[token]

        candidates = []
        for cand in self._generate_tone_shift_candidates(token):
            rank = self.global_vocab.get(cand)
            if rank is not None: candidates.append({"word": cand, "dist": 0, "type": 0, "source": rank, "tone": 0})
        for cand in self._generate_tone_change_candidates(token):
            rank = self.global_vocab.get(cand)
            if rank is not None: candidates.append({"word": cand, "dist": 0, "type": 0, "source": rank, "tone": 0})

        if candidates:
            candidates.sort(key=lambda x: (x["type"], x["source"], -x["tone"]))
            best = candidates[0]["word"]
            self.correction_cache[token] = best
            return best

        token_base = _remove_vietnamese_accents(token)
        token_cons = self._get_initial_consonant(token)
        potential_cands = set(self.global_base_map.get(token_base, []))
        if not potential_cands:
             for w, rank in self.global_vocab.items():
                if abs(len(w) - len(token)) > editlen: continue
                if rank > 0 and token_cons != self._get_initial_consonant(w): continue
                potential_cands.add((w, rank))

        for d_target in range(1, editlen + 1):
            valid_cands = []
            for w, src_rank in potential_cands:
                if abs(len(w) - len(token)) > d_target: continue
                d = editdistance.eval(token, w)
                if d == d_target:
                    curr_type = 1 if len(w) == len(token) else 2
                    valid_cands.append({"word": w, "dist": d, "type": curr_type, "source": src_rank, "tone": get_word_tone_score(w)})
            if valid_cands:
                valid_cands.sort(key=lambda x: (x["type"], x["source"], -x["tone"]))
                best = valid_cands[0]["word"]
                self.correction_cache[token] = best
                return best
        return None

    def _findRelatedOCR_adr(self, ocr_tokens, ocr_max_num, adv_probability, label_list, editlen):
        """
        Xây dựng label matrix theo đúng notebook gốc / TWA paper (Sec 3.4).

        - Token giữ nguyên (không augment) = positive pair, label 1.0.
        - Token bị char-noise hoặc thay bằng similar-word = semi-positive, label 0.9.
        - Cặp token khác text trong batch = negative, label 0.0.

        KHÔNG ignore cặp identical: paper coi token-không-đổi là positive 1.0 và
        dùng các cặp này làm "mỏ neo" hiệu chỉnh thang similarity. PAD word có
        feature = 0 sau pooling nên cặp PAD-PAD không tạo gradient (vô hại).
        """
        o2r = torch.ones(ocr_max_num, ocr_max_num, dtype=torch.float) * self.contrastive_ignore
        r2o = torch.ones(ocr_max_num, ocr_max_num, dtype=torch.float) * self.contrastive_ignore
        toks = ocr_tokens[:ocr_max_num]
        N = min(len(toks), ocr_max_num)
        related, padded, actions = [], [], []
        pad_tok = self.tokenizer.pad_token or "<pad>"

        for i in range(N):
            raw_tok = toks[i]
            norm_tok = _normalize_text(raw_tok, lowercase=True)
            if raw_tok == pad_tok or norm_tok in {"<pad>", "</s>"}:
                # PAD is NOT a contrastive sample: its pooled feature is ~0 and
                # labelling PAD-PAD as positive (=1.0) floods the positive class
                # with zero-feature pairs, collapsing TWC. Ignore it entirely (-1).
                rel = pad_tok
                o2r[i, i], r2o[i, i] = self.contrastive_ignore, self.contrastive_ignore
                actions.append("PAD")
                padded.append(norm_tok); related.append(rel); continue

            is_special = bool(self.regex_special.search(norm_tok))
            in_vocab = norm_tok in self.global_vocab
            is_number = norm_tok.isdigit()

            if is_special or is_number:
                rel = norm_tok; o2r[i, i], r2o[i, i] = 1.0, 1.0; actions.append("KEEP_SPECIAL")
            elif in_vocab:
                if random.random() < adv_probability and len(norm_tok) > 1:
                    rel = self._create_adv_word_adr(norm_tok)
                    o2r[i, i], r2o[i, i] = float(label_list[1]), float(label_list[0])
                    actions.append("NOISE")
                else:
                    rel = norm_tok; o2r[i, i], r2o[i, i] = 1.0, 1.0; actions.append("KEEP")
            else:
                if random.random() < adv_probability:
                    found = self._find_related_word(norm_tok, editlen)
                    if found:
                        rel = found; o2r[i, i], r2o[i, i] = float(label_list[0]), float(label_list[1]); actions.append("CORRECT")
                    else:
                        rel = norm_tok; o2r[i, i], r2o[i, i] = 1.0, 1.0; actions.append("KEEP_UNKNOWN")
                else:
                    rel = norm_tok; o2r[i, i], r2o[i, i] = 1.0, 1.0; actions.append("KEEP")
            padded.append(norm_tok); related.append(rel)

        for i in range(len(related)):
            for j in range(i + 1, len(related)):
                # If either token is PAD/ignored (diagonal == ignore), leave the
                # whole pair ignored so PAD never becomes a positive or negative.
                if o2r[i, i] == self.contrastive_ignore or o2r[j, j] == self.contrastive_ignore:
                    continue
                if padded[i].lower() == padded[j].lower():
                    o2r[i, j], o2r[j, i] = o2r[j, j], o2r[i, i]
                    r2o[i, j], r2o[j, i] = r2o[j, j], r2o[i, i]
                else:
                    o2r[i, j], o2r[j, i] = 0.0, 0.0
                    r2o[i, j], r2o[j, i] = 0.0, 0.0

        while len(padded) < ocr_max_num:
            padded.append(pad_tok); related.append(pad_tok)
            if self.debug: actions.append("PAD")
        if self.debug: return padded, related, o2r, r2o, actions
        return padded, related, o2r, r2o, None

    def _findRelatedOCR_plain(self, ocr_tokens, ocr_max_num, adv_probability, editlen):
        toks = ocr_tokens[:ocr_max_num]
        N = min(len(toks), ocr_max_num)
        related, padded = [], []
        for i in range(N):
            tok = toks[i].lower().strip()
            padded.append(tok)
            is_special = bool(self.regex_special.search(tok))
            in_vocab = tok in self.global_vocab
            is_number = tok.isdigit()
            if is_special or in_vocab or is_number: rel = tok
            else:
                if tok and (tok[0].isdigit() or tok[-1].isdigit()): rel = tok
                else:
                    if random.random() < adv_probability and len(tok) > 1:
                        found = self._find_related_word(tok, editlen)
                        rel = found if found else tok
                    else: rel = tok
            related.append(rel)
        while len(padded) < ocr_max_num:
            padded.append(self.tokenizer.pad_token); related.append(self.tokenizer.pad_token)
        return padded, related

    def _add_cons_ocr_info(self, ocr_tokens, ocr_max_num):
        C = self.char_max_num
        char_mask = torch.zeros(ocr_max_num, C, dtype=torch.float)
        char_ids = torch.zeros(ocr_max_num, C, dtype=torch.long)
        all_word_ids, token_lengths = [], []
        specials = {x for x in [getattr(self.tokenizer, "pad_token", None), getattr(self.tokenizer, "eos_token", None)] if x}
        unk_id = int(getattr(self.tokenizer, "unk_token_id", 2) or 2)

        N = min(len(ocr_tokens), ocr_max_num)
        for i in range(N):
            tok = ocr_tokens[i]
            if tok in specials or _normalize_text(tok, lowercase=True) in {"<pad>", "</s>"}:
                all_word_ids.append(self.pad_id); token_lengths.append(1); continue

            ids = self.tokenizer.encode(_normalize_text(tok, lowercase=True), add_special_tokens=False)
            if len(ids) > 0: all_word_ids.extend(ids); token_lengths.append(len(ids))
            else: all_word_ids.append(unk_id); token_lengths.append(1)

            norm = _normalize_text(tok, lowercase=True)
            Lc = min(len(norm), C)
            for c_i in range(Lc):
                c_idx = self.char_set.get(norm[c_i], self.char_set.get("<unk>", 1))
                char_ids[i, c_i] = c_idx; char_mask[i, c_i] = 1.0

        return char_ids, char_mask, torch.tensor(all_word_ids, dtype=torch.long), torch.tensor(token_lengths, dtype=torch.long)

    def _prepare_ocr(self, ocr_raw, max_len_in_batch=None, question=""):
        det = ocr_raw["det_features"]
        rec = ocr_raw["rec_features"]
        box = ocr_raw["boxes"]
        texts = ocr_raw.get("texts", [])

        texts_f, det_f, rec_f, box_f = self._filter_texts(texts, det, rec, box, question_str=question)

        if det_f.size(0) > self.seq_max:
            det_f = det_f[:self.seq_max]; rec_f = rec_f[:self.seq_max]; box_f = box_f[:self.seq_max]; texts_f = texts_f[:self.seq_max]

        target_len = max_len_in_batch if max_len_in_batch is not None else self.seq_max
        target_len = max(target_len, det_f.size(0))

        Lw = det_f.size(0)
        word_mask = torch.ones(Lw, dtype=torch.long, device=det_f.device)

        if Lw < target_len:
            pad_len = target_len - Lw
            device = det_f.device
            det_pad = torch.zeros(pad_len, det_f.size(1), dtype=det_f.dtype, device=device)
            rec_pad = torch.zeros(pad_len, rec_f.size(1), dtype=rec_f.dtype, device=device)
            box_pad = torch.zeros(pad_len, 4, dtype=box_f.dtype, device=device)

            det_f = torch.cat([det_f, det_pad], dim=0)
            rec_f = torch.cat([rec_f, rec_pad], dim=0)
            box_f = torch.cat([box_f, box_pad], dim=0)
            word_mask = torch.cat([word_mask, torch.zeros(pad_len, dtype=torch.long, device=device)], dim=0)

            pad_tok = self.tokenizer.pad_token or "<pad>"
            texts_f = texts_f + [pad_tok] * pad_len

        return {
            "det_features": det_f, "rec_features": rec_f, "boxes": box_f,
            "word_mask": word_mask, "width": ocr_raw["width"], "height": ocr_raw["height"],
            "texts": texts_f,
        }, texts_f

    def __call__(self, batch):
        paths = [b["image_path"] for b in batch]
        ocr_paths = [b.get("ocr_path") for b in batch]
        qs = [b["question"] for b in batch]
        ans = [b.get("answer", "") for b in batch]
        uids = [b.get("uid", p) for b, p in zip(batch, paths)]

        # --- 1. Xử lý Ảnh (Image) ---
        pil_images = []
        for p in paths:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)
            pil_images.append(img.convert("RGB"))

        proc = self.image_processor(images=pil_images, return_tensors="pt")
        pixel_values = proc["pixel_values"]

        # --- 2. Tokenize Answer và Question ---
        lab = self.tokenizer(ans, padding="max_length", truncation=True, max_length=self.tgt_max_len, return_tensors="pt").input_ids
        lab[lab == self.pad_id] = -100

        q_tok = self.tokenizer(qs, padding="max_length", truncation=True, max_length=self.txt_max_len, return_tensors="pt")

        # --- 3. Xử lý OCR và Tính độ dài Dynamic Padding ---
        ocr_raw_list = self.ocr_encoder(paths, ocr_paths=ocr_paths)

        batch_ocr_lens = [raw["det_features"].shape[0] for raw in ocr_raw_list]
        current_max_len = max(batch_ocr_lens) if batch_ocr_lens else 0
        current_max_len = min(current_max_len, self.seq_max)
        if current_max_len % 8 != 0: current_max_len = (current_max_len // 8 + 1) * 8
        current_max_len = max(current_max_len, 2)

        # Lưu vào ITM History
        for p, ocr_raw in zip(paths, ocr_raw_list):
            self.itm_history.append((p, ocr_raw))
            if len(self.itm_history) > self.itm_history_max: self.itm_history.pop(0)

        # =========================================================
        # =========================================================
        # NHÁNH PRETRAIN: PRESTU SPLITOCR (Task-Agnostic Core STU)
        # =========================================================
        if self.pretrain:
            B = len(batch)
            split_prompts = []
            split_targets = []

            pad_tok = self.tokenizer.pad_token or "<pad>"
            for i in range(B):
                info, raw_texts = self._prepare_ocr(ocr_raw_list[i], max_len_in_batch=current_max_len)
                norm_tokens = [
                    _normalize_text(t, lowercase=True).strip()
                    for t in raw_texts
                    if isinstance(t, str) and _normalize_text(t, lowercase=True).strip() and _normalize_text(t, lowercase=True).strip() not in {pad_tok, "<pad>", "<unk>", "</s>"}
                ]

                # --- 1. Sắp xếp không gian Top-Left -> Bottom-Right ---
                N_words = len(norm_tokens)

                # --- 2. Chọn điểm cắt ngẫu nhiên m in [0, N_words - 1] (PreSTU SplitOCR) ---
                if N_words == 0:
                    prefix_str = ""
                    target_str = ""
                else:
                    # Random split point: m = 0 -> Full OCR; m > 0 -> Split continuation
                    m = random.randint(0, max(0, N_words - 1))
                    prefix_words = norm_tokens[:m]
                    target_words = norm_tokens[m:]
                    prefix_str = " ".join(prefix_words).strip()
                    target_str = " ".join(target_words).strip()

                # Prompt chuẩn PreSTU
                prompt_text = f"Generate ocr_text in vi: {prefix_str}".strip() if prefix_str else "Generate ocr_text in vi:"
                split_prompts.append(prompt_text)
                split_targets.append(target_str)

            # Tokenize SplitOCR Inputs (Prompt + Prefix) và Labels (Suffix Target)
            prompt_tok = self.tokenizer(
                split_prompts,
                padding="max_length",
                truncation=True,
                max_length=self.txt_max_len,
                return_tensors="pt"
            )
            target_tok = self.tokenizer(
                split_targets,
                padding="max_length",
                truncation=True,
                max_length=self.tgt_max_len,
                return_tensors="pt"
            )
            labels = target_tok.input_ids
            labels[labels == self.pad_id] = -100

            return {
                "input_ids": prompt_tok.input_ids.to(pixel_values.device),
                "attention_mask": prompt_tok.attention_mask.to(pixel_values.device),
                "labels": labels.to(pixel_values.device),
                "pixel_values": pixel_values,
                "pil_images": pil_images,
            }

        # =========================================================
        # NHÁNH FINETUNE / INFERENCE
        # =========================================================
        else:
            ocr_info_list, twa_char_list, twa_char_mask_list, twa_word_ids_list, ocr_to_word_map_list, ocr_mask_list = [], [], [], [], [], []

            for i in range(len(batch)):
                info, raw_texts = self._prepare_ocr(ocr_raw_list[i], max_len_in_batch=current_max_len)
                norm_tokens = [_normalize_text(t, lowercase=True) for t in raw_texts]

                if use_ocr_aug:
                    pad_ocr, rel_ocr = self._findRelatedOCR_plain(norm_tokens, current_max_len, adv_pro, self.editlen)
                    char_a, mask_a, flat_ids_a, lens_a = self._add_cons_ocr_info(pad_ocr, current_max_len)
                    char_b, mask_b, flat_ids_b, lens_b = self._add_cons_ocr_info(rel_ocr, current_max_len)

                    twa_char_list.append(torch.cat([char_a, char_b], dim=0))
                    twa_char_mask_list.append(torch.cat([mask_a, mask_b], dim=0))
                    twa_word_ids_list.append(torch.cat([flat_ids_a, flat_ids_b], dim=0))

                    indices_map = []
                    for j, l in enumerate(torch.cat([lens_a, lens_b], dim=0)): indices_map.extend([j] * l.item())
                    ocr_to_word_map_list.append(torch.tensor(indices_map, dtype=torch.long))

                    boxes_all = torch.cat([info["boxes"], info["boxes"]], dim=0)
                    mask_all = torch.cat([info["word_mask"], info["word_mask"]], dim=0)

                else:
                    pad_ocr = norm_tokens[:current_max_len]
                    while len(pad_ocr) < current_max_len: pad_ocr.append(self.tokenizer.pad_token or "<pad>")
                    char_a, mask_a, flat_ids_a, lens_a = self._add_cons_ocr_info(pad_ocr, current_max_len)

                    twa_char_list.append(char_a)
                    twa_char_mask_list.append(mask_a)
                    twa_word_ids_list.append(flat_ids_a)

                    indices_map = []
                    for j, l in enumerate(lens_a): indices_map.extend([j] * l.item())
                    ocr_to_word_map_list.append(torch.tensor(indices_map, dtype=torch.long))

                    boxes_all = info["boxes"]
                    mask_all = info["word_mask"]

                info["boxes_word_all"] = boxes_all; info["word_mask_all"] = mask_all
                ocr_info_list.append(info)
                ocr_mask_list.append(mask_all.clone())

            return {
                "input_ids": q_tok.input_ids.to(pixel_values.device),
                "attention_mask": q_tok.attention_mask.to(pixel_values.device),
                "labels": lab.to(pixel_values.device) if lab is not None else None,
                "pixel_values": pixel_values, "pil_images": pil_images,
                "ocr_info": ocr_info_list,
                "ocr_mask_token": torch.stack(ocr_mask_list).to(pixel_values.device),
                "ocr_mask_box": torch.stack(ocr_mask_list).to(pixel_values.device),
                "twa_ocr_char": torch.stack(twa_char_list).to(pixel_values.device),
                "twa_ocr_char_mask": torch.stack(twa_char_mask_list).to(pixel_values.device),
                "twa_word_ids": torch.nn.utils.rnn.pad_sequence(twa_word_ids_list, batch_first=True, padding_value=self.pad_id).to(pixel_values.device),
                "ocr_to_word_map": torch.nn.utils.rnn.pad_sequence(ocr_to_word_map_list, batch_first=True, padding_value=-1).to(pixel_values.device)
            }