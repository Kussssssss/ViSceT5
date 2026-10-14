"""Lexicon-based OCR correction and TWA-style augmentation for the related OCR copy.

Two ways to build the related copy of an OCR word list (same length, same order):

* ``correct``: deterministic correction channel (used at inference, and in training
  when TWC is off). Specials/numbers/in-lexicon words are kept; other words are
  replaced by the best lexicon candidate (tone shift → tone change → edit distance
  ≤ ``edit_distance`` with the same base form or initial consonant).
* ``augment``: TWA augmentation for the TWC loss (Jin et al., MM'22 §3.4), ported from
  the previous pretrain collator (``_findRelatedOCR_adr``): with probability ``p`` an
  in-lexicon word gets one character edit (label 0.9) and an out-of-lexicon word is
  corrected (label 0.9); untouched words are positives (1.0). Words with the same text
  inherit each other's label; different texts are negatives (0.0).
"""

import json
import random
import re
import unicodedata
from collections import defaultdict

import editdistance

from data.vocab import COMBINED_CHARS, _normalize_text, _remove_vietnamese_accents

_TONE_MARKS = {"́": 1, "̀": 2, "̣": 3, "̉": 4, "̃": 5}
_TONE_BY_ID = {tone: mark for mark, tone in _TONE_MARKS.items()}
_SPECIAL = re.compile(
    r"^(http|https|www|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\d+[./-]\d+|\d{3,}|[a-zA-Z]+\d+|\d+[a-zA-Z]+)"
)
_CONSONANTS = ["ngh", "ng", "gh", "gi", "kh", "nh", "ph", "qu", "th", "tr", "ch",
               "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"]
_WORD = re.compile(r"[\w]+")


def tone_id(char):
    for mark in unicodedata.normalize("NFD", char):
        if mark in _TONE_MARKS:
            return _TONE_MARKS[mark]
    return 0


def word_tone_score(word):
    return max((tone_id(char) for char in word), default=0)


def _strip_tone(char):
    return unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", char) if c not in _TONE_MARKS)
    )


def _with_tone(char, tone):
    base = _strip_tone(char)
    if tone == 0:
        return base
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", base) + _TONE_BY_ID[tone])


def _is_vowel(char):
    return _remove_vietnamese_accents(char.lower()) in {"a", "e", "i", "o", "u", "y"}


def _initial_consonant(word):
    return next((c for c in _CONSONANTS if word.startswith(c)), "")


def _keep_as_is(word):
    """Special strings, numbers and words with a digit at either end are never edited."""
    return bool(_SPECIAL.search(word)) or word.isdigit() or word[0].isdigit() or word[-1].isdigit()


class OCRCorrector:
    def __init__(self, term_vocab_path=None, viet_vocab_path=None, train_frame=None,
                 edit_distance=2, label_list=(0.9, 0.9)):
        # rank: term list / train Q&A = 1, dictionary = 0 (lower sorts first).
        self.vocab = {}
        if term_vocab_path:
            with open(term_vocab_path, encoding="utf-8") as stream:
                for line in stream:
                    word = line.strip().lower()
                    if word:
                        self.vocab[word] = min(self.vocab.get(word, 1), 1)
        if train_frame is not None:
            for column in ("answer", "question"):
                if column not in train_frame:
                    continue
                for text in train_frame[column].dropna().astype(str):
                    for word in _WORD.findall(text.lower()):
                        if not word.isdigit() and 1 < len(word) <= 25:
                            self.vocab[word] = min(self.vocab.get(word, 1), 1)
        if viet_vocab_path:
            with open(viet_vocab_path, encoding="utf-8") as stream:
                for line in stream:
                    try:
                        raw = json.loads(line).get("text", "") if line.strip() else ""
                    except Exception:
                        continue
                    words = _WORD.findall(_normalize_text(raw, lowercase=True)) if raw else []
                    if 0 < len(words) <= 2:
                        for word in words:
                            if not word.isdigit() and 1 < len(word) <= 25:
                                self.vocab[word] = min(self.vocab.get(word, 0), 0)
        self.base_map = defaultdict(list)
        for word, rank in self.vocab.items():
            self.base_map[_remove_vietnamese_accents(word)].append((word, rank))
        self.edit_distance = int(edit_distance)
        self.label_list = [float(value) for value in label_list]
        self.noise_chars = [c for c in COMBINED_CHARS if c not in {"<unk>", "<s>", "</s>", "<pad>"}]
        self._cache = {}

    # ----------------------------------------------------------------- lookup
    def _tone_candidates(self, word):
        chars = list(word)
        vowels = [i for i, c in enumerate(chars) if _is_vowel(c)]
        toned = [i for i in vowels if tone_id(chars[i])]
        out = []
        if len(vowels) > 1 and len(toned) == 1:  # tone shift: move the mark to another vowel
            base = chars.copy()
            base[toned[0]] = _strip_tone(base[toned[0]])
            for i in vowels:
                if i != toned[0]:
                    shifted = base.copy()
                    shifted[i] = _with_tone(shifted[i], tone_id(chars[toned[0]]))
                    out.append("".join(shifted))
        targets = toned[:1] or vowels  # tone change on the marked (else every) vowel
        for i in targets:
            for tone in range(6):
                if tone != tone_id(chars[i]):
                    changed = chars.copy()
                    changed[i] = _with_tone(chars[i], tone)
                    out.append("".join(changed))
        return out

    def find_related(self, word):
        """Best lexicon candidate for an out-of-lexicon word, or None.

        Ties are broken by the word itself so the choice never depends on set order
        (PYTHONHASHSEED), i.e. training and inference correct a word the same way.
        """
        if not word or word[0].isdigit() or word[-1].isdigit():
            return None
        if word in self._cache:
            return self._cache[word]
        best = None
        hits = [(0, self.vocab[c], 0, c) for c in self._tone_candidates(word) if c in self.vocab]
        if hits:
            best = min(hits)[3]
        else:
            pool = set(self.base_map.get(_remove_vietnamese_accents(word), []))
            if not pool:
                initial = _initial_consonant(word)
                pool = {(w, r) for w, r in self.vocab.items()
                        if abs(len(w) - len(word)) <= self.edit_distance
                        and (r <= 0 or initial == _initial_consonant(w))}
            for distance in range(1, self.edit_distance + 1):
                found = [(1 if len(w) == len(word) else 2, r, -word_tone_score(w), w)
                         for w, r in pool
                         if abs(len(w) - len(word)) <= distance and editdistance.eval(word, w) == distance]
                if found:
                    best = min(found)[3]
                    break
        self._cache[word] = best
        return best

    # ------------------------------------------------------------- related copy
    def correct(self, words):
        related = []
        for word in words:
            if _keep_as_is(word) or word in self.vocab or len(word) < 2:
                related.append(word)
            else:
                related.append(self.find_related(word) or word)
        return related

    def _char_noise(self, word, rng):
        chars = list(word)
        if len(chars) < 4:
            position = rng.randint(1, len(chars))
            chars.insert(position, rng.choice(self.noise_chars))
        else:
            position = rng.randint(1, len(chars) - 1)
            if rng.random() < 0.5:
                del chars[position]
            else:
                chars[position] = rng.choice(self.noise_chars)
        return "".join(chars)

    def augment(self, words, probability, rng):
        """TWA augmentation → (related, o2r, r2o) with n×n label lists (no padding)."""
        n = len(words)
        related, diag_o2r, diag_r2o = [], [], []
        semi_o2r, semi_r2o = self.label_list[1], self.label_list[0]
        for word in words:
            label = (1.0, 1.0)
            if _keep_as_is(word):
                rel = word
            elif word in self.vocab:
                rel = word
                if rng.random() < probability and len(word) > 1:
                    rel, label = self._char_noise(word, rng), (semi_o2r, semi_r2o)  # NOISE
            else:
                rel = word
                if rng.random() < probability:
                    found = self.find_related(word)
                    if found:
                        rel, label = found, (self.label_list[0], self.label_list[1])  # CORRECT
            related.append(rel)
            diag_o2r.append(label[0])
            diag_r2o.append(label[1])
        o2r = [[0.0] * n for _ in range(n)]
        r2o = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j or words[i] == words[j]:  # same text → inherit the partner's label
                    o2r[i][j], r2o[i][j] = diag_o2r[j], diag_r2o[j]
        return related, o2r, r2o
