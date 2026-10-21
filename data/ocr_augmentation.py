"""Fine-tuning OCR correction restored from the legacy collator, without pretrain tasks."""

import collections
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import List

from rapidfuzz.distance import Levenshtein

from data.vocab import _normalize_text, _remove_vietnamese_accents

TONE_RANK = {index: index for index in range(6)}


def get_tone_id(char: str) -> int:
    base = unicodedata.normalize("NFD", char)
    for c in base:
        if c == "\u0301":
            return 1
        if c == "\u0300":
            return 2
        if c == "\u0309":
            return 4
        if c == "\u0303":
            return 5
        if c == "\u0323":
            return 3
    return 0


def get_word_tone_score(word: str) -> int:
    max_rank = 0
    for char in word:
        t_id = get_tone_id(char)
        rank = TONE_RANK.get(t_id, 0)
        if rank > max_rank:
            max_rank = rank
    return max_rank


class OCRTextAugmenter:
    """One deterministic related entry per original box; never reads QA labels."""

    def __init__(self, vocab_dir, probability=1.0, editlen=2):
        self.probability = float(probability)
        self.editlen = int(editlen)
        if not 0 <= self.probability <= 1 or self.editlen < 1:
            raise ValueError(
                "OCR augmentation requires probability in [0,1] and editlen >= 1."
            )
        self.regex_special = re.compile(
            r"^(http|https|www|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\d+[./-]\d+|\d{3,}|[a-zA-Z]+\d+|\d+[a-zA-Z]+)"
        )
        self.viet_consonants = [
            "ngh",
            "ng",
            "gh",
            "gi",
            "kh",
            "nh",
            "ph",
            "qu",
            "th",
            "tr",
            "ch",
            "b",
            "c",
            "d",
            "đ",
            "g",
            "h",
            "k",
            "l",
            "m",
            "n",
            "p",
            "r",
            "s",
            "t",
            "v",
            "x",
        ]
        directory = Path(vocab_dir)
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parents[1] / directory
        self.global_vocab = {}
        for name, rank in (
            ("term_vocab.txt", 1),
            ("viet_vocab.txt", 0),
            ("eng_vocab.txt", 2),
        ):
            path = directory / name
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if name == "viet_vocab.txt":
                        try:
                            words = re.findall(
                                r"\w+",
                                _normalize_text(json.loads(line).get("text", "")),
                            )
                        except (ValueError, AttributeError, TypeError):
                            continue
                        if len(words) > 2:
                            continue
                        words = [
                            word
                            for word in words
                            if 1 < len(word) <= 25 and not word.isdigit()
                        ]
                    else:
                        words = [line.strip().lower()]
                    for word in words:
                        if word:
                            self.global_vocab[word] = min(
                                rank, self.global_vocab.get(word, rank)
                            )
        if not self.global_vocab:
            raise ValueError(f"No OCR augmentation vocabulary found in {directory}.")
        self.global_base_map = collections.defaultdict(list)
        self.vocab_by_length = collections.defaultdict(list)
        for word, rank in self.global_vocab.items():
            self.global_base_map[_remove_vietnamese_accents(word)].append((word, rank))
            self.vocab_by_length[len(word)].append((word, rank))
        self.correction_cache = {}

    def __call__(self, texts):
        related = []
        for text in texts:
            token = text.lower().strip()
            if (
                not token
                or token in {"<unk>", "<pad>"}
                or self.regex_special.search(token)
                or token in self.global_vocab
                or token[0].isdigit()
                or token[-1].isdigit()
                or len(token) < 2
            ):
                related.append(text)
                continue
            # Stable by OCR entry, independent of batch size, worker count or split.
            sample = (
                int.from_bytes(
                    hashlib.sha256(token.encode("utf-8")).digest()[:8], "little"
                )
                / 2**64
            )
            found = (
                self._find_related_word(token, self.editlen)
                if sample < self.probability
                else None
            )
            related.append(found or text)
        return related

    def _get_initial_consonant(self, text):
        for cons in self.viet_consonants:
            if text.startswith(cons):
                return cons
        return ""

    def _is_vietnamese_vowel(self, ch: str) -> bool:
        if not ch:
            return False
        base = _remove_vietnamese_accents(ch.lower())
        return base in {"a", "e", "i", "o", "u", "y"}

    def _strip_tone_char(self, ch: str) -> str:
        tone_marks = {"\u0301", "\u0300", "\u0309", "\u0303", "\u0323"}
        base = unicodedata.normalize("NFD", ch)
        return unicodedata.normalize(
            "NFC", "".join(c for c in base if c not in tone_marks)
        )

    def _apply_tone_char(self, ch: str, tone_id: int) -> str:
        if tone_id == 0:
            return ch
        tone_map = {1: "\u0301", 2: "\u0300", 4: "\u0309", 5: "\u0303", 3: "\u0323"}
        mark = tone_map.get(tone_id)
        if mark is None:
            return ch
        base_wo_tone = self._strip_tone_char(ch)
        return unicodedata.normalize(
            "NFC", unicodedata.normalize("NFD", base_wo_tone) + mark
        )

    def _generate_tone_shift_candidates(self, token: str) -> List[str]:
        chars = list(token)
        vowels_idx = []
        tone_positions = []
        tone_id = 0
        for i, ch in enumerate(chars):
            if self._is_vietnamese_vowel(ch):
                vowels_idx.append(i)
                t = get_tone_id(ch)
                if t != 0:
                    tone_positions.append(i)
                    tone_id = t
        if tone_id == 0 or len(vowels_idx) <= 1 or len(tone_positions) != 1:
            return []
        tone_pos = tone_positions[0]
        base_chars = chars.copy()
        base_chars[tone_pos] = self._strip_tone_char(base_chars[tone_pos])
        candidates = []
        for idx in vowels_idx:
            if idx == tone_pos:
                continue
            new_chars = base_chars.copy()
            new_chars[idx] = self._apply_tone_char(new_chars[idx], tone_id)
            candidates.append("".join(new_chars))
        return candidates

    def _generate_tone_change_candidates(self, token: str) -> List[str]:
        chars = list(token)
        vowels_idx = []
        for i, ch in enumerate(chars):
            if self._is_vietnamese_vowel(ch):
                vowels_idx.append(i)
        if not vowels_idx:
            return []
        current_tone_idx = -1
        for idx in vowels_idx:
            if get_tone_id(chars[idx]) != 0:
                current_tone_idx = idx
                break
        target_indices = [current_tone_idx] if current_tone_idx != -1 else vowels_idx
        possible_tones = [0, 1, 2, 3, 4, 5]
        candidates = []
        for idx in target_indices:
            base_char = self._strip_tone_char(chars[idx])
            for t_id in possible_tones:
                if t_id == get_tone_id(chars[idx]):
                    continue
                new_chars = chars.copy()
                new_chars[idx] = self._apply_tone_char(base_char, t_id)
                candidates.append("".join(new_chars))
        return candidates

    def _find_related_word(self, token, editlen):
        token = token.lower().strip()
        if not token or token[0].isdigit() or token[-1].isdigit():
            return None
        if token in self.correction_cache:
            return self.correction_cache[token]

        candidates = []
        for cand in self._generate_tone_shift_candidates(token):
            rank = self.global_vocab.get(cand)
            if rank is not None:
                candidates.append(
                    {"word": cand, "dist": 0, "type": 0, "source": rank, "tone": 0}
                )
        for cand in self._generate_tone_change_candidates(token):
            rank = self.global_vocab.get(cand)
            if rank is not None:
                candidates.append(
                    {"word": cand, "dist": 0, "type": 0, "source": rank, "tone": 0}
                )

        if candidates:
            candidates.sort(
                key=lambda x: (x["type"], x["source"], -x["tone"], x["word"])
            )
            best = candidates[0]["word"]
            self.correction_cache[token] = best
            return best

        token_base = _remove_vietnamese_accents(token)
        token_cons = self._get_initial_consonant(token)
        potential_cands = set(self.global_base_map.get(token_base, []))
        if not potential_cands:
            for length in range(max(1, len(token) - editlen), len(token) + editlen + 1):
                for w, rank in self.vocab_by_length[length]:
                    if rank > 0 and token_cons != self._get_initial_consonant(w):
                        continue
                    potential_cands.add((w, rank))

        for d_target in range(1, editlen + 1):
            valid_cands = []
            for w, src_rank in potential_cands:
                if abs(len(w) - len(token)) > d_target:
                    continue
                d = Levenshtein.distance(token, w, score_cutoff=editlen)
                if d == d_target:
                    curr_type = 1 if len(w) == len(token) else 2
                    valid_cands.append(
                        {
                            "word": w,
                            "dist": d,
                            "type": curr_type,
                            "source": src_rank,
                            "tone": get_word_tone_score(w),
                        }
                    )
            if valid_cands:
                valid_cands.sort(
                    key=lambda x: (x["type"], x["source"], -x["tone"], x["word"])
                )
                best = valid_cands[0]["word"]
                self.correction_cache[token] = best
                return best
        self.correction_cache[token] = None
        return None
