"""
data/vocab.py
Character vocab, text normalisation, noise-detection helpers, vocab builder.
"""

import os
import re
import json
import collections
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
import editdistance
import pandas as pd
import unicodedata
import numpy as np

TEST = False
TONE_RANK = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
# fix: Thêm để trả về <unk> nếu kỹ tự không nằm trong 'COMBINED_CHARS' -> Fallback
_CHAR_UNK_IDX = 0
COMBINED_CHARS = [
    "<unk>",
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

    'à', 'á', 'ả', 'ã', 'ạ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
    'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ',
    'đ',
    'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ',
    'ì', 'í', 'ỉ', 'ĩ', 'ị',
    'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
    'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ',
    'ù', 'ú', 'ủ', 'ũ', 'ụ', 'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự',
    'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ',

    ' ', '.', ',', '!', '?', '-', '_', ':', ';', '"', "'",
    # fix: thêm @
    '(', ')', '[', ']', '/', '@',
    '#', '$', '%', '&', '*', '+', '=', '<', '>',
]

def get_tone_id(char: str) -> int:
    base = unicodedata.normalize("NFD", char)
    for c in base:
        if c == "\u0301": return 1
        if c == "\u0300": return 2
        if c == "\u0309": return 4
        if c == "\u0303": return 5
        if c == "\u0323": return 3
    return 0

def get_word_tone_score(word: str) -> int:
    max_rank = 0
    for char in word:
        t_id = get_tone_id(char)
        rank = TONE_RANK.get(t_id, 0)
        if rank > max_rank:
            max_rank = rank
    return max_rank


def _stable_hash_int(s: str) -> int:
    h = hashlib.md5(str(s).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def _normalize_text(s: str, lowercase: bool = True) -> str:
    s = unicodedata.normalize("NFC", str(s))
    s = s.strip(" .,:;!?\"'")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if lowercase else s


def _remove_vietnamese_accents(s: str) -> str:
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[oòóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[OÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[uùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[UÙÚỤỦŨƯỪỨỰỬỮ]', 'U', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[yỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[YỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[đ]', 'd', s)
    s = re.sub(r'[Đ]', 'D', s)
    return s


def _non_alnum_ratio(s: str) -> float:
    if not s:
        return 1.0
    non_alnum = sum(1 for ch in s if not ch.isalnum())
    return non_alnum / max(1, len(s))


def _is_repeated_runs(s: str) -> bool:
    return (
        bool(re.search(r"(.)\1{3,}", s))
        or bool(re.search(r"(.{1,3})\1{3,}", s))
    )


def _looks_like_code_garbage(s: str) -> bool:
    return (
        bool(re.fullmatch(r"[0-9a-f]{8,}", s))
        or bool(re.fullmatch(r"[A-Za-z0-9+/=]{12,}", s))
    )


def _char_diversity_low(s: str) -> bool:
    s2 = re.sub(r"\s+", "", s)
    return len(s2) >= 4 and len(set(s2)) <= 2

def setup_augmented_vocab(
    viet_vocab_path,
    eng_vocab_path,
    dataframe,
    output_dir="./dict",
    question_key="question",
    answer_key="answer",
    pretrain_vocab_path=None,
):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Viet_Vocab.txt")

    vocab_set = set()

    if pretrain_vocab_path and os.path.exists(pretrain_vocab_path):
        with open(pretrain_vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    vocab_set.add(w)

    answers = dataframe[answer_key].dropna().astype(str).tolist()
    questions = dataframe[question_key].dropna().astype(str).tolist()
    combined_text = answers + questions
    tokenizer_regex = re.compile(r"[\w]+")

    for text in combined_text:
        words = tokenizer_regex.findall(text.lower())
        for w in words:
            if w.isdigit():
                continue
            if len(w) > 25:
                continue
            if len(w) == 1:
                continue
            vocab_set.add(w)

    if viet_vocab_path and os.path.exists(viet_vocab_path):
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
                if not toks or len(toks) > 2:
                    continue
                for w in toks:
                    w = w.strip()
                    if not w or w.isdigit() or len(w) > 25 or len(w) == 1:
                        continue
                    vocab_set.add(w)

    if eng_vocab_path and os.path.exists(eng_vocab_path):
        with open(eng_vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    vocab_set.add(text.lower())

    whitelist = {
        "www", "http", "https", "com", "vn", "net", "org",
        "gmail", "email", "tnhh", "hcm", "tp", "hn", "sđt",
        "hotline", "fax", "website", "facebook", "youtube",
        "zalo", "tiktok", "instagram", "coffee", "cafe", "menu",
        "wifi", "pass", "password", "free", "vnd", "usd",
        "k", "kg", "km", "cm", "mm", "ml", "l",
        "b1", "b2", "a1", "a2", "c1", "c2",
        "ielts", "toeic", "vat", "vip", "qty", "total", "subtotal",
    }
    vocab_set.update(whitelist)

    final_vocab = sorted(vocab_set)
    with open(output_path, "w", encoding="utf-8") as f:
        for w in final_vocab:
            f.write(w + "\n")

    return output_path

