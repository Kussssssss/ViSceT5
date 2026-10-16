"""Build deterministic image/question/OCR batches for answer generation."""

import hashlib
import random
import re

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence

from data.vocab import (
    COMBINED_CHARS,
    _char_diversity_low,
    _is_repeated_runs,
    _looks_like_code_garbage,
    _non_alnum_ratio,
    _normalize_text,
)

RELATED_MODES = ("none", "correct", "twa")


def _stable_hash(text):
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "little")


class ViT5VQADataCollator:
    """OCR features stay aligned with their original image and text entries.

    related_mode:
      none    — OCR gốc (W entry).
      correct — thêm bản OCR đã sửa (kênh sửa OCR, xác định) → 2W entry.
      twa     — thêm bản tăng cường kiểu TWA + nhãn TWC (chỉ dùng khi train với TWC).
    Nửa related đứng sau nửa gốc và dùng chung box/det/rec của entry gốc tương ứng.
    """

    def __init__(self, tokenizer, image_processor, ocr_encoder, config,
                 corrector=None, related_mode="none", seed=42):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.ocr_encoder = ocr_encoder
        self.cfg = config
        self.pad_id = tokenizer.pad_token_id
        self.char_set = {char: index for index, char in enumerate(COMBINED_CHARS)}
        self.char_max_num = config.char_max_num
        self.seq_max = config.ocr_max_scene_text
        self.corrector = corrector
        self.related_mode = related_mode
        self.seed = int(seed)
        self.return_ocr_words = False  # chỉ dùng cho scripts/visualize_attention.py
        if related_mode not in RELATED_MODES:
            raise ValueError(f"related_mode must be one of {RELATED_MODES}.")
        if related_mode != "none" and corrector is None:
            raise ValueError("The related OCR copy needs an OCRCorrector.")
        if self.seq_max < 1 or self.char_max_num < 1:
            raise ValueError("OCR word and character limits must be positive.")
        if self.pad_id is None:
            raise ValueError("The tokenizer must define a padding token.")
        if tokenizer.unk_token_id is None:
            raise ValueError("The tokenizer must define an unknown token.")
        if config.char_num < len(COMBINED_CHARS):
            raise ValueError("char_num must cover the OCR character vocabulary.")

    @staticmethod
    def _box_iou(first, second):
        x0 = max(float(first[0]), float(second[0]))
        y0 = max(float(first[1]), float(second[1]))
        x1 = min(float(first[2]), float(second[2]))
        y1 = min(float(first[3]), float(second[3]))
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        area_a = float((first[2] - first[0]) * (first[3] - first[1]))
        area_b = float((second[2] - second[0]) * (second[3] - second[1]))
        return intersection / max(area_a + area_b - intersection, 1e-6)

    @staticmethod
    def _jaccard_tokens(first, second):
        first = set(re.findall(r"\w+", first.lower()))
        second = set(re.findall(r"\w+", second.lower()))
        return len(first & second) / len(first | second) if first and second else 0.0

    def _is_noise_text(self, text):
        return (
            not text
            or (len(text) < self.cfg.ocr_min_text_len_keep and not text.isdigit())
            or _non_alnum_ratio(text) > self.cfg.ocr_max_non_alnum_ratio
            or _is_repeated_runs(text)
            or _looks_like_code_garbage(text)
            or _char_diversity_low(text)
        )

    def _prepare_ocr(self, raw):
        keep, texts, seen = [], [], []
        for index, original in enumerate(raw["texts"]):
            text = _normalize_text(original, lowercase=self.cfg.ocr_lowercase)
            if self._is_noise_text(text):
                continue
            if len(self.tokenizer.encode(text, add_special_tokens=False)) > 25:
                continue
            box = raw["boxes"][index]
            duplicate = any(
                (text == previous or self._jaccard_tokens(text, previous) > 0.9)
                and self._box_iou(box, previous_box) > 0.5
                for previous, previous_box in seen
            )
            if duplicate:
                continue
            keep.append(index)
            texts.append(text)
            seen.append((text, box))
            if len(keep) == self.seq_max:
                break

        if keep:
            info = {
                key: raw[key][keep] for key in ("boxes", "det_features", "rec_features")
            }
        else:
            texts = [self.tokenizer.unk_token or "<unk>"]
            info = {
                "boxes": torch.zeros(1, 4),
                "det_features": torch.zeros(1, self.cfg.ocr_d_det),
                "rec_features": torch.zeros(1, self.cfg.ocr_d_rec),
            }
        info.update(texts=texts, width=raw["width"], height=raw["height"], placeholder=not keep)
        return info

    def _related(self, item, sample):
        """Related copy (+ TWC labels in twa mode). The <unk> placeholder is never edited."""
        texts = item["texts"]
        if item["placeholder"]:
            return list(texts), None, None
        if self.related_mode == "correct":
            return self.corrector.correct(texts), None, None
        # TWA: cố định theo mẫu (như seed tăng cường cố định của TWA) → tái lập được,
        # độc lập số worker DataLoader và PYTHONHASHSEED.
        rng = random.Random(_stable_hash(f"{self.seed}|{sample['image_path']}|{sample['question']}"))
        return self.corrector.augment(texts, self.cfg.twc_adv_probability, rng)

    def _encode_ocr_text(self, texts, word_count, offset=0):
        char_ids = torch.zeros(word_count, self.char_max_num, dtype=torch.long)
        char_mask = torch.zeros_like(char_ids, dtype=torch.float)
        token_ids, word_map = [], []
        for index, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                ids = [self.tokenizer.unk_token_id]
            token_ids.extend(ids)
            word_map.extend([offset + index] * len(ids))
            for position, char in enumerate(text[: self.char_max_num]):
                char_ids[index, position] = self.char_set.get(
                    char, self.char_set["<unk>"]
                )
                char_mask[index, position] = 1.0
        return token_ids, word_map, char_ids, char_mask

    def __call__(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch.")
        paths = [sample["image_path"] for sample in batch]
        images = []
        for path in paths:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode == "P" and "transparency" in image.info:
                    image = image.convert("RGBA")
                images.append(image.convert("RGB"))
        questions = self.tokenizer(
            [sample["question"] for sample in batch],
            padding="max_length",
            truncation=True,
            max_length=self.cfg.text_max_input_length,
            return_tensors="pt",
        )
        raw_ocr = self.ocr_encoder(
            paths, ocr_paths=[sample.get("ocr_path") for sample in batch]
        )
        info = [self._prepare_ocr(raw) for raw in raw_ocr]
        word_count = min(
            self.seq_max, ((max(len(item["texts"]) for item in info) + 7) // 8) * 8
        )
        with_related = self.related_mode != "none"
        twa = self.related_mode == "twa"
        token_ids, word_maps, char_ids, char_masks, word_masks = [], [], [], [], []
        o2r_all, r2o_all, groups, text_ids, words = [], [], [], [], []
        for sample, item in zip(batch, info):
            count = len(item["texts"])
            ids, mapping, chars, mask = self._encode_ocr_text(item["texts"], word_count)
            valid = torch.arange(word_count) < count
            for key in ("boxes", "det_features", "rec_features"):
                item[key] = torch.cat(
                    [item[key], item[key].new_zeros(word_count - count, item[key].shape[-1])]
                )
            related = None
            if with_related:
                related, o2r, r2o = self._related(item, sample)
                rel_ids, rel_map, rel_chars, rel_mask = self._encode_ocr_text(
                    related, word_count, offset=word_count
                )
                ids, mapping = ids + rel_ids, mapping + rel_map
                chars, mask = torch.cat([chars, rel_chars]), torch.cat([mask, rel_mask])
                valid = torch.cat([valid, valid])
                for key in ("boxes", "det_features", "rec_features"):
                    item[key] = torch.cat([item[key], item[key]])
                if twa:
                    # PAD và placeholder <unk> = -1 (không phải mẫu tương phản).
                    full_o2r = torch.full((word_count, word_count), -1.0)
                    full_r2o = torch.full((word_count, word_count), -1.0)
                    group = torch.full((word_count,), -1, dtype=torch.long)
                    texts_hashed = torch.full((2 * word_count,), -1, dtype=torch.long)
                    if o2r is not None:
                        full_o2r[:count, :count] = torch.tensor(o2r)
                        full_r2o[:count, :count] = torch.tensor(r2o)
                        group[:count] = torch.tensor(
                            [_stable_hash(f"{sample['image_path']}|{text}") & 0x3FFFFFFF
                             for text in item["texts"]]
                        )
                        texts_hashed[:count] = torch.tensor(
                            [_stable_hash(text) & 0x3FFFFFFF for text in item["texts"]]
                        )
                        texts_hashed[word_count:word_count + count] = torch.tensor(
                            [_stable_hash(text) & 0x3FFFFFFF for text in related]
                        )
                    o2r_all.append(full_o2r)
                    r2o_all.append(full_r2o)
                    groups.append(group)
                    text_ids.append(texts_hashed)
            token_ids.append(torch.tensor(ids, dtype=torch.long))
            word_maps.append(torch.tensor(mapping, dtype=torch.long))
            char_ids.append(chars)
            char_masks.append(mask)
            word_masks.append(valid)
            words.append({"orig": list(item["texts"]), "rel": related})
        ids = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_id)
        result = {
            "input_ids": questions.input_ids,
            "attention_mask": questions.attention_mask,
            "pixel_values": self.image_processor(images=images, return_tensors="pt")[
                "pixel_values"
            ],
            "ocr_info": info,
            "ocr_input_ids": ids,
            "ocr_attention_mask": (ids != self.pad_id).long(),
            "ocr_to_word_map": pad_sequence(
                word_maps, batch_first=True, padding_value=-1
            ),
            "ocr_char_ids": torch.stack(char_ids),
            "ocr_char_mask": torch.stack(char_masks),
            "ocr_word_mask": torch.stack(word_masks).long(),
        }
        if twa:
            result.update(
                twc_o2r=torch.stack(o2r_all),
                twc_r2o=torch.stack(r2o_all),
                twc_group_ids=torch.stack(groups),
                twc_text_ids=torch.stack(text_ids),
            )
        if self.return_ocr_words:
            result["ocr_words"] = words
        # Unlabelled inference batches do not create an all-ignored decoder loss.
        answers = [sample.get("answer") for sample in batch]
        if all(answer is not None and str(answer).strip() for answer in answers):
            labels = self.tokenizer(
                answers,
                padding="max_length",
                truncation=True,
                max_length=self.cfg.text_max_target_length,
                return_tensors="pt",
            ).input_ids
            result["labels"] = labels.masked_fill(labels == self.pad_id, -100)
        elif any(answer is not None and str(answer).strip() for answer in answers):
            raise ValueError(
                "A batch must be entirely labelled or entirely unlabelled."
            )
        return result
