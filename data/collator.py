"""Build deterministic image/question/OCR batches for answer generation."""

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

# TWA builds the related copy of every sample with random.Random(13) (findRelatedOCR*).
TWA_SEED = 13


class ViT5VQADataCollator:
    """OCR features stay aligned with their original image and text entries.

    related: also build the related OCR copy as its own sequence ocr_rel_* (word slots 0..W-1,
    sharing box/det/rec with the original word), by related_mode:
      augment — TWA keep / noise / correct with probability twc_adv_probability (training);
      correct — every correctable word corrected, no noise (evaluation / test).
    twc_labels: emit the TWC labels twc_o2r / twc_r2o (training, augment only).
    link_labels: emit ocr_link_labels, the target of the ConsFormer links (training).
    """

    def __init__(self, tokenizer, image_processor, ocr_encoder, config,
                 corrector=None, related=False, twc_labels=False, related_mode="augment",
                 link_labels=False):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.ocr_encoder = ocr_encoder
        self.cfg = config
        self.pad_id = tokenizer.pad_token_id
        self.char_set = {char: index for index, char in enumerate(COMBINED_CHARS)}
        self.char_max_num = config.char_max_num
        self.seq_max = config.ocr_max_scene_text
        self.corrector = corrector
        self.related = bool(related)
        self.twc_labels = bool(twc_labels)
        self.link_labels = bool(link_labels)
        self.related_mode = str(related_mode)
        if self.related_mode not in ("augment", "correct"):
            raise ValueError("related_mode must be augment|correct.")
        if self.twc_labels and self.related_mode != "augment":
            raise ValueError("TWC labels come from the TWA augmentation (related_mode='augment').")
        self.return_ocr_words = False  # chỉ dùng cho scripts/visualize_attention.py
        if self.related and corrector is None:
            raise ValueError("The related OCR copy needs an OCRCorrector.")
        if self.twc_labels and not self.related:
            raise ValueError("TWC labels need the related OCR copy.")
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

    def _related(self, item):
        """Related copy + TWC labels (TWA findRelatedOCR_adr). The <unk> placeholder of an image
        without OCR is never edited and gets no labels (all -1, as TWA's PAD)."""
        texts = item["texts"]
        if item["placeholder"]:
            return list(texts), None, None
        if self.related_mode == "correct":
            return self.corrector.correct(texts), None, None
        return self.corrector.augment(texts, self.cfg.twc_adv_probability, random.Random(TWA_SEED))

    def _encode_ocr_text(self, texts, word_count):
        char_ids = torch.zeros(word_count, self.char_max_num, dtype=torch.long)
        char_mask = torch.zeros_like(char_ids, dtype=torch.float)
        token_ids, word_map = [], []
        for index, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                ids = [self.tokenizer.unk_token_id]
            token_ids.extend(ids)
            word_map.extend([index] * len(ids))
            for position, char in enumerate(text[: self.char_max_num]):
                char_ids[index, position] = self.char_set.get(
                    char, self.char_set["<unk>"]
                )
                char_mask[index, position] = 1.0
        return token_ids, word_map, char_ids, char_mask

    def _link_targets(self, ids, word_map, answer, length):
        """Target of the link between token j-1 and j of the original OCR, -1 = no target: 1 inside
        a word and inside an occurrence of the answer, 0 where an occurrence starts or ends."""
        targets = torch.full((length,), -1.0)
        same_word = word_map[1:] == word_map[:-1]
        targets[1 : len(ids)][same_word] = 1.0
        text = _normalize_text(str(answer), lowercase=self.cfg.ocr_lowercase)
        span = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = ids.tolist()
        for start in range(len(tokens) - len(span) + 1 if span else 0):
            if tokens[start : start + len(span)] == span:
                for edge in (start, start + len(span)):
                    if 0 < edge < len(tokens) and word_map[edge] != word_map[edge - 1]:
                        targets[edge] = 0.0
        for start in range(len(tokens) - len(span) + 1 if span else 0):
            if tokens[start : start + len(span)] == span:
                targets[start + 1 : start + len(span)] = 1.0
        return targets

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
        token_ids, word_maps, char_ids, char_masks, word_masks = [], [], [], [], []
        rel_token_ids, rel_word_maps, rel_char_ids, rel_char_masks = [], [], [], []
        o2r_all, r2o_all, words = [], [], []
        for item in info:
            count = len(item["texts"])
            ids, mapping, chars, mask = self._encode_ocr_text(item["texts"], word_count)
            for key in ("boxes", "det_features", "rec_features"):
                item[key] = torch.cat(
                    [item[key], item[key].new_zeros(word_count - count, item[key].shape[-1])]
                )
            related = None
            if self.related:
                related, o2r, r2o = self._related(item)
                rel_ids, rel_map, rel_chars, rel_mask = self._encode_ocr_text(related, word_count)
                rel_token_ids.append(torch.tensor(rel_ids, dtype=torch.long))
                rel_word_maps.append(torch.tensor(rel_map, dtype=torch.long))
                rel_char_ids.append(rel_chars)
                rel_char_masks.append(rel_mask)
                if self.twc_labels:
                    # PAD và placeholder <unk> = -1 (không phải mẫu tương phản), như TWA.
                    full_o2r = torch.full((word_count, word_count), -1.0)
                    full_r2o = torch.full((word_count, word_count), -1.0)
                    if o2r is not None:
                        full_o2r[:count, :count] = torch.tensor(o2r)
                        full_r2o[:count, :count] = torch.tensor(r2o)
                    o2r_all.append(full_o2r)
                    r2o_all.append(full_r2o)
            token_ids.append(torch.tensor(ids, dtype=torch.long))
            word_maps.append(torch.tensor(mapping, dtype=torch.long))
            char_ids.append(chars)
            char_masks.append(mask)
            word_masks.append(torch.arange(word_count) < count)
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
        if self.related:
            rel_padded = pad_sequence(rel_token_ids, batch_first=True, padding_value=self.pad_id)
            result.update(
                ocr_rel_input_ids=rel_padded,
                ocr_rel_attention_mask=(rel_padded != self.pad_id).long(),
                ocr_rel_to_word_map=pad_sequence(rel_word_maps, batch_first=True, padding_value=-1),
                ocr_rel_char_ids=torch.stack(rel_char_ids),
                ocr_rel_char_mask=torch.stack(rel_char_masks),
            )
        if self.twc_labels:
            result.update(twc_o2r=torch.stack(o2r_all), twc_r2o=torch.stack(r2o_all))
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
            if self.link_labels:
                result["ocr_link_labels"] = torch.stack([
                    self._link_targets(sample_ids, sample_map, answer, ids.shape[1])
                    for sample_ids, sample_map, answer in zip(token_ids, word_maps, answers)
                ])
        elif any(answer is not None and str(answer).strip() for answer in answers):
            raise ValueError(
                "A batch must be entirely labelled or entirely unlabelled."
            )
        return result
