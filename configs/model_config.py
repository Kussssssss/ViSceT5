"""
configs/model_config.py
OpenViVQAConfig — HuggingFace PretrainedConfig for the full OpenViVQA model.
"""

from typing import Optional, List
from transformers import PretrainedConfig

# ─────────────────────────────────────────────────────────
# Character vocabulary constants
# ─────────────────────────────────────────────────────────
CHAR_SPECIAL_TOKENS = [" ", "<s>", "</s>", "<unk>"]
VIET_ALPHA = list(
    "aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghijklmnopqrstuvwxyz"
    "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬBCDĐEÉÈẺẼẸÊẾỀỂỄỆFGHIJKLMNOPQRSTUVWXYZ"
)
DIGITS_PUNC = list("0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~–—''•…€£¥₹©®™°")

COMBINED_CHARS = CHAR_SPECIAL_TOKENS + list(dict.fromkeys(VIET_ALPHA + DIGITS_PUNC))
DEFAULT_CHAR_NUM = len(COMBINED_CHARS)


class OpenViVQAConfig(PretrainedConfig):
    model_type = "openvivqa"

    def __init__(
        self,
        vit5_name: str = "VietAI/vit5-base",
        clip_vision_name: str = "openai/clip-vit-base-patch16",
        ablation_use_qaclip: bool = True,
        ablation_use_ocr: bool = True,
        qa_clip_d_text: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        ocr_d_det: int = 256,
        ocr_d_rec: int = 256,
        ocr_max_scene_text: int = 180,
        ocr_num_distances: int = 32,
        ocr_max_2d_position_embeddings: int = 1024,
        ocr_cuda_device: str = "cuda:0",
        text_max_input_length: int = 32,
        text_max_target_length: int = 56,
        ocr_lowercase: bool = True,
        ocr_max_non_alnum_ratio: float = 0.6,
        ocr_min_text_len_keep: int = 2,
        ocr_jaccard_dupe: float = 0.78,
        char_max_num: int = 50,
        char_num: int = DEFAULT_CHAR_NUM,
        generation_max_new_tokens: int = 56,
        generation_num_beams: int = 4,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_resize: bool = True,
        do_center_crop: bool = False,
        image_size: Optional[int] = None,
        ocrseq_margin_frac: float = 0.14,
        ocrseq_pool_size: int = 3,
        ocrseq_use_spatial: bool = True,
        ocrseq_dropout: float = 0.1,
        **kwargs,
    ):
        # Migrate image preprocessing from checkpoints that included Visual Search.
        legacy_image_size = kwargs.pop("vs_target_size", None)
        if image_size is None:
            image_size = legacy_image_size
        for name in (
            "vs_backbone",
            "ablation_use_vs",
            "vs_percentile",
            "vs_min_area",
            "vs_dilate_half_patch",
            "vs_patch_size",
            "vs_max_cov",
            "vs_margin",
        ):
            kwargs.pop(name, None)
        super().__init__(**kwargs)

        self.vit5_name = vit5_name
        self.clip_vision_name = clip_vision_name
        self.ablation_use_qaclip = bool(ablation_use_qaclip)
        self.ablation_use_ocr = bool(ablation_use_ocr)

        self.qa_clip_d_text = qa_clip_d_text
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id

        self.ocr_d_det = int(ocr_d_det)
        self.ocr_d_rec = int(ocr_d_rec)
        self.ocr_max_scene_text = int(ocr_max_scene_text)
        self.ocr_num_distances = int(ocr_num_distances)
        self.ocr_max_2d_position_embeddings = int(ocr_max_2d_position_embeddings)
        self.ocr_cuda_device = str(ocr_cuda_device)

        self.text_max_input_length = int(text_max_input_length)
        self.text_max_target_length = int(text_max_target_length)

        self.ocr_lowercase = bool(ocr_lowercase)
        self.ocr_max_non_alnum_ratio = float(ocr_max_non_alnum_ratio)
        self.ocr_min_text_len_keep = int(ocr_min_text_len_keep)
        self.ocr_jaccard_dupe = float(ocr_jaccard_dupe)

        self.char_max_num = int(char_max_num)
        self.char_num = int(char_num)

        self.generation_max_new_tokens = int(generation_max_new_tokens)
        self.generation_num_beams = int(generation_num_beams)

        self.image_mean = list(
            image_mean
            if image_mean is not None
            else [0.48145466, 0.4578275, 0.40821073]
        )
        self.image_std = list(
            image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        )

        self.do_resize = bool(do_resize)
        self.do_center_crop = bool(do_center_crop)
        self.image_size = None if image_size is None else int(image_size)

        self.ocrseq_margin_frac = float(ocrseq_margin_frac)
        self.ocrseq_pool_size = int(ocrseq_pool_size)
        self.ocrseq_use_spatial = bool(ocrseq_use_spatial)
        self.ocrseq_dropout = float(ocrseq_dropout)
