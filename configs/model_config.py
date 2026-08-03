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
# NOTE: giữ NGUYÊN 4 smart-quote Unicode ‘ ’ “ ” (KHÔNG để editor tự đổi thành
# ASCII ' ' " "). Nếu bị đổi, chúng trùng với ' và " đã có → dedupe mất 4 ký tự →
# DEFAULT_CHAR_NUM tụt 172→168, lệch shape char_embedding so với notebook gốc.
DIGITS_PUNC = list(
    "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~–—‘’“”•…€£¥₹©®™°"
)

COMBINED_CHARS = CHAR_SPECIAL_TOKENS + list(dict.fromkeys(VIET_ALPHA + DIGITS_PUNC))
DEFAULT_CHAR_NUM = len(COMBINED_CHARS)


class OpenViVQAConfig(PretrainedConfig):
    model_type = "openvivqa"

    def __init__(
        self,
        vit5_name: str = "VietAI/vit5-base",
        clip_vision_name: str = "openai/clip-vit-base-patch16",
        vs_backbone: str = "facebook/convnextv2-tiny-22k-224",
        bootstrap_from_pretrained: bool = True,
        local_submodule_dir: Optional[str] = None,
        pretrain: bool = True,
        pretrain_mask_prob: float = 0.15,
        pretrain_mask_seed: int = 42,
        qa_clip_d_text: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        mlm_ignore_index: int = -100,
        contrastive_ignore_value: float = -1.0,
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
        adv_probability: float = 1.0,
        contrastive_label_list: List[float] = [0.9, 0.9],
        editlen: int = 2,
        term_vocab_path: str = "/kaggle/input/term-vietnamese-vocab/words.txt",
        adv_probability_pretrain: float = 0.35,
        adv_probability_finetune: float = 1.0,
        generation_max_new_tokens: int = 56,
        generation_num_beams: int = 4,
        vs_percentile: int = 99,
        vs_min_area: int = 1,
        vs_dilate_half_patch: bool = True,
        vs_patch_size: int = 16,
        vs_max_cov: float = 1.0,
        vs_margin: int = 0,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_resize: bool = True,
        do_center_crop: bool = False,
        vs_target_size: Optional[int] = 224,
        ocrseq_margin_frac: float = 0.14,
        ocrseq_pool_size: int = 3,
        ocrseq_use_spatial: bool = True,
        ocrseq_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vit5_name = vit5_name
        self.clip_vision_name = clip_vision_name
        self.vs_backbone = vs_backbone
        self.bootstrap_from_pretrained = bool(bootstrap_from_pretrained)
        self.local_submodule_dir = local_submodule_dir

        self.pretrain = bool(pretrain)
        self.pretrain_mask_prob = float(pretrain_mask_prob)
        self.pretrain_mask_seed = int(pretrain_mask_seed)

        self.qa_clip_d_text = qa_clip_d_text
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id

        self.mlm_ignore_index = int(mlm_ignore_index)
        self.contrastive_ignore_value = float(contrastive_ignore_value)

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
        self.adv_probability = float(adv_probability)
        self.contrastive_label_list = (
            list(contrastive_label_list)
            if contrastive_label_list is not None
            else [0.9, 0.9]
        )
        self.editlen = int(editlen)
        self.term_vocab_path = str(term_vocab_path)

        self.adv_probability_pretrain = float(adv_probability_pretrain)
        self.adv_probability_finetune = float(adv_probability_finetune)

        self.generation_max_new_tokens = int(generation_max_new_tokens)
        self.generation_num_beams = int(generation_num_beams)

        self.vs_percentile = int(vs_percentile)
        self.vs_min_area = int(vs_min_area)
        self.vs_dilate_half_patch = bool(vs_dilate_half_patch)
        self.vs_patch_size = int(vs_patch_size)
        self.vs_max_cov = float(vs_max_cov)
        self.vs_margin = int(vs_margin)

        self.image_mean = list(
            image_mean
            if image_mean is not None
            else [0.48145466, 0.4578275, 0.40821073]
        )
        self.image_std = list(
            image_std
            if image_std is not None
            else [0.26862954, 0.26130258, 0.27577711]
        )

        self.do_resize = bool(do_resize)
        self.do_center_crop = bool(do_center_crop)
        self.vs_target_size = None if vs_target_size is None else int(vs_target_size)

        self.ocrseq_margin_frac = float(ocrseq_margin_frac)
        self.ocrseq_pool_size = int(ocrseq_pool_size)
        self.ocrseq_use_spatial = bool(ocrseq_use_spatial)
        self.ocrseq_dropout = float(ocrseq_dropout)
