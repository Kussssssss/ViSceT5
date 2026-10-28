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


# Switches of older code. Their "off" values match this code; any other value is refused.
LEGACY_FIELDS = (
    "clip_late_adapter", "ablation_use_ocr_aug", "ocr_aug_mode", "ocr_related", "ocr_lattice_keep",
    "twc_loss", "twc_feature", "twc_queue_size", "twc_queue_start", "twc_mask_same_text",
    "twc_objective", "twc_answer_weight", "twc_related_train", "twc_view",
)


class OpenViVQAConfig(PretrainedConfig):
    model_type = "openvivqa"

    def __init__(
        self,
        vit5_name: str = "VietAI/vit5-base",
        clip_vision_name: str = "openai/clip-vit-base-patch16",
        ablation_use_qaclip: bool = False,
        ablation_use_ocr: bool = True,
        ablation_use_image: bool = True,
        use_twc: bool = False,
        twc_weight: float = 1.0,
        twc_adv_probability: float = 0.35,
        twc_label_list: Optional[List[float]] = None,
        twc_temperature_init: float = 0.07,
        ocr_feature_norm: Optional[str] = None,
        ocr_full_features: Optional[bool] = None,
        twc_eval_related: Optional[str] = None,
        consformer_constituent: Optional[str] = None,
        ocr_edit_distance: int = 2,
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
            "use_visual_search",
            "ablation_use_vs",
            "vs_percentile",
            "vs_min_area",
            "vs_dilate_half_patch",
            "vs_patch_size",
            "vs_max_cov",
            "vs_margin",
        ):
            kwargs.pop(name, None)
        legacy = {name: kwargs.pop(name, None) for name in LEGACY_FIELDS}
        if (legacy["clip_late_adapter"] or legacy["ablation_use_ocr_aug"]
                or legacy["twc_objective"] in ("answer", "both")):
            raise ValueError(
                "This configuration comes from older code (QA-ViT without question, OCR Augmentation "
                "as a separate module, or answer TWC), which this code no longer builds; "
                "use commit bf74388 for it."
            )
        super().__init__(**kwargs)

        self.vit5_name = vit5_name
        self.clip_vision_name = clip_vision_name
        # QA-ViT: the question goes into the late CLIP layers through the learned adapter.
        # False = plain frozen CLIP, no question and no adapter.
        self.ablation_use_qaclip = bool(ablation_use_qaclip)
        # Consformer: the ConsFormer contextual encoder of the OCR tokens. False = the ViT5 token
        # embeddings instead (box/det/rec/char then depend on ocr_full_features).
        self.ablation_use_ocr = bool(ablation_use_ocr)
        # Bỏ ảnh: CLIP không được dựng, encoder chỉ nhận [câu hỏi, OCR].
        self.ablation_use_image = bool(ablation_use_image)
        # TWC (TWA, Jin et al. MM'22): the related OCR copy (TWA augmentation, OCR Augmentation)
        # enters the encoder after the original OCR, at train and at test; the auxiliary loss
        # contrasts the encoder outputs of the two copies during training.
        self.use_twc = bool(use_twc)
        self.twc_weight = float(twc_weight)
        # TWA adv_pro: probability that a word of the related copy is changed
        # (in-lexicon word → one character edit, other word → lexicon correction).
        self.twc_adv_probability = float(twc_adv_probability)
        if not 0.0 <= self.twc_adv_probability <= 1.0:
            raise ValueError("twc_adv_probability must lie in [0, 1].")
        # TWA label_list: [o2r, r2o] of a changed word; an unchanged word = 1.0.
        self.twc_label_list = list(twc_label_list) if twc_label_list is not None else [0.9, 0.9]
        # Initial TWC temperature τ (learnable: logit scale = exp(θ), θ0 = log(1/τ)); TWA: 0.07.
        self.twc_temperature_init = float(twc_temperature_init)
        if self.twc_temperature_init <= 0:
            raise ValueError("twc_temperature_init must be positive.")
        # Norm of each OCR stream (box/det/rec/token/char) in SemanticOCREmbedding.
        # layernorm = pre-refactor code; None (configs saved before the field) = rmsnorm.
        self.ocr_feature_norm = str(ocr_feature_norm or "rmsnorm")
        if self.ocr_feature_norm not in ("layernorm", "rmsnorm"):
            raise ValueError("ocr_feature_norm must be layernorm|rmsnorm.")
        # OCR input features of every run, Consformer on or off: token + char + box + det + rec
        # (SemanticOCREmbedding). None (configs before this field) = False: without Consformer the
        # OCR was text-only (ViT5 embedding → RMSNorm → Linear), so those checkpoints load as trained.
        self.ocr_full_features = bool(ocr_full_features) if ocr_full_features is not None else False
        # Related OCR copy at evaluation / test (training always uses the TWA augmentation):
        # correct = every correctable word corrected, no character noise (the noise only trains
        # TWC, it carries no information about the answer); augment = the training procedure.
        # None (configs before this field) = augment, so those checkpoints predict as trained.
        self.twc_eval_related = str(twc_eval_related or "augment")
        if self.twc_eval_related not in ("augment", "correct"):
            raise ValueError("twc_eval_related must be augment|correct.")
        # ConsFormer constituent prior: tree = Tree Transformer / ViConsFormer, C[i, j] =
        # prod_{k=i}^{j-1} a[k, k+1], + output projection; legacy = the code before this field,
        # whose prior is ~0 off the diagonal (gated self-attention ~ off). None (older configs) =
        # legacy, so those checkpoints load and predict as trained.
        self.consformer_constituent = str(consformer_constituent or "legacy")
        if self.consformer_constituent not in ("tree", "legacy"):
            raise ValueError("consformer_constituent must be tree|legacy.")
        self.ocr_edit_distance = int(ocr_edit_distance)

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
