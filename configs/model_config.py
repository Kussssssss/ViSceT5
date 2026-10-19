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
        ablation_use_qaclip: bool = False,
        ablation_use_ocr: bool = True,
        ablation_use_image: bool = True,
        ablation_use_ocr_aug: bool = True,
        use_twc: bool = False,
        twc_weight: float = 1.0,
        twc_adv_probability: float = 0.35,
        twc_label_list: Optional[List[float]] = None,
        twc_loss: str = "infonce",
        twc_feature: str = "lexical",
        twc_queue_size: int = 16384,
        twc_queue_start: int = 1000,
        twc_mask_same_text: bool = True,
        twc_objective: str = "answer",
        twc_answer_weight: float = 0.1,
        twc_related_train: str = "correct",
        ocr_feature_norm: Optional[str] = None,
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
        super().__init__(**kwargs)

        self.vit5_name = vit5_name
        self.clip_vision_name = clip_vision_name
        self.ablation_use_qaclip = bool(ablation_use_qaclip)
        self.ablation_use_ocr = bool(ablation_use_ocr)
        # Bỏ ảnh: CLIP không được dựng, encoder chỉ nhận [câu hỏi, OCR].
        self.ablation_use_image = bool(ablation_use_image)
        # Kênh sửa OCR: nối bản OCR đã sửa (related) sau bản gốc.
        self.ablation_use_ocr_aug = bool(ablation_use_ocr_aug)
        # TWC (TWA, Jin et al. MM'22): loss phụ khi train, cần nửa related → cần OCR-Aug.
        self.use_twc = bool(use_twc)
        self.twc_weight = float(twc_weight)
        self.twc_adv_probability = float(twc_adv_probability)
        # [o2r, r2o] cho cặp bị tăng cường (nhiễu ký tự / sửa từ); cặp giữ nguyên = 1.0.
        self.twc_label_list = list(twc_label_list) if twc_label_list is not None else [0.9, 0.9]
        self.ocr_edit_distance = int(ocr_edit_distance)
        # infonce: softmax over all in-batch candidates with soft multi-positive targets
        #          (TWA paper Eq. 4–6); bce: sigmoid BCE averaged over all cells (TWA code).
        self.twc_loss = str(twc_loss)
        # lexical: OCR features before fusion (ConsFormer text + characters, no box/det/rec);
        # fused: multimodal encoder output. The related copy shares box/det/rec with its
        # original, so on fused features a pair can be matched by layout alone.
        self.twc_feature = str(twc_feature)
        if self.twc_loss not in ("infonce", "bce") or self.twc_feature not in ("lexical", "fused"):
            raise ValueError("twc_loss must be infonce|bce and twc_feature lexical|fused.")
        # Cross-batch memory (XBM, Wang et al. CVPR'20): FIFO of detached word features
        # from previous steps, used as extra negatives for the softmax TWC. A batch of 8
        # gives ~300 candidates; 16384 entries cover the last ~55 steps. 0 = in-batch only.
        # The memory is read only after twc_queue_start training forwards, once features
        # drift slowly (it is filled from the start, so it is full when first read).
        self.twc_queue_size = int(twc_queue_size)
        self.twc_queue_start = int(twc_queue_start)
        # A candidate whose text (original or related) equals the query's original or related
        # text is never a negative: identical strings get identical lexical features, so
        # pushing them apart is a false negative. Within an image same-text pairs stay
        # positives (TWA label inheritance); every other text collision is ignored.
        self.twc_mask_same_text = bool(twc_mask_same_text)
        if self.twc_queue_size < 0 or self.twc_queue_start < 0:
            raise ValueError("twc_queue_size and twc_queue_start must be >= 0.")
        if self.use_twc and self.twc_queue_size > 0 and self.twc_loss != "infonce" and twc_objective != "answer":
            raise ValueError("The TWC memory adds negatives to the softmax loss; use twc_loss=infonce.")
        # answer : question ↔ OCR words of the answer (answer-grounded contrast on the fused
        #          encoder output; candidates = every OCR word of the batch, original and
        #          corrected copies). Supervises WHERE the answer is, which the decoder copies.
        # lexical: original ↔ related copy of each OCR word (TWA word-level contrast).
        # both   : sum of the two (weights twc_answer_weight and twc_weight).
        # Norm of each OCR stream (box/det/rec/token/char) in SemanticOCREmbedding.
        # layernorm = pre-refactor code (the Vast/old runs); rmsnorm = refactor. None (configs
        # saved before this switch existed) = rmsnorm, so those checkpoints still load as trained.
        self.ocr_feature_norm = str(ocr_feature_norm or "rmsnorm")
        if self.ocr_feature_norm not in ("layernorm", "rmsnorm"):
            raise ValueError("ocr_feature_norm must be layernorm|rmsnorm.")
        self.twc_objective = str(twc_objective)
        self.twc_answer_weight = float(twc_answer_weight)
        # Related OCR copy at TRAIN time. correct = the deterministic correction used at test
        # (train/test identical input); twa = TWA augmentation (noise / correction, p=0.35),
        # which feeds the VQA loss a related half it never sees at test.
        self.twc_related_train = str(twc_related_train)
        if self.twc_objective not in ("answer", "lexical", "both"):
            raise ValueError("twc_objective must be answer|lexical|both.")
        if self.twc_related_train not in ("correct", "twa"):
            raise ValueError("twc_related_train must be correct|twa.")
        if self.use_twc and self.twc_objective != "answer" and not self.ablation_use_ocr_aug:
            raise ValueError("Lexical TWC pairs each OCR word with its related copy: needs ablation_use_ocr_aug=True.")

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
