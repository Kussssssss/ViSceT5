"""
configs/ocr_config.py
Default OCR embedding configuration used by Vision_Encode_Ocr_Feature.
"""

DEFAULT_OCR_CONFIG = {
    "ocr_embedding": {
        "sort_type": "top-left bottom-right",
        "path_ocr": None,
        "threshold": 0.3,
        "remove_accents_rate": 0,
        "use_word_seg": False,
        "max_scene_text": 180,
        "d_det": 256,
        "d_rec": 256,
        "max_2d_position_embeddings": 1024,
        "num_distances": 32,
    },
}
