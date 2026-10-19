"""
scripts/port_old_checkpoint.py
Convert a checkpoint trained with the pre-refactor code (e.g. commit 7b99d60, the Vast run
Kus669/ViSceT5-ft-vitextvqa-only_consformer) into this branch's format, so that this
branch's preprocessing, forward pass and predict.py can be checked against the old
submission: if predicting the ported weights gives the old test score, everything between
the data files and the generated answer is equivalent, and any gap in a new run comes from
training, not from the code that is run.

    python scripts/port_old_checkpoint.py --repo Kus669/ViSceT5-ft-vitextvqa-only_consformer \
        --ckpt checkpoint-21975 --out /content/ported_only_consformer

Mapping (old → new):
  * vit5.*, qa_clip.*, ocr_encoder.*, semantic_ocr_embedding.*, char_position_embedding,
    ocr_char_layernorm: same names and shapes. The new model is built with the pre-refactor
    switches (ocr_feature_norm=layernorm, clip_late_adapter=True), so the OCR LayerNorms
    and the gated late-CLIP adapter exist and load as trained.
  * char_embedding: the old collator indexed 128 characters from 0 ('a') and sent every
    unknown character to id 1 ('b'); the new vocabulary has '<unk>' and '@' in front. Rows
    are copied per character; '<unk>' and '@' (unknown to the old code) get the old row 1.
  * dropped: modules the old code built but never used for this configuration
    (visual_search, ocr_lite_*, pollute_head, spatial_embedding, logit_scale, ...).
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from configs.model_config import OpenViVQAConfig
from data.vocab import COMBINED_CHARS
from models.openvivqa_model import OpenViVQAModel

# data/vocab.py COMBINED_CHARS at commit 7b99d60 (collator character ids of the old code).
OLD_CHARS = (
    list("abcdefghijklmnopqrstuvwxyz0123456789")
    + list("àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ")
    + [" ", ".", ",", "!", "?", "-", "_", ":", ";", '"', "'", "(", ")", "[", "]", "/",
       "#", "$", "%", "&", "*", "+", "=", "<", ">"]
)
OLD_UNKNOWN = 1  # old collator: char_set.get(c, char_set.get("<unk>", 1))
KEEP = ("vit5.", "qa_clip.", "ocr_encoder.", "semantic_ocr_embedding.", "char_position_embedding.",
        "ocr_char_layernorm.")
SHARED = ("ocr_max_scene_text", "ocr_min_text_len_keep", "ocr_max_non_alnum_ratio", "ocr_jaccard_dupe",
          "ocr_lowercase", "char_max_num", "text_max_input_length", "text_max_target_length",
          "generation_max_new_tokens", "generation_num_beams", "ocr_d_det", "ocr_d_rec",
          "vit5_name", "clip_vision_name", "image_mean", "image_std", "do_resize", "do_center_crop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", default="", help="sub-folder, e.g. checkpoint-21975 ('' = repo root)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(OLD_CHARS) == 128, len(OLD_CHARS)

    prefix = f"{a.ckpt}/" if a.ckpt else ""
    src = snapshot_download(a.repo, repo_type="model", token=os.environ.get("HF_TOKEN") or None,
                            allow_patterns=[f"{prefix}{p}" for p in ("config.json", "generation_config.json",
                                                                       "model.safetensors", "tokenizer*",
                                                                       "special_tokens_map.json", "spiece.model")])
    src = Path(src) / a.ckpt if a.ckpt else Path(src)
    old_cfg = json.load(open(src / "config.json", encoding="utf-8"))
    print("old ablation flags:", {k: old_cfg.get(k) for k in ("ablation_use_qaclip", "ablation_use_vs",
                                                            "ablation_use_ocr", "ablation_use_ocr_aug")})
    if old_cfg.get("ablation_use_vs"):
        sys.exit("Visual Search does not exist in this branch; only configurations without it can be ported.")

    config = OpenViVQAConfig(
        ablation_use_image=True,
        ablation_use_qaclip=bool(old_cfg.get("ablation_use_qaclip")),
        ablation_use_ocr=bool(old_cfg.get("ablation_use_ocr", True)),
        ablation_use_ocr_aug=bool(old_cfg.get("ablation_use_ocr_aug")),
        use_twc=False,
        ocr_feature_norm="layernorm",
        clip_late_adapter=True,
        **{k: old_cfg[k] for k in SHARED if k in old_cfg},
    )
    model = OpenViVQAModel(config)
    new_state = model.state_dict()
    old_state = load_file(str(src / "model.safetensors"))

    ported, skipped = {}, []
    for key, value in old_state.items():
        if key.startswith(KEEP) and key in new_state:
            if tuple(value.shape) != tuple(new_state[key].shape):
                sys.exit(f"shape mismatch {key}: old {tuple(value.shape)} vs new {tuple(new_state[key].shape)}")
            ported[key] = value
        elif key != "char_embedding.weight":
            skipped.append(key)
    if "char_embedding.weight" in old_state:
        old_rows = old_state["char_embedding.weight"]
        rows = new_state["char_embedding.weight"].clone()
        for index, char in enumerate(COMBINED_CHARS):
            rows[index] = old_rows[OLD_CHARS.index(char) if char in OLD_CHARS else OLD_UNKNOWN]
        ported["char_embedding.weight"] = rows

    missing, unexpected = model.load_state_dict(ported, strict=False)
    # Only the tied copies of the shared embedding may be absent from the old file.
    tied = {"vit5.encoder.embed_tokens.weight", "vit5.decoder.embed_tokens.weight", "vit5.lm_head.weight"}
    real_missing = [k for k in missing if k not in tied]
    groups = sorted({re.sub(r"\.\d+\.", ".N.", k).split(".")[0] for k in skipped})
    print(f"ported {len(ported)} tensors | dropped old-only modules: {groups}")
    if real_missing or unexpected:
        sys.exit(f"❌ incomplete port: missing={real_missing[:20]} unexpected={unexpected[:20]}")
    gates = [float(v) for k, v in ported.items() if k.endswith("instruction_proj_gate")]
    print(f"late-CLIP adapter gates (trained): {[round(g, 4) for g in gates]}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    model.image_processor.save_pretrained(out / "image_processor")
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    print(f"✅ wrote {out} (predict with: STAGE=predict PREDICT_CKPT_DIR={out})")


if __name__ == "__main__":
    main()
