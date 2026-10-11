"""Shared configuration, dataset and checkpoint setup for VQA entry points."""

import sys
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

from configs.base_config import OUTPUT_PATH
from configs.model_config import OpenViVQAConfig
from configs.ocr_config import DEFAULT_OCR_CONFIG
from data.collator import ViT5VQADataCollator
from models.openvivqa_model import OpenViVQAModel
from models.modules.ocr_encoder_feature import Vision_Encode_Ocr_Feature

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args_with_yaml_and_cli(parser, args_list=None, default_yaml=None):
    if args_list is not None:
        arguments = list(args_list)
    elif "ipykernel" in sys.modules:
        arguments = [str(default_yaml)] if default_yaml else []
    else:
        arguments = sys.argv[1:]
    if not arguments or not arguments[0].endswith((".yaml", ".yml")):
        return parser.parse_args_into_dataclasses(args=arguments)
    with open(arguments[0], encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    actions = {
        option: action for action in parser._actions for option in action.option_strings
    }
    required = {action: action.required for action in parser._actions}
    try:
        for action in parser._actions:
            action.required = False
        overrides = parser.parse_args(arguments[1:])
    finally:
        for action, value in required.items():
            action.required = value
    for argument in arguments[1:]:
        action = actions.get(argument.split("=", 1)[0])
        if action is not None:
            values[action.dest] = getattr(overrides, action.dest)
    return parser.parse_dict(values)


def load_dataframes(dataset_name, data_dir):
    paths = {
        split: Path(OUTPUT_PATH) / f"merged_{suffix}.csv"
        for split, suffix in (
            ("train", "train"),
            ("validation", "val"),
            ("test", "test"),
        )
    }
    if all(path.is_file() for path in paths.values()):
        frames = {split: pd.read_csv(path) for split, path in paths.items()}
        if all(
            "dataset" in frame and set(frame["dataset"].dropna()) == {dataset_name}
            for frame in frames.values()
            if len(frame)
        ):
            return frames
    from argparse import Namespace
    from scripts.prepare_dataset import main as prepare_dataset

    prepare_dataset(
        Namespace(
            config=str(PROJECT_ROOT / "configs" / "data" / f"{dataset_name}.yaml"),
            data_dir=data_dir,
        )
    )
    return {split: pd.read_csv(path) for split, path in paths.items()}


def load_model_and_tokenizer(checkpoint=None):
    if checkpoint:
        config = OpenViVQAConfig.from_pretrained(checkpoint)
        model = OpenViVQAModel.from_pretrained(checkpoint, config=config)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        processor_dir = Path(checkpoint) / "image_processor"
        if processor_dir.is_dir():
            model.image_processor = model.image_processor.from_pretrained(processor_dir)
            model.visual_search.vit_processor = model.image_processor
    else:
        config = OpenViVQAConfig()
        tokenizer = AutoTokenizer.from_pretrained(config.vit5_name)
        model = OpenViVQAModel(config)
    model.sync_tokenizer_ids(tokenizer)
    return model, tokenizer


def build_collator(model, tokenizer):
    ocr_config = {
        "ocr_embedding": {
            **DEFAULT_OCR_CONFIG["ocr_embedding"],
            "d_det": model.config.ocr_d_det,
            "d_rec": model.config.ocr_d_rec,
            "max_scene_text": model.config.ocr_max_scene_text,
        }
    }
    return ViT5VQADataCollator(
        tokenizer,
        model.image_processor,
        Vision_Encode_Ocr_Feature(ocr_config),
        model.config,
    )
