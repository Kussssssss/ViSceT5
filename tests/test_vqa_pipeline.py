"""Exercise the real multimodal modules with tiny, offline random backbones."""

import copy
import itertools
import json
import sys
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
)

from configs.arguments import CustomTrainingArguments, DataArguments, ModelArguments
from configs.model_config import OpenViVQAConfig
from data.dataset import ViT5VQADataset
from data.collator import _stable_hash
from data.ocr_correction import OCRCorrector
from models.modules.qa_clip import QACLIPEncoder
from models.openvivqa_model import OpenViVQAModel
from training.common import (
    PROJECT_ROOT,
    build_collator,
    load_model_and_tokenizer,
    parse_args_with_yaml_and_cli,
)
from training.metrics import compute_f1_em
from training.trainer import VQATrainer
from transformers.models.t5.modeling_t5 import T5LayerNorm
import transformers.modeling_utils as hf_modeling_utils


def test_memory_audit_uses_same_batch_and_real_optimizer(
    make_model, tokenizer, samples, tmp_path, monkeypatch
):
    from scripts import profile_training_memory as audit

    model = make_model(ablation_use_qaclip=True, ablation_use_ocr_aug=False)
    monkeypatch.setattr(
        audit, "load_model_and_tokenizer", lambda checkpoint, overrides=None: (model, tokenizer)
    )
    monkeypatch.setattr(
        audit, "load_dataframes", lambda *args: {"train": pd.DataFrame(samples)}
    )
    batch_path = tmp_path / "audit-batch.pt"
    output = tmp_path / "memory-audit"
    arguments = [
        "--output_dir",
        str(output),
        "--audit_batch_file",
        str(batch_path),
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "2",
        "--remove_unused_columns",
        "false",
        "--audit_steps",
        "3",
    ]
    audit.main(arguments)
    manifest = json.loads((output / "manifest.json").read_text())
    events = json.loads((output / "memory_trace.json").read_text())
    report = json.loads((output / "parameters_after_optimizer.json").read_text())
    assert batch_path.is_file()
    assert len([e for e in events if e["phase"] == "after_forward"]) == 6
    assert len([e for e in events if e["phase"] == "after_optimizer"]) == 3
    assert all(
        group["trainable"] == group["in_optimizer"]
        for group in report["groups"].values()
    )
    assert report["groups"]["vit5"]["with_gradient"] > 0
    assert report["groups"]["ocr_encoder"]["with_gradient"] > 0
    assert report["groups"]["qa_clip"]["with_gradient"] > 0
    assert "visual_search" not in report["groups"]
    missing = [
        n
        for n, p in report["parameters"].items()
        if p["requires_grad"] and not p["has_gradient"]
    ]
    assert not missing  # unused branches are not built at all
    shape = next(e for e in events if e["phase"] == "after_forward")["shapes"][
        "vit5.encoder"
    ]
    assert len(shape) == 2
    assert shape[1]["shape"] == [2, 16, 32]  # question 8 + patches 4 + OCR tokens 4

    def unexpected_data_load(*args):
        raise AssertionError("A saved audit batch must bypass the dataset/collator.")

    monkeypatch.setattr(audit, "load_dataframes", unexpected_data_load)
    audit.main(arguments)
    repeated = json.loads((output / "manifest.json").read_text())
    assert manifest["batch_sha256"] == repeated["batch_sha256"]


@pytest.fixture(autouse=True)
def offline_backbones(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    text_config = T5Config(
        vocab_size=32,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    vision_config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        instruction_dim=32,
        integration_point="late",
        freeze_clip=True,
    )
    monkeypatch.setattr(
        T5ForConditionalGeneration,
        "from_pretrained",
        lambda *a, **k: T5ForConditionalGeneration(text_config),
    )
    monkeypatch.setattr(
        QACLIPEncoder,
        "from_pretrained",
        lambda *a, **k: QACLIPEncoder(
            copy.deepcopy(vision_config), integration_point=k.get("integration_point")
        ),
    )


@pytest.fixture
def tokenizer():
    words = [
        "<pad>",
        "</s>",
        "<unk>",
        "biển",
        "hiệu",
        "ghi",
        "gì",
        "cà",
        "phê",
        "sữa",
        "123",
        "trà",
    ]
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )


TINY = dict(
    image_size=32,
    ocr_d_det=4,
    ocr_d_rec=4,
    ocr_max_scene_text=9,
    char_max_num=12,
    text_max_input_length=8,
    text_max_target_length=6,
    generation_max_new_tokens=3,
    generation_num_beams=1,
)


@pytest.fixture
def make_model():
    def build(**flags):
        torch.manual_seed(7)
        return OpenViVQAModel(OpenViVQAConfig(**{**TINY, **flags}))

    return build


@pytest.fixture
def model(make_model):
    # Original-OCR layout (no related copy) for the collator/loader tests.
    return make_model(ablation_use_ocr_aug=False)


@pytest.fixture
def corrector(samples):
    return OCRCorrector(train_frame=pd.DataFrame(samples), label_list=[0.9, 0.9])


@pytest.fixture
def samples(tmp_path):
    paths = []
    for index in range(2):
        image_path, ocr_path = (
            tmp_path / f"image-{index}.png",
            tmp_path / f"ocr-{index}.npy",
        )
        Image.new("RGB", (40, 32), color=(90 + index * 20, 120, 180)).save(image_path)
        features = {
            "texts": ["CÀ PHÊ", "cà phê", "!!!", "123", "SỮA"],
            "boxes": np.array(
                [
                    [0, 0, 16, 8],
                    [0, 0, 16, 8],
                    [20, 0, 32, 8],
                    [0, 16, 10, 24],
                    [20, 16, 36, 24],
                ],
                dtype=np.float32,
            ),
            "det_features": np.repeat(
                np.arange(5, dtype=np.float32)[:, None], 4, axis=1
            ),
            "rec_features": np.repeat(
                (np.arange(5, dtype=np.float32) + 10)[:, None], 4, axis=1
            ),
            "scores": np.ones(5, dtype=np.float32),
            "width": 40,
            "height": 32,
        }
        np.save(ocr_path, features)
        paths.append(
            {
                "image_path": str(image_path),
                "ocr_path": str(ocr_path),
                "question": "biển hiệu ghi gì",
                "answer": "cà phê",
            }
        )
    return paths


def test_collator_preserves_original_alignment(model, tokenizer, samples):
    collator = build_collator(model, tokenizer)
    batch = collator(samples)
    assert batch["ocr_info"][0]["texts"] == ["cà phê", "123", "sữa"]
    assert batch["ocr_info"][0]["det_features"][:3, 0].tolist() == [0, 3, 4]
    assert batch["ocr_info"][0]["rec_features"][:3, 0].tolist() == [10, 13, 14]
    assert batch["ocr_to_word_map"][0].tolist() == [0, 0, 1, 2]
    assert batch["ocr_word_mask"][0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert (batch["labels"] == -100).any()
    again = collator(samples)
    for key, value in batch.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, again[key])


def test_collator_unlabelled_and_missing_features(model, tokenizer, samples, tmp_path):
    collator = build_collator(model, tokenizer)
    unlabelled = [{**sample, "answer": "", "ocr_path": None} for sample in samples]
    batch = collator(unlabelled)
    assert "labels" not in batch
    assert batch["ocr_info"][0]["texts"] == ["<unk>"]
    assert batch["ocr_attention_mask"].sum().item() == 2
    with pytest.raises(ValueError, match="entirely"):
        collator([samples[0], unlabelled[1]])
    path = tmp_path / "text-only.npy"
    np.save(path, {"texts": ["cà phê"], "boxes": [[0, 0, 0.5, 0.5]]})
    result = collator([{**samples[0], "ocr_path": str(path)}])
    assert result["ocr_info"][0]["det_features"].count_nonzero().item() == 0


def test_ocr_limits_and_padding_mapping(model, tokenizer, samples):
    model.config.ocr_max_scene_text = 2
    collator = build_collator(model, tokenizer)
    batch = collator([samples[0], {**samples[1], "ocr_path": None}])
    assert batch["ocr_char_ids"].shape[1] == 2
    assert batch["ocr_info"][0]["texts"] == ["cà phê", "123"]
    assert batch["ocr_to_word_map"][1].tolist() == [0, -1, -1]
    assert batch["ocr_attention_mask"][1].tolist() == [1, 0, 0]


def test_palette_transparency_converts_without_warning(
    model, tokenizer, samples, tmp_path
):
    image = Image.new("P", (40, 32), color=1)
    image.putpalette([0, 0, 0, 80, 120, 180] + [0] * (256 * 3 - 6))
    image.info["transparency"] = bytes([0, 255] + [255] * 254)
    path = tmp_path / "palette.png"
    image.save(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = build_collator(model, tokenizer)(
            [{**samples[0], "image_path": str(path)}]
        )
    expected_image = Image.new("RGB", (40, 32), color=(80, 120, 180))
    expected_pixels = model.image_processor(
        images=[expected_image], return_tensors="pt"
    )["pixel_values"]
    torch.testing.assert_close(batch["pixel_values"], expected_pixels)
    assert "pil_images" not in batch
    assert not any("Palette images" in str(warning.message) for warning in caught)


def test_qaclip_load_initializes_new_instruction_weights(tmp_path, monkeypatch):
    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        instruction_dim=32,
    )
    backbone = CLIPVisionModel(config)
    checkpoint = tmp_path / "clip"
    backbone.save_pretrained(checkpoint)
    # Use the real loader, rather than the random-backbone fixture's replacement.
    monkeypatch.delattr(QACLIPEncoder, "from_pretrained")
    original_init = torch.nn.Linear.__init__

    def poison_skipped_linear(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hf_modeling_utils._init_weights:
            with torch.no_grad():
                self.weight.fill_(float("nan"))

    with patch.object(torch.nn.Linear, "__init__", poison_skipped_linear):
        loaded = QACLIPEncoder.from_pretrained(
            checkpoint,
            local_files_only=True,
            attn_implementation="eager",
        )
    loaded.config.freeze_clip = True
    loaded._apply_freeze()
    loaded.init_qavit_comps()
    assert all(torch.isfinite(parameter).all() for parameter in loaded.parameters())
    torch.testing.assert_close(
        loaded.vision_model.embeddings.patch_embedding.weight,
        backbone.vision_model.embeddings.patch_embedding.weight,
        rtol=0,
        atol=0,
    )
    loaded.eval()
    outputs = loaded(
        pixel_values=torch.randn(2, 3, 32, 32),
        text_emb=torch.randn(2, 3, 32),
        text_mask=torch.ones(2, 3, dtype=torch.long),
        output_attentions=True,
    )
    assert torch.isfinite(outputs.last_hidden_state).all()
    outputs.last_hidden_state.square().mean().backward()
    instruction_weights = [
        parameter
        for name, parameter in loaded.named_parameters()
        if "instruct_dim_reduce" in name
    ]
    assert all(
        parameter.requires_grad and parameter.grad is not None
        for parameter in instruction_weights
    )
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in instruction_weights
    )
    reloaded_checkpoint = tmp_path / "qaclip"
    loaded.save_pretrained(reloaded_checkpoint)
    restored = QACLIPEncoder.from_pretrained(
        reloaded_checkpoint,
        local_files_only=True,
        attn_implementation="eager",
    )
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(
            parameter, dict(restored.named_parameters())[name], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "image,qaclip,ocr,aug,twc",
    [
        (image, qaclip, ocr, aug, twc)
        for image, qaclip, ocr, aug, twc in itertools.product([False, True], repeat=5)
        if (image or not qaclip) and (aug or not twc)
    ],
)
def test_all_ablations_forward_backward_generate(
    make_model, tokenizer, samples, corrector, image, qaclip, ocr, aug, twc
):
    model = make_model(
        ablation_use_image=image,
        ablation_use_qaclip=qaclip,
        ablation_use_ocr=ocr,
        ablation_use_ocr_aug=aug,
        use_twc=twc,
        twc_objective="both",
    )
    batch = build_collator(model, tokenizer, corrector, training=True)(samples)
    outputs = model(**batch)
    assert outputs["logits"].shape == (2, 6, 32)
    assert torch.isfinite(outputs["loss"])
    loss = outputs["loss"]
    if twc:
        for name in ("twc", "twc_answer"):
            assert torch.isfinite(outputs[f"{name}_loss"]) and 0 <= outputs[f"{name}_acc"] <= 1
            loss = loss + outputs[f"{name}_loss"]
    else:
        assert "twc_loss" not in outputs and "twc_answer_loss" not in outputs
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    expected = batch["input_ids"].shape[1] + (4 if image else 0) + batch["ocr_input_ids"].shape[1]
    assert outputs["attention_mask"].shape == (2, expected)
    assert model.get_input_embeddings().weight.grad is not None
    # Every built parameter that is trainable receives a gradient (nothing dead).
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, dead
    model.eval()
    inputs = build_collator(model, tokenizer, corrector, training=False)(samples)
    inputs.pop("labels")
    with torch.no_grad():
        direct = model.generate(**inputs)
        encoded, mask, _ = model.encode(**inputs)
        cached = model.generate(encoder_outputs=encoded, attention_mask=mask)
    torch.testing.assert_close(direct, cached)
    assert cached.shape[0] == 2 and cached.shape[1] <= 4


@pytest.mark.parametrize("use_question", [False, True])
def test_image_encoder_only_returns_patches_without_attention_export(
    make_model, use_question
):
    from types import SimpleNamespace

    model = make_model(ablation_use_qaclip=use_question)
    hidden = torch.randn(2, 5, 32)  # CLS + four patches
    with patch.object(
        model.qa_clip,
        "forward",
        return_value=SimpleNamespace(last_hidden_state=hidden, attentions=None),
    ) as forward:
        tokens, mask = model._encode_image(
            torch.randn(2, 3, 32, 32),
            torch.randint(3, 12, (2, 3)),
            torch.ones(2, 3, dtype=torch.long),
        )
    torch.testing.assert_close(tokens, hidden[:, 1:])
    assert mask.shape == (2, 4)
    assert forward.call_args.kwargs["output_attentions"] is False
    question = forward.call_args.kwargs["text_emb"]
    assert (question is not None and question.shape[1] == 3) if use_question else question is None


def test_checkpoint_roundtrip(model, tokenizer, samples, tmp_path):
    model.eval()
    batch = build_collator(model, tokenizer)(samples)
    with torch.no_grad():
        expected = model(**batch)["logits"]
    model.save_pretrained(tmp_path, max_shard_size="50KB")
    tokenizer.save_pretrained(tmp_path)
    model.image_processor.save_pretrained(tmp_path / "image_processor")
    restored, restored_tokenizer = load_model_and_tokenizer(str(tmp_path))
    restored.eval()
    with torch.no_grad():
        actual = restored(**build_collator(restored, restored_tokenizer)(samples))[
            "logits"
        ]
    # Weights are identical (checked below); CPU kernels may differ in the last bits.
    torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-5)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    assert (
        restored.vit5.shared.weight.data_ptr()
        == restored.vit5.lm_head.weight.data_ptr()
    )


def test_legacy_checkpoint_drops_unused_branches_and_preserves_remaining_weights(
    model, tokenizer, samples, tmp_path
):
    model.eval()
    state = model.state_dict()
    legacy_keys = {
        "visual_search.cnn.embeddings.patch_embeddings.weight": torch.randn(4, 3, 4, 4),
        "visual_search.cnn.layernorm.weight": torch.randn(32),
        "image_projector.weight": torch.randn(32, 32),
        "image_projector.bias": torch.randn(32),
        "ocr_projector.weight": torch.randn(32, 32),
        "ocr_projector.bias": torch.randn(32),
    }
    model.save_pretrained(
        tmp_path, safe_serialization=False, state_dict={**state, **legacy_keys}
    )
    config_path = tmp_path / "config.json"
    legacy_config = json.loads(config_path.read_text())
    legacy_config.pop("image_size")
    legacy_config.update(
        vs_target_size=32,
        vs_backbone="unused-convnext",
        ablation_use_vs=True,
        use_visual_search=True,
    )
    config_path.write_text(json.dumps(legacy_config), encoding="utf-8")
    restored, info = OpenViVQAModel.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"]
    assert set(info["unexpected_keys"]) == set(legacy_keys)
    assert restored.config.image_size == 32
    assert not any(
        key.startswith("vs_") or key in ("ablation_use_vs", "use_visual_search")
        for key in restored.config.to_dict()
    )
    assert not hasattr(restored, "visual_search")
    for name, value in state.items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    batch = build_collator(restored, tokenizer)(samples)
    with torch.no_grad():
        torch.testing.assert_close(
            model(**batch)["logits"], restored(**batch)["logits"], rtol=0, atol=0
        )


def test_image_size_overrides_legacy_visual_search_size():
    config = OpenViVQAConfig(image_size=32, vs_target_size=224, ablation_use_vs=False)
    assert config.image_size == 32
    assert "vs_target_size" not in config.to_dict()
    assert "ablation_use_vs" not in config.to_dict()


def test_ocr_norms_use_fresh_weights_and_vit5_config_epsilon():
    backbone = T5ForConditionalGeneration.from_pretrained("offline")
    source = backbone.encoder.final_layer_norm
    backbone.config.layer_norm_epsilon = 2e-5
    source.variance_epsilon = 9e-4
    with torch.no_grad():
        source.weight.copy_(torch.linspace(0.5, 1.5, backbone.config.d_model))
    norms = []
    for use_ocr in (True, False):
        with patch.object(T5ForConditionalGeneration, "from_pretrained", return_value=backbone):
            model = OpenViVQAModel(
                OpenViVQAConfig(image_size=32, ocr_d_det=4, ocr_d_rec=4, ablation_use_ocr=use_ocr)
            )
        if use_ocr:
            norms += [model.ocr_char_layernorm] + [
                getattr(model.semantic_ocr_embedding, name)
                for name in ("layer_norm_bboxes", "layer_norm_det", "layer_norm_rec",
                             "layer_norm_tok", "layer_norm_char")
            ]
        else:
            norms.append(model.ocr_lite_text_ln)
    for norm in norms:
        assert isinstance(norm, T5LayerNorm) and not hasattr(norm, "bias")
        assert norm.variance_epsilon == backbone.config.layer_norm_epsilon
        torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
    assert len({norm.weight.data_ptr() for norm in [source, *norms]}) == 8


@pytest.mark.parametrize("image,qaclip,ocr,twc", [(True, False, True, False), (True, True, False, True),
                                                  (False, False, True, True)])
def test_unused_modules_are_not_built(make_model, image, qaclip, ocr, twc):
    model = make_model(ablation_use_image=image, ablation_use_qaclip=qaclip,
                       ablation_use_ocr=ocr, use_twc=twc)
    names = [name for name, _ in model.named_parameters()]
    assert hasattr(model, "qa_clip") == image
    assert any("instruct_dim_reduce" in n for n in names) == qaclip
    assert hasattr(model, "ocr_encoder") == ocr and hasattr(model, "semantic_ocr_embedding") == ocr
    assert hasattr(model, "ocr_lite_text_proj") == (not ocr)
    assert hasattr(model, "twc_answer_query") == twc  # default objective: answer
    assert not hasattr(model, "twc_logit_scale")


def test_shared_modules_get_the_same_init_across_configurations(make_model):
    full = make_model(ablation_use_qaclip=True, use_twc=True).state_dict()
    plain = make_model(ablation_use_image=False).state_dict()
    shared = [k for k in plain if k in full]
    assert any(k.startswith("ocr_encoder.") for k in shared)
    for key in shared:
        torch.testing.assert_close(plain[key], full[key], rtol=0, atol=0)


def test_trainer_train_evaluate_predict(make_model, tokenizer, samples, corrector, tmp_path):
    model = make_model(use_twc=True, twc_objective="both", twc_related_train="twa")
    dataset = ViT5VQADataset(pd.DataFrame(samples))
    trainer = VQATrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=build_collator(model, tokenizer, corrector, training=True),
        eval_data_collator=build_collator(model, tokenizer, corrector, training=False),
        train_dataset=dataset,
        eval_dataset=dataset,
        args=Seq2SeqTrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            remove_unused_columns=False,
            predict_with_generate=True,
            generation_max_length=3,
            generation_num_beams=1,
            report_to="none",
            save_strategy="no",
            logging_steps=1,
            disable_tqdm=True,
            dataloader_pin_memory=False,
            use_cpu=True,
        ),
    )
    result = trainer.train()
    assert np.isfinite(result.training_loss)
    logged = [entry for entry in trainer.state.log_history if "twc_loss" in entry]
    assert logged and np.isfinite(logged[0]["twc_loss"]) and np.isfinite(logged[0]["twc_answer_loss"])
    assert trainer.get_eval_dataloader().collate_fn.related_mode == "correct"
    assert trainer.get_train_dataloader().collate_fn.related_mode == "twa"
    with patch.object(model, "encode", wraps=model.encode) as encode:
        metrics = trainer.evaluate()
        assert encode.call_count == 2
    assert np.isfinite(metrics["eval_loss"])
    unlabelled = ViT5VQADataset(
        pd.DataFrame([{**sample, "answer": np.nan} for sample in samples])
    )
    predicted = trainer.predict(unlabelled)
    assert predicted.predictions.shape[0] == 2 and predicted.label_ids is None
    assert (tmp_path / "train_metrics.jsonl").is_file()


def test_config_yaml_cli_overrides():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, _, train_args = parse_args_with_yaml_and_cli(
        parser,
        [
            str(PROJECT_ROOT / "configs" / "finetune.yaml"),
            "--ablation_use_qaclip",
            "False",
            "--smoke_test=True",
            "--num_train_epochs",
            "2",
        ],
    )
    assert not model_args.ablation_use_qaclip and train_args.smoke_test
    assert train_args.num_train_epochs == 2


def test_answer_normalization_metrics():
    f1, em = compute_f1_em([" CÀ   PHÊ ", "sữa"], ["cà phê", "sữa trà"])
    assert em == 0.5 and f1 == pytest.approx((1 + 2 / 3) / 2)


def test_training_entrypoint_save_resume_and_evaluation(
    model, tokenizer, samples, tmp_path, monkeypatch
):
    from training import evaluate, finetune

    frames = {split: pd.DataFrame(samples) for split in ("train", "validation", "test")}
    monkeypatch.setattr(
        finetune,
        "load_dataframes",
        lambda *a: {name: frame.copy() for name, frame in frames.items()},
    )
    monkeypatch.setattr(evaluate, "load_dataframes", lambda *a: frames)
    monkeypatch.setattr(
        finetune,
        "load_model_and_tokenizer",
        lambda checkpoint, overrides=None: (
            load_model_and_tokenizer(checkpoint, overrides) if checkpoint else (model, tokenizer)
        ),
    )
    monkeypatch.setattr(finetune, "hub_target", lambda: (None, None))
    arguments = [
        str(PROJECT_ROOT / "configs" / "finetune.yaml"),
        "--output_dir",
        str(tmp_path),
        "--logging_dir",
        str(tmp_path / "logs"),
        "--smoke_test",
        "True",
        "--smoke_max_steps",
        "1",
        "--per_device_train_batch_size",
        "2",
        "--per_device_eval_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "1",
        "--generation_max_length",
        "3",
        "--generation_num_beams",
        "1",
        "--use_cpu",
        "True",
        "--disable_tqdm",
        "True",
        "--dataloader_pin_memory",
        "False",
        "--ocr_feature_norm",
        "rmsnorm",  # the fixture model; resume re-applies the architecture flags
        "--qavit_mode",
        "off",
    ]
    trainer = finetune.main(arguments)
    assert trainer.state.global_step == 1
    assert (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / "tokenizer.json").is_file()
    assert (tmp_path / "image_processor" / "preprocessor_config.json").is_file()
    checkpoint = tmp_path / "checkpoint-1"
    assert (checkpoint / "optimizer.pt").is_file()
    resumed = finetune.main(
        arguments
        + ["--smoke_max_steps", "2", "--resume_from_checkpoint", str(checkpoint)]
    )
    assert resumed.state.global_step == 2
    metrics = evaluate.main(
        ["--checkpoint", str(tmp_path), "--output_dir", str(tmp_path / "evaluate")]
    )
    assert "validation_em" in metrics and "validation_em_f1" in metrics
    from training.metrics import java_available

    assert ("validation_cider" in metrics) == java_available()
    assert all(np.isfinite(value) for value in metrics.values())


def test_pipeline_arguments_do_not_modify_process_argv(monkeypatch):
    from run_pipeline import training_arguments

    monkeypatch.setenv("MODEL_NAME_OR_PATH", "checkpoint-10")
    monkeypatch.setenv("MOCK_TEST", "true")
    monkeypatch.setenv("SMOKE_MAX_STEPS", "2")
    original_argv = sys.argv[:]
    arguments = training_arguments()
    assert sys.argv == original_argv
    assert arguments[arguments.index("--model_name_or_path") + 1] == "checkpoint-10"
    assert arguments[arguments.index("--smoke_max_steps") + 1] == "2"
    assert arguments[arguments.index("--smoke_test") + 1] == "True"


def _reference_twc(logits, o2r_blocks, r2o_blocks, groups, words):
    """Previous pretrain implementation (model TWC branch + create_batch_labels + loss)."""

    def create_batch_labels(result):
        result = result.clone()
        ignore = torch.diagonal(result) == -1
        result[ignore, :] = -1.0
        result[:, ignore] = -1.0
        return result

    o2r = torch.block_diag(*o2r_blocks)
    r2o = torch.block_diag(*r2o_blocks)
    g = groups.reshape(-1)
    m = o2r.size(0)
    samp = torch.arange(m) // words
    inherit = (g.unsqueeze(1) == g.unsqueeze(0)) & (g.unsqueeze(1) >= 0) & (samp.unsqueeze(1) != samp.unsqueeze(0))
    o2r = torch.where(inherit, torch.diagonal(o2r).clone().unsqueeze(0).expand(m, m), o2r)
    r2o = torch.where(inherit, torch.diagonal(r2o).clone().unsqueeze(0).expand(m, m), r2o)
    o2r_l, r2o_l = create_batch_labels(o2r), create_batch_labels(r2o)
    mask = o2r_l != -1
    loss_i = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], o2r_l[mask])
    loss_t = torch.nn.functional.binary_cross_entropy_with_logits(logits.t()[mask], r2o_l[mask])
    return (loss_i + loss_t) / 2


def test_twc_loss_matches_previous_pretrain_implementation(make_model, tokenizer, samples, corrector):
    # BCE, no memory, no text masking = the previous TWA-code implementation.
    model = make_model(use_twc=True, twc_loss="bce", twc_queue_size=0, twc_mask_same_text=False,
                       twc_objective="lexical", twc_related_train="twa")
    # Same image asked twice: same (image, text) groups, different per-sample augmentation.
    pair = [samples[0], {**samples[0], "question": "hiệu gì"}]
    batch = build_collator(model, tokenizer, corrector, training=True)(pair)
    # Same image + same text in both samples → cross-sample inheritance is exercised.
    assert (batch["twc_group_ids"][0][:3] == batch["twc_group_ids"][1][:3]).all()
    words = batch["twc_o2r"].shape[1]
    hidden = torch.randn(2, 40, 32)
    offset = 5
    token_mask = batch["ocr_attention_mask"]
    tokens = hidden[:, offset: offset + token_mask.shape[1]]
    loss, accuracy = model._twc(tokens, batch["ocr_to_word_map"], token_mask,
                                batch["twc_o2r"], batch["twc_r2o"], batch["twc_group_ids"])
    word_map = batch["ocr_to_word_map"]
    feats = torch.zeros(2, 2 * words, 32)
    counts = torch.zeros(2, 2 * words)
    for b in range(2):
        for t in range(word_map.shape[1]):
            if word_map[b, t] >= 0 and token_mask[b, t]:
                feats[b, word_map[b, t]] += tokens[b, t]
                counts[b, word_map[b, t]] += 1
    feats = feats / counts.clamp_min(1e-9).unsqueeze(-1)
    norm = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    orig, rel = norm(feats[:, :words]).reshape(-1, 32), norm(feats[:, words:]).reshape(-1, 32)
    logits = model.twc_logit_scale.exp() * orig @ rel.t()
    reference = _reference_twc(logits, batch["twc_o2r"], batch["twc_r2o"], batch["twc_group_ids"], words)
    torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
    assert 0 <= float(accuracy) <= 1


def test_twc_collator_layout_and_labels(make_model, tokenizer, samples, corrector):
    model = make_model(use_twc=True, twc_objective="lexical", twc_related_train="twa")
    batch = build_collator(model, tokenizer, corrector, training=True)(samples)
    words = batch["twc_o2r"].shape[1]
    real = len(batch["ocr_info"][0]["texts"])
    assert batch["ocr_char_ids"].shape[1] == 2 * words
    assert batch["ocr_info"][0]["boxes"].shape[0] == 2 * words
    mapping = batch["ocr_to_word_map"][0]
    assert set(mapping[mapping >= 0].tolist()) == set(range(real)) | set(range(words, words + real))
    torch.testing.assert_close(batch["ocr_info"][0]["boxes"][:words], batch["ocr_info"][0]["boxes"][words:])
    o2r = batch["twc_o2r"][0]
    assert (o2r[real:, :] == -1).all() and (o2r[:, real:] == -1).all()
    assert set(o2r[:real, :real].diagonal().tolist()) <= {1.0, 0.9}
    assert (batch["twc_group_ids"][0][real:] == -1).all()
    text_ids = batch["twc_text_ids"][0]
    assert text_ids.shape == (2 * words,)
    assert (text_ids[:real] >= 0).all() and (text_ids[real:words] == -1).all() and (text_ids[words + real:] == -1).all()
    assert text_ids[0] == _stable_hash(batch["ocr_info"][0]["texts"][0]) & 0x3FFFFFFF
    again = build_collator(model, tokenizer, corrector, training=True)(samples)
    torch.testing.assert_close(batch["twc_o2r"], again["twc_o2r"])  # per-sample fixed augmentation


def test_corrector_labels_follow_twa():
    corrector = OCRCorrector(train_frame=pd.DataFrame({"question": ["cà phê sữa"], "answer": ["trà"]}))
    words = ["cà", "phê", "cà", "123", "xyzw"]
    related, o2r, r2o = corrector.augment(words, 0.0, __import__("random").Random(0))
    assert related == words
    assert [o2r[i][i] for i in range(5)] == [1.0] * 5
    assert o2r[0][2] == o2r[2][0] == 1.0 and o2r[0][1] == 0.0
    related, o2r, r2o = corrector.augment(words, 1.0, __import__("random").Random(0))
    assert related[3] == "123" and o2r[3][3] == 1.0  # numbers are never edited
    assert related[0] != "cà" and o2r[0][0] == 0.9  # in-lexicon word gets a char edit
    assert o2r[0][2] == o2r[2][2] and o2r[2][0] == o2r[0][0]  # same text inherits
    assert corrector.correct(["cà", "phẽ"]) == ["cà", "phê"]  # tone change → lexicon word



def test_loading_a_checkpoint_keeps_clip_frozen(make_model, tokenizer, tmp_path):
    model = make_model(ablation_use_qaclip=True)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    restored, _ = load_model_and_tokenizer(str(tmp_path))
    trainable = lambda m: {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable(restored) == trainable(model)
    assert not any(p.requires_grad for n, p in restored.qa_clip.named_parameters()
                   if "instruct" not in n and "instruction" not in n)


def _twc_batch(model, tokenizer, samples, corrector):
    batch = build_collator(model, tokenizer, corrector, training=True)(samples)
    tokens = torch.randn(len(samples), batch["ocr_input_ids"].shape[1], 32, requires_grad=True)
    args = (tokens, batch["ocr_to_word_map"], batch["ocr_attention_mask"], batch["twc_o2r"],
            batch["twc_r2o"], batch["twc_group_ids"], batch["twc_text_ids"])
    return batch, tokens, args


def test_twc_text_collision():
    query = torch.tensor([[1, 2], [3, 3], [-1, -1]])
    keys = torch.tensor([[2, 9], [4, 5], [5, 3], [-1, -1]])
    hit = OpenViVQAModel._text_collision(query, keys)
    expected = torch.tensor([[True, False, False, False], [False, False, True, False],
                             [False, False, False, False]])
    assert torch.equal(hit, expected)


def test_twc_memory_adds_negatives_after_start(make_model, tokenizer, samples, corrector):
    model = make_model(use_twc=True, twc_queue_size=64, twc_queue_start=1, twc_objective="lexical")
    model.train()
    batch, tokens, args = _twc_batch(model, tokenizer, samples, corrector)
    real = int((batch["twc_o2r"].diagonal(dim1=1, dim2=2) != -1).sum())
    first, _ = model._twc(*args)            # forward 1: memory filled, not read
    memory = model._twc_memory_state
    assert memory["filled"] == min(64, real) and memory["forwards"] == 1
    second, accuracy = model._twc(*args)    # forward 2: memory read → more negatives
    assert memory["filled"] == min(64, 2 * real)
    assert torch.isfinite(second) and 0 <= float(accuracy) <= 1
    second.backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert all("_twc_memory" not in key for key in model.state_dict())


def test_twc_memory_negatives_change_loss(make_model, tokenizer, samples, corrector):
    model = make_model(use_twc=True, twc_queue_size=64, twc_queue_start=0, twc_objective="lexical")
    model.train()
    batch, tokens, args = _twc_batch(model, tokenizer, samples, corrector)
    plain, _ = model._twc(*args)            # memory empty on the first forward
    memory = model._twc_memory_state
    unrelated = 10**9 + torch.arange(2 * memory["filled"]).reshape(-1, 2)  # texts not in the batch
    memory["text"] = memory["text"].index_copy(0, torch.arange(memory["filled"]), unrelated)
    with_memory, _ = model._twc(*args)
    assert float(with_memory) > float(plain)  # extra (unmasked) negatives only add to the softmax


def test_twc_memory_unused_in_eval(make_model, tokenizer, samples, corrector):
    model = make_model(use_twc=True, twc_queue_size=64, twc_queue_start=0, twc_objective="lexical")
    model.eval()
    _, _, args = _twc_batch(model, tokenizer, samples, corrector)
    model._twc(*args)
    assert getattr(model, "_twc_memory_state", None) is None


def test_twc_memory_needs_softmax():
    with pytest.raises(ValueError):
        OpenViVQAConfig(use_twc=True, twc_loss="bce", twc_queue_size=16, twc_objective="lexical")


def test_answer_twc_collator_targets(make_model, tokenizer, samples, corrector):
    model = make_model(use_twc=True)  # default: answer objective, related = correction
    collator = build_collator(model, tokenizer, corrector, training=True)
    assert collator.related_mode == "correct" and collator.twc_answer and not collator.twc_lexical
    batch = collator(samples)
    assert "twc_o2r" not in batch
    labels, words = batch["twc_answer_labels"][0], batch["ocr_words"][0] if "ocr_words" in batch else None
    texts = batch["ocr_info"][0]["texts"]
    slots = batch["ocr_char_ids"].shape[1]
    half = slots // 2
    assert labels.shape == (slots,)
    for k, text in enumerate(texts):  # answer "cà phê": the entry "cà phê" is a positive, "sữa" is not
        assert labels[k] == float(set(text.split()) <= {"cà", "phê"})
    assert labels[len(texts):half].eq(-1).all()
    assert (labels == 1).any()
    assert (batch["twc_answer_text_ids"] >= -1).all()
    eval_batch = build_collator(model, tokenizer, corrector, training=False)(samples)
    assert "twc_answer_labels" not in eval_batch
    # Same related copy at train and at test.
    torch.testing.assert_close(batch["ocr_input_ids"], eval_batch["ocr_input_ids"])


def test_answer_twc_retrieves_answer_words(make_model):
    model = make_model(use_twc=True)
    torch.manual_seed(0)
    batch, slots, dim = 2, 4, 32
    hidden_q = torch.randn(batch, 3, dim)
    tokens = torch.randn(batch, slots, dim)
    word_map = torch.arange(slots).repeat(batch, 1)
    token_mask = torch.ones(batch, slots, dtype=torch.long)
    labels = torch.tensor([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0]])
    text_ids = torch.tensor([[7, 8, 9, -1], [7, 5, 6, 4]])
    answer_ids = torch.tensor([[7], [5]])
    loss, accuracy = model._twc_answer(hidden_q, torch.ones(batch, 3), tokens, word_map, token_mask,
                                       labels, text_ids, answer_ids)
    assert torch.isfinite(loss) and 0 <= float(accuracy) <= 1
    # Sample 1's slot 0 has sample 0's answer text (7): ignored for sample 0, not a negative.
    plain, _ = model._twc_answer(hidden_q, torch.ones(batch, 3), tokens, word_map, token_mask,
                                 labels, text_ids.masked_fill(text_ids == 7, 99), answer_ids)
    assert not torch.allclose(loss, plain)
    # Aligning the question with its answer word lowers the loss.
    aligned = tokens.clone()
    aligned[0, 0] = 10 * hidden_q[0].mean(0)
    aligned[1, 1] = 10 * hidden_q[1].mean(0)
    better, acc = model._twc_answer(hidden_q, torch.ones(batch, 3), aligned, word_map, token_mask,
                                    labels, text_ids, answer_ids)
    assert float(better) < float(loss) and float(acc) == 1.0


def test_lexical_twc_labels_from_correction(make_model, tokenizer, samples, corrector):
    model = make_model(use_twc=True, twc_objective="lexical")  # related = correction (default)
    collator = build_collator(model, tokenizer, corrector, training=True)
    assert collator.related_mode == "correct" and collator.twc_lexical
    batch = collator(samples)
    real = len(batch["ocr_info"][0]["texts"])
    assert set(batch["twc_o2r"][0][:real, :real].diagonal().tolist()) <= {1.0, 0.9}


def test_ocr_feature_norm_switch(make_model):
    layer = make_model(ocr_feature_norm="layernorm").semantic_ocr_embedding
    rms = make_model().semantic_ocr_embedding  # configs without the field = rmsnorm (refactor checkpoints)
    assert isinstance(layer.layer_norm_det, torch.nn.LayerNorm) and layer.layer_norm_det.bias is not None
    assert isinstance(rms.layer_norm_det, T5LayerNorm)
    with pytest.raises(ValueError):
        OpenViVQAConfig(ocr_feature_norm="batchnorm")


def test_clip_late_adapter_trains_without_question(make_model, tokenizer, samples, corrector):
    model = make_model(clip_late_adapter=True, ablation_use_qaclip=False, ablation_use_ocr_aug=False)
    gates = [p for n, p in model.named_parameters() if n.endswith("instruction_proj_gate")]
    assert gates and all(g.requires_grad for g in gates)
    frozen = [n for n, p in model.qa_clip.named_parameters()
              if not p.requires_grad and "instruction" not in n and "instruct_dim_reduce" not in n]
    assert frozen  # CLIP itself stays frozen
    batch = build_collator(model, tokenizer, corrector, training=True)(samples)
    model(**batch)["loss"].backward()
    assert any(g.grad is not None and g.grad.abs().sum() > 0 for g in gates)
    plain = make_model(ablation_use_qaclip=False, ablation_use_ocr_aug=False)
    assert not any(n.endswith("instruction_proj_gate") for n, _ in plain.named_parameters())


def test_t1_recipe_from_yaml(make_model, tokenizer, samples, corrector):
    import yaml
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "finetune.yaml", encoding="utf-8"))
    keys = ("twc_loss", "twc_feature", "twc_queue_size", "twc_mask_same_text", "twc_objective",
            "twc_related_train", "ocr_feature_norm")
    assert {k: cfg[k] for k in keys} == {
        "twc_loss": "bce", "twc_feature": "fused", "twc_queue_size": 0, "twc_mask_same_text": False,
        "twc_objective": "lexical", "twc_related_train": "twa", "ocr_feature_norm": "layernorm",
    }
    assert cfg["qavit_mode"] == "off"
    assert (cfg["per_device_train_batch_size"], cfg["gradient_accumulation_steps"], cfg["label_smoothing_factor"]) == (8, 1, 0.1)
    model = make_model(use_twc=True, **{k: cfg[k] for k in keys})
    collator = build_collator(model, tokenizer, corrector, training=True)
    assert collator.related_mode == "twa" and collator.twc_lexical and not collator.twc_answer
    outputs = model(**collator(samples))
    assert torch.isfinite(outputs["twc_loss"]) and "twc_answer_loss" not in outputs
    assert not hasattr(model, "twc_answer_query")


def test_temperature_init_and_lr_groups(make_model, tokenizer, samples, corrector, tmp_path):
    import math
    model = make_model(use_twc=True, twc_objective="lexical", twc_loss="bce", twc_feature="fused",
                       twc_queue_size=0, twc_related_train="twa", twc_temperature_init=0.4,
                       twc_adv_probability=0.7, clip_late_adapter=True, ablation_use_qaclip=False)
    assert math.isclose(float(model.twc_logit_scale.exp()), 2.5, rel_tol=1e-5)
    args = CustomTrainingArguments(output_dir=str(tmp_path), learning_rate=3e-5, weight_decay=0.1,
                                   new_module_learning_rate=1e-4, scalar_learning_rate=1e-3,
                                   max_steps=1, per_device_train_batch_size=2, logging_steps=1,
                                   remove_unused_columns=False, report_to="none", save_strategy="no",
                                   use_cpu=True, dataloader_pin_memory=False, disable_tqdm=True)
    dataset = ViT5VQADataset(pd.DataFrame(samples))
    trainer = VQATrainer(model=model, args=args, tokenizer=tokenizer, train_dataset=dataset,
                         data_collator=build_collator(model, tokenizer, corrector, training=True))
    optimizer = trainer.create_optimizer()
    by_id = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    named = dict(model.named_parameters())
    assert by_id[id(named["twc_logit_scale"])]["lr"] == 1e-3
    assert by_id[id(named["twc_logit_scale"])]["weight_decay"] == 0.0
    gate = next(n for n in named if n.endswith("instruction_proj_gate"))
    assert by_id[id(named[gate])]["lr"] == 1e-3
    consformer = next(n for n in named if n.startswith("ocr_encoder.") and n.endswith("weight"))
    assert by_id[id(named[consformer])]["lr"] == 1e-4
    t5 = "vit5.shared.weight"
    assert by_id[id(named[t5])]["lr"] == 3e-5 and by_id[id(named[t5])]["weight_decay"] == 0.1
    assert sum(len(g["params"]) for g in optimizer.param_groups) == sum(p.requires_grad for p in model.parameters())
    trainer.train()
    taus = [e["twc_tau"] for e in trainer.state.log_history if "twc_tau" in e]
    assert taus and 0.3 < taus[0] < 0.5
    batch = build_collator(model, tokenizer, corrector, training=True)
    assert batch.cfg.twc_adv_probability == 0.7


@pytest.mark.parametrize("feature", ["fused", "lexical"])
def test_twc_separate_view(make_model, tokenizer, samples, feature):
    # "sữa" is a dictionary word here, so at p=1.0 it always gets character noise.
    corrector = OCRCorrector(train_frame=pd.DataFrame({"question": ["sữa cà phê"], "answer": ["sữa"]}),
                             label_list=[0.9, 0.9])
    model = make_model(use_twc=True, twc_objective="lexical", twc_loss="bce", twc_feature=feature,
                       twc_queue_size=0, twc_related_train="twa", twc_view="separate", twc_adv_probability=1.0)
    collator = build_collator(model, tokenizer, corrector, training=True)
    assert collator.related_mode == "twa" and collator.twc_separate
    batch = collator(samples)
    eval_batch = build_collator(model, tokenizer, corrector, training=False)(samples)
    # The decoder input is exactly the test-time input (deterministic correction) ...
    for key in ("ocr_input_ids", "ocr_to_word_map", "ocr_char_ids"):
        torch.testing.assert_close(batch[key], eval_batch[key])
    # ... while TWC gets its own augmented copy (100% here: it differs from the correction).
    assert "twc_ocr_input_ids" in batch and not torch.equal(batch["twc_ocr_char_ids"], batch["ocr_char_ids"])
    outputs = model(**batch)
    assert torch.isfinite(outputs["loss"]) and torch.isfinite(outputs["twc_loss"])
    (outputs["loss"] + outputs["twc_loss"]).backward()
    assert model.twc_logit_scale.grad is not None
    model.eval()
    with torch.no_grad():
        model.generate(**{k: v for k, v in batch.items() if k != "labels"}, num_beams=1, max_new_tokens=2)


def test_twc_view_validation():
    with pytest.raises(ValueError):
        OpenViVQAConfig(use_twc=True, twc_objective="answer", twc_view="separate")


def test_unset_group_rates_give_stock_optimizer(make_model, tokenizer, samples, corrector, tmp_path):
    # Both rates unset → same groups as the stock Trainer: one LR, HF weight-decay rule (gates decayed).
    model = make_model(clip_late_adapter=True, ablation_use_ocr_aug=False, ocr_feature_norm="layernorm")
    args = CustomTrainingArguments(output_dir=str(tmp_path), learning_rate=3e-5, weight_decay=0.1,
                                   report_to="none", use_cpu=True, remove_unused_columns=False)
    trainer = VQATrainer(model=model, args=args, tokenizer=tokenizer,
                         train_dataset=ViT5VQADataset(pd.DataFrame(samples)),
                         data_collator=build_collator(model, tokenizer, corrector, training=True))
    optimizer = trainer.create_optimizer()
    assert {g["lr"] for g in optimizer.param_groups} == {3e-5}
    named = dict(model.named_parameters())
    gate = next(n for n in named if n.endswith("instruction_proj_gate"))
    by_id = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    assert by_id[id(named[gate])]["weight_decay"] == 0.1

@pytest.mark.parametrize("env,expected", [
    ({"QAVIT_MODE": "no_question"}, "no_question"),
    ({"ABLATION_USE_QACLIP": "true"}, "question"),
    ({"ABLATION_USE_QACLIP": "false", "CLIP_LATE_ADAPTER": "true"}, "no_question"),
    ({"ABLATION_USE_QACLIP": "false"}, "off"),
])
def test_qavit_mode_from_env(monkeypatch, env, expected):
    import run_pipeline
    for key in ("QAVIT_MODE", "ABLATION_USE_QACLIP", "CLIP_LATE_ADAPTER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    arguments = run_pipeline.training_arguments()
    assert arguments[len(arguments) - 1 - arguments[::-1].index("--qavit_mode") + 1] == expected
