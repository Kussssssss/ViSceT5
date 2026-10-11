"""Exercise the real multimodal modules with tiny, offline random backbones."""

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
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    ConvNextV2Config,
    ConvNextV2Model,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
)

from configs.arguments import CustomTrainingArguments, DataArguments, ModelArguments
from configs.model_config import OpenViVQAConfig
from data.dataset import ViT5VQADataset
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


def test_memory_audit_uses_same_batch_and_real_optimizer(model, tokenizer, samples, tmp_path, monkeypatch):
    from scripts import profile_training_memory as audit

    monkeypatch.setattr(audit, "load_model_and_tokenizer", lambda checkpoint: (model, tokenizer))
    monkeypatch.setattr(audit, "load_dataframes", lambda *args: {"train": pd.DataFrame(samples)})
    batch_path = tmp_path / "audit-batch.pt"
    output = tmp_path / "memory-audit"
    arguments = [
        "--output_dir", str(output),
        "--audit_batch_file", str(batch_path),
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "2",
        "--remove_unused_columns", "false",
        "--audit_steps", "3",
    ]
    audit.main(arguments)
    manifest = json.loads((output / "manifest.json").read_text())
    events = json.loads((output / "memory_trace.json").read_text())
    report = json.loads((output / "parameters_after_optimizer.json").read_text())
    assert batch_path.is_file()
    assert len([e for e in events if e["phase"] == "after_forward"]) == 6
    assert len([e for e in events if e["phase"] == "after_optimizer"]) == 3
    assert all(group["trainable"] == group["in_optimizer"] for group in report["groups"].values())
    assert report["groups"]["vit5"]["with_gradient"] > 0
    assert report["groups"]["ocr_encoder"]["with_gradient"] > 0
    assert report["groups"]["qa_clip"]["with_gradient"] > 0
    assert report["groups"]["visual_search"]["optimizer_state_bytes"] > 0
    missing = [n for n, p in report["parameters"].items() if p["requires_grad"] and not p["has_gradient"]]
    assert all(n.startswith("ocr_lite_") or n.startswith("visual_search.cnn.layernorm.") for n in missing)
    shape = next(e for e in events if e["phase"] == "after_forward")["shapes"]["vit5.encoder"]
    assert len(shape) == 2
    assert shape[1]["shape"] == [2, 18, 32]

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
    cnn_config = ConvNextV2Config(hidden_sizes=[4, 8, 16, 32], depths=[1, 1, 1, 1])
    monkeypatch.setattr(
        T5ForConditionalGeneration,
        "from_pretrained",
        lambda *a, **k: T5ForConditionalGeneration(text_config),
    )
    monkeypatch.setattr(
        QACLIPEncoder, "from_pretrained", lambda *a, **k: QACLIPEncoder(vision_config)
    )
    monkeypatch.setattr(
        ConvNextV2Model, "from_pretrained", lambda *a, **k: ConvNextV2Model(cnn_config)
    )
    monkeypatch.setattr(
        "models.modules.visual_search.AutoImageProcessor.from_pretrained",
        lambda *a, **k: CLIPImageProcessor(size=32, crop_size=32),
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


@pytest.fixture
def model():
    return OpenViVQAModel(
        OpenViVQAConfig(
            vs_target_size=32,
            ocr_d_det=4,
            ocr_d_rec=4,
            ocr_max_scene_text=9,
            char_max_num=12,
            text_max_input_length=8,
            text_max_target_length=6,
            generation_max_new_tokens=3,
            generation_num_beams=1,
        )
    )


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
    assert batch["pil_images"][0].mode == "RGB"
    assert batch["pil_images"][0].getpixel((0, 0)) == (80, 120, 180)
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
    "qaclip,vs,ocr", list(itertools.product([False, True], repeat=3))
)
def test_all_ablations_forward_backward_generate(
    model, tokenizer, samples, qaclip, vs, ocr
):
    model.config.ablation_use_qaclip = qaclip
    model.config.ablation_use_vs = vs
    model.config.ablation_use_ocr = ocr
    batch = build_collator(model, tokenizer)(samples)
    outputs = model(**batch, return_visual_search_debug=True)
    assert outputs["logits"].shape == (2, 6, 32)
    assert torch.isfinite(outputs["loss"])
    assert (
        outputs["attention_mask"].shape
        == outputs["encoder_outputs"].last_hidden_state.shape[:2]
    )
    outputs["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert model.visual_search.cnn.embeddings.patch_embeddings.weight.grad is not None
    assert model.get_input_embeddings().weight.grad is not None
    model.eval()
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    with torch.no_grad():
        direct = model.generate(**inputs)
        encoded, mask, _ = model.encode(**inputs)
        cached = model.generate(encoder_outputs=encoded, attention_mask=mask)
    torch.testing.assert_close(direct, cached)
    assert cached.shape[0] == 2 and cached.shape[1] <= 4
    if vs:
        assert "boxes" in outputs["vs_debug"]


@pytest.mark.parametrize("use_question", [False, True])
def test_visual_search_reads_correct_attention_columns(model, use_question):
    from types import SimpleNamespace

    model.config.ablation_use_qaclip = use_question
    hidden = torch.randn(2, 5, 32)  # CLS + four patches
    attention = torch.zeros(2, 4, 5, 8 if use_question else 5)
    expected_scores = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    if use_question:
        attention[:, :, 1:, :5] = 0.8  # Decoy visual keys
        attention[:, :, 1:, 5:] = expected_scores[:, None, :, None]
    else:
        attention[:, :, 0, 1:5] = expected_scores[:, None, :]
    with patch.object(
        model.qa_clip,
        "forward",
        return_value=SimpleNamespace(last_hidden_state=hidden, attentions=(attention,)),
    ):
        tokens, mask, scores = model._encode_image(
            torch.randn(2, 3, 32, 32),
            torch.randn(2, 3, 32),
            torch.ones(2, 3, dtype=torch.long),
        )
    torch.testing.assert_close(tokens, hidden[:, 1:])
    torch.testing.assert_close(scores, expected_scores)
    assert mask.shape == (2, 4)


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
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert (
        restored.vit5.shared.weight.data_ptr()
        == restored.vit5.lm_head.weight.data_ptr()
    )


def test_ocr_norms_use_fresh_weights_and_vit5_config_epsilon():
    backbone = T5ForConditionalGeneration.from_pretrained("offline")
    source = backbone.encoder.final_layer_norm
    backbone.config.layer_norm_epsilon = 2e-5
    source.variance_epsilon = 9e-4
    with torch.no_grad():
        source.weight.copy_(torch.linspace(0.5, 1.5, backbone.config.d_model))
    with patch.object(
        T5ForConditionalGeneration, "from_pretrained", return_value=backbone
    ):
        model = OpenViVQAModel(
            OpenViVQAConfig(vs_target_size=32, ocr_d_det=4, ocr_d_rec=4)
        )
    norms = [model.ocr_lite_text_ln, model.ocr_char_layernorm] + [
        getattr(model.semantic_ocr_embedding, name)
        for name in (
            "layer_norm_bboxes",
            "layer_norm_det",
            "layer_norm_rec",
            "layer_norm_tok",
            "layer_norm_char",
        )
    ]
    for norm in norms:
        assert isinstance(norm, T5LayerNorm) and not hasattr(norm, "bias")
        assert norm.variance_epsilon == backbone.config.layer_norm_epsilon
        torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
        assert not torch.equal(norm.weight, source.weight)
    assert len({norm.weight.data_ptr() for norm in [source, *norms]}) == 8
    with torch.no_grad():
        norms[0].weight.add_(1)
    torch.testing.assert_close(norms[1].weight, torch.ones_like(norms[1].weight))
    torch.testing.assert_close(
        source.weight, torch.linspace(0.5, 1.5, backbone.config.d_model)
    )


def test_inactive_ocr_branch_has_parameters_but_no_gradients(model, tokenizer, samples):
    lite_names = (
        "ocr_lite_text_ln",
        "ocr_lite_text_proj",
        "ocr_lite_text_ff",
        "ocr_lite_box_proj",
        "ocr_lite_box_ff",
    )
    lite_parameters = [
        parameter
        for name in lite_names
        for parameter in getattr(model, name).parameters()
    ]
    assert (
        sum(parameter.numel() for parameter in lite_parameters)
        == 5 * model.d_model**2 + 11 * model.d_model
    )
    assert all(parameter.requires_grad for parameter in lite_parameters)
    model(**build_collator(model, tokenizer)(samples))["loss"].backward()
    assert all(parameter.grad is None for parameter in lite_parameters)
    assert any(
        parameter.grad is not None
        for parameter in model.semantic_ocr_embedding.parameters()
    )


def test_trainer_train_evaluate_predict(model, tokenizer, samples, tmp_path):
    dataset = ViT5VQADataset(pd.DataFrame(samples))
    trainer = VQATrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=build_collator(model, tokenizer),
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
            "--ablation_use_vs",
            "False",
            "--smoke_test=True",
            "--num_train_epochs",
            "2",
        ],
    )
    assert not model_args.ablation_use_vs and train_args.smoke_test
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
        lambda checkpoint: (
            load_model_and_tokenizer(checkpoint) if checkpoint else (model, tokenizer)
        ),
    )
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
    assert "validation_em" in metrics and "validation_cider" in metrics
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
