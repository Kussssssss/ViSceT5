"""QA-ViT loading, masking, integration schedules and checkpointed gradients."""

import copy
from contextlib import nullcontext

import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

from models.modules.qa_clip import MMCLIPEncoderLayer, QACLIPEncoder


@pytest.fixture
def vision_config():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    return CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        attention_dropout=0,
        attn_implementation="eager",
    )


@pytest.fixture
def inputs():
    return {
        "pixel_values": torch.randn(2, 3, 32, 32),
        "text_emb": torch.randn(2, 3, 24),
        "text_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "output_attentions": True,
    }


@pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
def test_backbone_loading_and_trained_adapter_roundtrip(
    vision_config, tmp_path, low_cpu_mem_usage, inputs
):
    backbone = CLIPVisionModel(vision_config)
    path = tmp_path / "backbone"
    backbone.save_pretrained(path)
    model, info = QACLIPEncoder.from_pretrained(
        path,
        instruction_dim=24,
        integration_point="early",
        freeze_clip=True,
        output_loading_info=True,
        local_files_only=True,
        attn_implementation="eager",
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    assert info["unexpected_keys"] == []
    assert all(parameter.device.type != "meta" for parameter in model.parameters())
    assert model.config.instruction_dim == 24
    assert model.config.integration_point == "early"
    for name, parameter in backbone.named_parameters():
        torch.testing.assert_close(
            parameter, dict(model.named_parameters())[name], rtol=0, atol=0
        )
        assert not dict(model.named_parameters())[name].requires_grad
    outputs = model(**inputs)
    assert torch.isfinite(outputs.last_hidden_state).all()
    (
        outputs.last_hidden_state * torch.randn_like(outputs.last_hidden_state)
    ).sum().backward()
    for layer in model.vision_model.encoder.layers[:2]:
        assert layer.instruct_dim_reduce[1].in_features == 24
        torch.testing.assert_close(
            layer.self_attn.instruction_out_proj.weight, layer.self_attn.out_proj.weight
        )
        torch.testing.assert_close(
            layer.self_attn.instruction_out_proj.bias, layer.self_attn.out_proj.bias
        )
        assert layer.self_attn.instruction_proj_gate.item() == 0
        assert layer.instruct_dim_reduce[1].weight.requires_grad
        assert torch.isfinite(layer.instruct_dim_reduce[1].weight.grad).all()
        assert layer.instruct_dim_reduce[1].weight.grad.norm() > 0
        assert layer.self_attn.instruction_proj_gate.grad.norm() > 0
        assert (
            torch.count_nonzero(layer.self_attn.instruction_out_proj.weight.grad) == 0
        )
        with torch.no_grad():
            layer.self_attn.instruction_out_proj.weight.add_(0.25)
            layer.self_attn.instruction_out_proj.bias.fill_(0.125)
            layer.self_attn.instruction_proj_gate.fill_(0.5)
    path = tmp_path / "trained"
    model.save_pretrained(path)
    restored = QACLIPEncoder.from_pretrained(
        path,
        local_files_only=True,
        attn_implementation="eager",
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter, dict(restored.named_parameters())[name], rtol=0, atol=0
        )
        assert (
            parameter.requires_grad
            == dict(restored.named_parameters())[name].requires_grad
        )

    # A partial checkpoint must retain an existing trained adapter weight even
    # when its bias needs initialization from the backbone.
    state = model.state_dict()
    missing_bias = "vision_model.encoder.layers.0.self_attn.instruction_out_proj.bias"
    del state[missing_bias]
    partial_path = tmp_path / "partial"
    model.save_pretrained(partial_path, state_dict=state)
    partial = QACLIPEncoder.from_pretrained(
        partial_path,
        local_files_only=True,
        attn_implementation="eager",
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    original_layer = model.vision_model.encoder.layers[0].self_attn
    restored_layer = partial.vision_model.encoder.layers[0].self_attn
    torch.testing.assert_close(
        restored_layer.instruction_out_proj.weight,
        original_layer.instruction_out_proj.weight,
    )
    torch.testing.assert_close(
        restored_layer.instruction_out_proj.bias, original_layer.out_proj.bias
    )


@pytest.mark.parametrize(
    "schedule,expected",
    [
        ("all", [0, 1, 2, 3]),
        ("early", [0, 1]),
        ("late", [2, 3]),
        ("late2", [3]),
        ("late3", [1, 2, 3]),
        ("sparse", [0, 2]),
    ],
)
def test_integration_schedules_forward(vision_config, inputs, schedule, expected):
    model = QACLIPEncoder(
        vision_config, instruction_dim=24, integration_point=schedule
    ).eval()
    assert [
        i
        for i, layer in enumerate(model.vision_model.encoder.layers)
        if isinstance(layer, MMCLIPEncoderLayer)
    ] == expected
    outputs = model(**inputs)
    assert outputs.last_hidden_state.shape == (2, 5, 32)
    for i in expected:
        assert outputs.attentions[i].shape == (2, 4, 5, 8)
    assert torch.isfinite(outputs.last_hidden_state).all()


def test_padding_is_invariant_and_attention_keeps_gradients(vision_config, inputs):
    model = QACLIPEncoder(vision_config, instruction_dim=24, freeze_clip=True).eval()
    question = inputs["text_emb"].requires_grad_()
    outputs = model(**inputs)
    changed = dict(inputs)
    changed["text_emb"] = question.detach().clone()
    changed["text_emb"][inputs["text_mask"] == 0] = (
        torch.randn_like(changed["text_emb"][inputs["text_mask"] == 0]) * 100
    )
    torch.testing.assert_close(
        outputs.last_hidden_state, model(**changed).last_hidden_state, rtol=0, atol=0
    )
    attention = outputs.attentions[-1]
    padding = ~inputs["text_mask"].bool()[:, None, None, :]
    assert torch.count_nonzero(attention[..., 5:].masked_select(padding)) == 0
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones_like(attention[..., 0])
    )
    (attention * torch.randn_like(attention)).sum().backward()
    assert question.grad[inputs["text_mask"].bool()].norm() > 0
    assert torch.count_nonzero(question.grad[inputs["text_mask"] == 0]) == 0
    with pytest.raises(ValueError, match="text_mask must have shape"):
        model(**{**inputs, "text_mask": inputs["text_mask"][:, :2]})


def test_visual_first_matches_legacy_outputs_and_gradients(vision_config):
    model = QACLIPEncoder(vision_config, instruction_dim=24, freeze_clip=True).eval()
    attention = model.vision_model.encoder.layers[-1].self_attn
    with torch.no_grad():
        attention.instruction_proj_gate.fill_(0.3)
    visual = torch.randn(2, 5, 32, requires_grad=True)
    question = torch.randn(2, 3, 32, requires_grad=True)
    question_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    output, weights = attention(
        visual, kv_states=question, kv_masks=question_mask, output_attentions=True
    )

    # Reference before this change: [question, visual], all queries projected,
    # then the question rows discarded. Test both column layout and gradients.
    joint = torch.cat((question, visual), dim=1)
    queries = attention._shape(attention.q_proj(joint) * attention.scale, 8, 2)
    keys = attention._shape(attention.k_proj(joint), 8, 2)
    values = attention._shape(attention.v_proj(joint), 8, 2)
    key_mask = torch.cat(
        (question_mask.bool(), torch.ones(2, 5, dtype=torch.bool)), dim=1
    )
    scores = torch.matmul(queries, keys.transpose(-1, -2))[:, :, 3:, :]
    scores = scores.masked_fill(
        ~key_mask[:, None, None, :], torch.finfo(scores.dtype).min
    )
    legacy_weights = scores.softmax(dim=-1)
    attended = torch.matmul(legacy_weights, values).transpose(1, 2).reshape(2, 5, 32)
    legacy_output = attention.out_proj(
        attended
    ) + attention.instruction_proj_gate.tanh() * attention.instruction_out_proj(
        attended
    )
    reordered_weights = torch.cat(
        (legacy_weights[..., 3:], legacy_weights[..., :3]), dim=-1
    )
    torch.testing.assert_close(output, legacy_output)
    torch.testing.assert_close(weights, reordered_weights)

    target = torch.randn_like(output)
    question_target = torch.randn_like(weights[..., 5:])
    variables = (visual, question) + tuple(
        p for p in attention.parameters() if p.requires_grad
    )
    gradients = torch.autograd.grad(
        (output * target).sum() + (weights[..., 5:] * question_target).sum(), variables
    )
    legacy_gradients = torch.autograd.grad(
        (legacy_output * target).sum()
        + (legacy_weights[..., :3] * question_target).sum(),
        variables,
    )
    for actual, expected in zip(gradients, legacy_gradients):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpointing_matches_outputs_and_gradients(vision_config, inputs, reentrant):
    baseline = QACLIPEncoder(
        vision_config, instruction_dim=24, freeze_clip=True
    ).train()
    checkpointed = copy.deepcopy(baseline)
    if reentrant:
        checkpointed.gradient_checkpointing_enable({"use_reentrant": True})
    else:
        checkpointed.gradient_checkpointing_enable()
    # Vision pixels stay detached; a frozen CLIP still needs instruction gradients.
    question = inputs["text_emb"].detach().clone().requires_grad_()
    other_question = question.detach().clone().requires_grad_()
    outputs = baseline(**{**inputs, "text_emb": question})
    with (
        pytest.warns(UserWarning, match="None of the inputs have requires_grad")
        if reentrant
        else nullcontext()
    ):
        other_outputs = checkpointed(**{**inputs, "text_emb": other_question})
    target = torch.randn_like(outputs.last_hidden_state)
    (outputs.last_hidden_state * target).sum().backward()
    (other_outputs.last_hidden_state * target).sum().backward()
    torch.testing.assert_close(
        outputs.last_hidden_state, other_outputs.last_hidden_state
    )
    torch.testing.assert_close(outputs.attentions[-1], other_outputs.attentions[-1])
    torch.testing.assert_close(question.grad, other_question.grad)
    for name, parameter in baseline.named_parameters():
        other = dict(checkpointed.named_parameters())[name]
        if parameter.requires_grad:
            assert parameter.grad is not None and other.grad is not None
            torch.testing.assert_close(parameter.grad, other.grad)
    checkpointed.gradient_checkpointing_disable()
    assert not checkpointed.vision_model.encoder.gradient_checkpointing


def test_autocast_attention_stays_fp32(vision_config, inputs):
    model = QACLIPEncoder(vision_config, instruction_dim=24, freeze_clip=True).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = model(**inputs)
        loss = outputs.last_hidden_state.square().mean()
    assert outputs.attentions[-1].dtype == torch.float32
    assert torch.isfinite(outputs.last_hidden_state).all()
    loss.backward()
    for layer in model.vision_model.encoder.layers[2:]:
        assert torch.isfinite(layer.instruct_dim_reduce[1].weight.grad).all()


def test_without_instruction_matches_clip(vision_config, tmp_path):
    backbone = CLIPVisionModel(vision_config).eval()
    backbone.save_pretrained(tmp_path)
    model = QACLIPEncoder.from_pretrained(
        tmp_path, instruction_dim=24, local_files_only=True, attn_implementation="eager"
    ).eval()
    pixels = torch.randn(2, 3, 32, 32)
    outputs = model(pixel_values=pixels, output_attentions=True)
    torch.testing.assert_close(
        outputs.last_hidden_state, backbone(pixel_values=pixels).last_hidden_state
    )
    assert outputs.attentions[-1].shape == (2, 4, 5, 5)
    assert isinstance(model(pixel_values=pixels, return_dict=False), tuple)
