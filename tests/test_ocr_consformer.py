"""Regression checks against the GroupAttention source supplied by the user."""

import math

import pytest
import torch
from torch import nn

from models.modules.ocr_consformer import (
    GroupAttention,
    OCREncoder,
    build_neighbor_mask,
)


def original_group_attention(module, context, key_mask, prior):
    """Original equations, including both triangular masks and the epsilon terms."""
    batch, length = context.shape[:2]
    context = (
        module.norm(context).view(batch, length, module.h, module.d_k).transpose(1, 2)
    )
    a = torch.diag(torch.ones(length - 1, device=context.device), 1).long()
    b = torch.eye(length, device=context.device).long()
    c = a.t()
    mask = torch.logical_and(key_mask[:, None, None, :], a + c)
    key, query = module.linear_key(context), module.linear_query(context)
    scores = torch.matmul(query, key.transpose(-2, -1)) / module.d_k
    neighbor = torch.softmax(scores.masked_fill(mask == 0, -1e4), dim=-1)
    neighbor = torch.sqrt(neighbor * neighbor.transpose(-2, -1) + 1e-4)
    neighbor = prior + (1 - prior) * neighbor
    tri = torch.triu(torch.ones(length, length, device=context.device))[None, None]
    t = torch.log(neighbor + 1e-9).masked_fill(a == 0, 0).matmul(tri)
    group = tri.matmul(t).exp().masked_fill((tri.int() - b) == 0, 0)
    group = group + group.transpose(-2, -1) + neighbor.masked_fill(b == 0, 1e-4)
    return group, neighbor


@pytest.mark.parametrize("length", [1, 2, 5, 9])
@pytest.mark.parametrize("with_prior", [False, True])
def test_group_attention_matches_original_outputs_and_gradients(length, with_prior):
    torch.manual_seed(19)
    module = GroupAttention(3, 12)
    context = torch.randn(2, length, 12, requires_grad=True)
    reference_context = context.detach().clone().requires_grad_()
    mask = torch.ones(2, length, dtype=torch.long)
    prior = torch.rand(2, 3, length, length) * 0.5 if with_prior else 0.0
    actual = module(context, build_neighbor_mask(mask), prior)
    expected = original_group_attention(module, reference_context, mask, prior)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-6, rtol=1e-5)
    parameters = tuple(module.parameters())
    actual_gradients = torch.autograd.grad(
        sum(t.square().mean() for t in actual), (context, *parameters)
    )
    reference_gradients = torch.autograd.grad(
        sum(t.square().mean() for t in expected), (reference_context, *parameters)
    )
    for actual_grad, reference_grad in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(actual_grad, reference_grad, atol=1e-6, rtol=1e-5)


def test_group_probability_is_product_of_adjacent_links():
    torch.manual_seed(23)
    module = GroupAttention(2, 8)
    context = torch.randn(1, 6, 8)
    mask = build_neighbor_mask(torch.ones(1, 6))
    group, neighbor = module(context, mask, 0.0)
    for i in range(6):
        torch.testing.assert_close(group[..., i, i], neighbor[..., i, i])
        for j in range(i + 1, 6):
            product = torch.stack(
                [neighbor[..., k, k + 1] for k in range(i, j)], dim=-1
            ).prod(-1)
            torch.testing.assert_close(group[..., i, j], product + 1e-4)
    torch.testing.assert_close(group, group.transpose(-2, -1))
    # Prior values outside adjacent links / diagonal must not enter the path product.
    positions = torch.arange(6)
    far = (positions[:, None] - positions[None, :]).abs() > 1
    changed_prior = far.float()[None, None].expand(1, 2, 6, 6) * 0.7
    changed_group, _ = module(context, mask, changed_prior)
    torch.testing.assert_close(changed_group, group)


def test_three_encoder_layers_match_original_group_equations():
    torch.manual_seed(29)
    encoder = OCREncoder(2, 8, 4, 16, nn.Embedding(20, 8)).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    actual, breaks = encoder(ids, mask)
    expected = encoder._legacy_embed(ids)
    prior, reference_breaks = 0.0, []
    for layer in encoder.layers:
        prior, neighbor = original_group_attention(
            layer.group_attn, expected, mask, prior
        )
        reference_breaks.append(neighbor)
        expected = layer.sublayer[0](
            expected, lambda x: layer.self_attn(x, x, x, prior, mask)
        )
        expected = layer.sublayer[1](expected, layer.feed_forward)
    expected = encoder.norm(expected)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        breaks, torch.stack(reference_breaks, dim=1), atol=1e-6, rtol=1e-5
    )
    # Both the final state and returned per-layer probabilities remain differentiable.
    (actual.square().mean() + breaks.square().mean()).backward()
    for layer in encoder.layers:
        assert layer.group_attn.linear_query.weight.grad is not None
        assert torch.isfinite(layer.group_attn.linear_query.weight.grad).all()


def test_padding_does_not_change_valid_encoder_tokens():
    torch.manual_seed(31)
    encoder = OCREncoder(2, 8, 4, 16, nn.Embedding(30, 8)).eval()
    ids = torch.tensor([[1, 2, 3]])
    expected, expected_breaks = encoder(ids, torch.ones_like(ids))
    padded = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]])
    actual, breaks = encoder(padded, mask)
    torch.testing.assert_close(actual[:, :3], expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        breaks[..., :3, :3], expected_breaks, atol=1e-6, rtol=1e-5
    )
    # Keep the working key mask; the original snippet discarded masked_fill's result.
    attention = encoder.layers[0].self_attn
    tokens = torch.randn(1, 7, 8)
    changed = tokens.clone()
    changed[:, 3:] += 100
    original_output = attention(tokens[:, :3], tokens, tokens, 1.0, mask)
    changed_output = attention(tokens[:, :3], changed, changed, 1.0, mask)
    torch.testing.assert_close(original_output, changed_output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 7])
def test_group_attention_is_finite_with_padding_and_low_precision(dtype, length):
    module = GroupAttention(2, 8).to(dtype)
    context = torch.randn(2, length, 8, dtype=dtype, requires_grad=True)
    mask = torch.ones(2, length, dtype=torch.long)
    mask[1] = 0
    group, neighbor = module(context, build_neighbor_mask(mask), 0.0)
    assert group.dtype == neighbor.dtype == dtype
    assert torch.isfinite(group).all() and torch.isfinite(neighbor).all()
    (group.float().square().mean() + neighbor.float().square().mean()).backward()
    assert torch.isfinite(context.grad).all()


def test_regular_self_attention_keeps_sqrt_dkv_scaling():
    torch.manual_seed(37)
    encoder = OCREncoder(2, 8, 4, 16, nn.Embedding(20, 8)).eval()
    attention = encoder.layers[0].self_attn
    x = torch.randn(1, 3, 8)
    q = attention.fc_q(x).view(1, 3, 2, 4).transpose(1, 2)
    k = attention.fc_k(x).view(1, 3, 2, 4).permute(0, 2, 3, 1)
    v = attention.fc_v(x).view(1, 3, 2, 4).transpose(1, 2)
    expected = ((q @ k / math.sqrt(4)).softmax(-1) @ v).transpose(1, 2).reshape(1, 3, 8)
    torch.testing.assert_close(attention(x, x, x, 1.0, torch.ones(1, 3)), expected)
