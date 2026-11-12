"""Chunked MoE rounding boundaries and eager fallbacks on small CPU tensors."""

import pytest
import torch

from praxis.kernels.layers_triton import moe_chunked_combine
from praxis.layers import _routed_chunks, _sum_rows_of_tokens


def assert_bytes_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("top_k", [2, 8])
@pytest.mark.parametrize("scaling", [0.0, -1.25, 2.5, 1.00013])
@pytest.mark.parametrize("with_shared", [False, True])
def test_chunked_combine_keeps_fp32_products(dtype, top_k, scaling, with_shared):
    torch.manual_seed(19)
    rows = torch.randn(5 * top_k, 129, dtype=dtype)
    gates = torch.randn(5 * top_k, dtype=dtype)
    inverse = torch.randperm(5 * top_k)
    shared = torch.randn(5, 129, dtype=dtype) if with_shared else None
    expected = _sum_rows_of_tokens(
        rows, inverse.view(5, top_k), (gates.float() * scaling)[inverse].view(5, top_k)
    )
    if shared is not None:
        expected = expected.to(dtype) + shared
    actual = moe_chunked_combine(rows, gates, inverse, top_k, scaling, shared=shared)
    assert_bytes_equal(actual, expected)


def test_chunked_combine_output_slice_and_gradients():
    torch.manual_seed(21)
    rows = torch.randn(24, 128, requires_grad=True)
    gates = torch.randn(24, requires_grad=True)
    shared = torch.randn(3, 128, requires_grad=True)
    inverse = torch.randperm(24)
    scale = (gates * 2.5)[inverse].view(3, 8)
    expected = _sum_rows_of_tokens(rows, inverse.view(3, 8), scale) + shared
    storage = torch.full((5, 128), 123.0)
    actual = moe_chunked_combine(
        rows, gates, inverse, 8, 2.5, shared=shared, out=storage[1:4]
    )
    assert actual.data_ptr() == storage[1:4].data_ptr()
    assert_bytes_equal(actual, expected)
    assert torch.equal(storage[[0, 4]], torch.full((2, 128), 123.0))
    actual_grad = torch.autograd.grad(actual.square().sum(), (rows, gates, shared))
    expected_grad = torch.autograd.grad(expected.square().sum(), (rows, gates, shared))
    for left, right in zip(actual_grad, expected_grad, strict=True):
        assert_bytes_equal(left, right)


@pytest.mark.parametrize("tokens,step", [(0, 5), (1, 5), (23, 5), (23, 100)])
def test_chunked_shared_epilogue_keeps_chunk_boundaries(tokens, step):
    torch.manual_seed(23)
    x = torch.randn(tokens, 16)
    ids = torch.randint(0, 4, (tokens, 2))
    gates = torch.rand(tokens, 2)
    gate_up = torch.randn(4, 16, 16)
    down = torch.randn(4, 8, 16)
    shared = torch.randn_like(x)
    args = (x, ids, gates, gate_up, down, 2.5, step)
    expected = _routed_chunks(*args) + shared
    actual = _routed_chunks(*args, fuse=True, shared=shared)
    assert_bytes_equal(actual, expected)
