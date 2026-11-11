"""Exact dispatch metadata/order and rounded scaling/addition for MoE."""

import pytest
import torch

import praxis.layers as layers
from praxis.kernels.layers_triton import moe_dispatch, moe_scale_add


def eager_dispatch(x, top_idx, gate, order, num_experts, output_dtype):
    """Reference dispatch using eager PyTorch operations."""
    tokens = (
        torch.arange(x.shape[0], device=x.device)
        .unsqueeze(1)
        .expand(-1, top_idx.shape[1])
        .reshape(-1)
    )
    rows = x[tokens[order]].to(output_dtype)
    sorted_gate = gate.reshape(-1)[order]
    offsets = torch.searchsorted(
        top_idx.reshape(-1)[order],
        torch.arange(1, num_experts + 1, device=x.device),
        out_int32=True,
    )
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=x.device)
    return rows, sorted_gate, inverse, offsets


def assert_bytes_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,channels,experts,k",
    [(0, 128, 16, 8), (1, 129, 17, 2), (33, 256, 16, 8), (257, 3072, 128, 8)],
)
@pytest.mark.parametrize("concentrated", [False, True])
def test_exact_dispatch(dtype, tokens, channels, experts, k, concentrated):
    torch.manual_seed(123)
    x = torch.randn(tokens, channels, device="cuda", dtype=dtype)
    ids = torch.randint(0, 1 if concentrated else experts, (tokens, k), device="cuda")
    gate = torch.randn(tokens, k, device="cuda", dtype=dtype)
    order = ids.flatten().argsort(stable=True)
    with torch.inference_mode():
        actual = moe_dispatch(x, ids, gate, order, experts, dtype)
        expected = eager_dispatch(x, ids, gate, order, experts, dtype)
    for a, b in zip(actual, expected, strict=True):
        assert_bytes_equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [0.0, -1.25, 2.5, 1.00013])
def test_scale_add_exact(dtype, scale):
    torch.manual_seed(321)
    routed = torch.randn(33, 129, device="cuda", dtype=dtype)
    shared = torch.randn_like(routed)
    routed[0, :8] = torch.tensor(
        [
            0.0,
            -0.0,
            torch.finfo(dtype).tiny,
            -torch.finfo(dtype).tiny,
            1.0,
            -1.0,
            128.0,
            -128.0,
        ],
        device="cuda",
        dtype=dtype,
    )
    with torch.inference_mode():
        assert_bytes_equal(
            moe_scale_add(routed, shared, scale), routed * scale + shared
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
def test_complete_moe_against_previous_dispatch(monkeypatch):
    torch.manual_seed(5)
    model = layers.MoEFeedForward(256, 128, 16, 8).cuda().bfloat16().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    x = torch.randn(1, 33, 256, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = model(x)
        monkeypatch.setattr(layers, "moe_dispatch", eager_dispatch)
        monkeypatch.setattr(layers, "moe_scale_add", lambda a, b, s: a * s + b)
        expected = model(x)
    assert_bytes_equal(actual, expected)


def test_cpu_fallback_gradients():
    x = torch.randn(3, 128, requires_grad=True)
    gate = torch.rand(3, 2, requires_grad=True)
    ids = torch.tensor([[0, 1], [1, 1], [0, 2]])
    order = ids.flatten().argsort(stable=True)
    actual = moe_dispatch(x, ids, gate, order, 4, x.dtype)
    expected = eager_dispatch(x, ids, gate, order, 4, x.dtype)
    for a, b in zip(actual, expected, strict=True):
        assert_bytes_equal(a, b)
    out = moe_scale_add(actual[0], torch.ones_like(actual[0]), 2.5)
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
def test_dispatch_graph_replays_changed_routing():
    x = torch.randn(33, 129, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 17, (33, 8), device="cuda")
    gate = torch.rand(33, 8, device="cuda", dtype=torch.bfloat16)
    order = ids.flatten().argsort(stable=True)
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                moe_dispatch(x, ids, gate, order, 17, x.dtype)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = moe_dispatch(x, ids, gate, order, 17, x.dtype)
            epilogue = moe_scale_add(actual[0], actual[0], 2.5)
        for concentrated in (False, True):
            ids.random_(0, 1 if concentrated else 17)
            order.copy_(ids.flatten().argsort(stable=True))
            x.normal_()
            gate.uniform_()
            graph.replay()
            expected = eager_dispatch(x, ids, gate, order, 17, x.dtype)
            for a, b in zip(actual, expected, strict=True):
                assert_bytes_equal(a, b)
            assert_bytes_equal(epilogue, expected[0] * 2.5 + expected[0])
