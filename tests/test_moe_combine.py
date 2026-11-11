"""Check fused MoE combination against the original eager operation sequence."""

import pytest
import torch

import praxis.layers as layers
from praxis.kernels.layers_triton import moe_combine


def eager_combine(out, gate, inverse, top_k):
    weighted = out * gate.unsqueeze(-1)
    return weighted[inverse].reshape(-1, top_k, out.shape[1]).sum(dim=1)


def assert_bytes_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,channels", [(0, 128), (1, 128), (17, 129), (257, 1024), (31, 3072)]
)
def test_cuda_exact_rounding_and_channel_tails(dtype, tokens, channels):
    torch.manual_seed(11)
    out = torch.randn(tokens * 8, channels, device="cuda", dtype=dtype)
    # Exercise cancellation, wide magnitudes, negative and zero gates.
    out *= torch.tensor([0, 0.125, 1, 128], device="cuda", dtype=dtype)[
        torch.arange(tokens * 8, device="cuda") % 4
    ][:, None]
    gate = torch.randn(tokens * 8, device="cuda", dtype=dtype)
    inverse = torch.randperm(tokens * 8, device="cuda")
    with torch.inference_mode():
        expected = eager_combine(out, gate, inverse, 8)
        actual = moe_combine(out, gate, inverse, 8)
        assert_bytes_equal(actual, expected)
        assert_bytes_equal(actual, moe_combine(out, gate, inverse, 8))


@pytest.mark.parametrize("top_k", [2, 8])
def test_cpu_fallback_preserves_autograd(top_k):
    torch.manual_seed(4)
    out = torch.randn(3 * top_k, 128, requires_grad=True)
    gate = torch.randn(3 * top_k, requires_grad=True)
    inverse = torch.randperm(3 * top_k)
    actual = moe_combine(out, gate, inverse, top_k)
    expected = eager_combine(out, gate, inverse, top_k)
    assert_bytes_equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (out, gate))
    expected_grad = torch.autograd.grad(expected.square().sum(), (out, gate))
    for a, b in zip(actual_grad, expected_grad, strict=True):
        assert_bytes_equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fallback")
def test_cuda_autocast_preserves_eager_sum_dtype():
    out = torch.randn(24, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.rand(24, device="cuda", dtype=torch.bfloat16)
    inverse = torch.randperm(24, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = eager_combine(out, gate, inverse, 8)
        actual = moe_combine(out, gate, inverse, 8)
    assert expected.dtype == torch.float32
    assert_bytes_equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fallback")
def test_cuda_differentiable_call_preserves_autograd():
    out = torch.randn(24, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate = torch.rand(24, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    inverse = torch.randperm(24, device="cuda")
    actual = moe_combine(out, gate, inverse, 8)
    expected = eager_combine(out, gate, inverse, 8)
    assert_bytes_equal(actual, expected)
    a = torch.autograd.grad(actual.float().square().sum(), (out, gate))
    b = torch.autograd.grad(expected.float().square().sum(), (out, gate))
    for left, right in zip(a, b, strict=True):
        assert_bytes_equal(left, right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_small_values_and_signed_zero(dtype):
    tiny = torch.finfo(dtype).tiny
    values = torch.tensor(
        [tiny, -tiny, tiny / 2, -tiny / 2, 0.0, -0.0, 1, -1], device="cuda", dtype=dtype
    )
    # Put one nonzero expert in each column, so opposite tiny values cannot
    # cancel and hide a subnormal-rounding difference.
    selected = torch.arange(128, device="cuda")[None, :] % 8
    out = torch.where(
        torch.arange(8, device="cuda")[:, None] == selected,
        values[:, None],
        0,
    ).contiguous()
    gate = torch.full((8,), 0.125, device="cuda", dtype=dtype)
    inverse = torch.arange(8, device="cuda")
    with torch.inference_mode():
        assert_bytes_equal(
            moe_combine(out, gate, inverse, 8), eager_combine(out, gate, inverse, 8)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
def test_cuda_graph_replays_changed_inputs():
    out = torch.randn(17 * 8, 256, device="cuda", dtype=torch.bfloat16)
    gate = torch.rand(17 * 8, device="cuda", dtype=torch.bfloat16)
    inverse = torch.randperm(17 * 8, device="cuda")
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                moe_combine(out, gate, inverse, 8)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = moe_combine(out, gate, inverse, 8)
        for _ in range(3):
            out.normal_()
            gate.uniform_()
            graph.replay()
            assert_bytes_equal(actual, eager_combine(out, gate, inverse, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped GEMM")
@pytest.mark.parametrize("concentrated", [False, True])
def test_complete_moe_preserves_outputs(monkeypatch, concentrated):
    torch.manual_seed(5)
    module = layers.MoEFeedForward(256, 128, 16, 8).cuda().bfloat16().eval()
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    if concentrated:
        module.expert_bias[:8] = 100
    x = torch.randn(1, 33, 256, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = module(x)
        monkeypatch.setattr(layers, "moe_combine", eager_combine)
        expected = module(x)
    assert_bytes_equal(actual, expected)
