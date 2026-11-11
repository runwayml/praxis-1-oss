"""CPU fallback parity and portable fused expert dispatch."""

import pytest
import torch

from praxis import layers
from praxis.layers import MoEFeedForward


@pytest.mark.parametrize("chunk_size", [None, 3, 16])
@pytest.mark.parametrize("fused", [False, True])
def test_paired_experts_cpu_fallback_preserves_outputs_and_gradients(
    monkeypatch, chunk_size, fused
):
    monkeypatch.setattr(layers, "FUSED_MOE", fused)
    torch.manual_seed(42)
    module = MoEFeedForward(16, 8, 4, 2, dispatch_chunk_size=chunk_size)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.02)
    xs = [torch.randn(1, n, 16, requires_grad=True) for n in (4, 7)]
    expected = [module(x) for x in xs]
    expected_grad = torch.autograd.grad(sum(y.square().sum() for y in expected), xs)
    actual = module.forward_pair(xs)
    actual_grad = torch.autograd.grad(sum(y.square().sum() for y in actual), xs)
    for left, right in zip(
        expected + list(expected_grad), actual + list(actual_grad), strict=True
    ):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@pytest.mark.parametrize("arch", [(9, 0), (10, 0), (10, 3), (11, 0), (12, 0)])
def test_solo_fusion_dispatch_has_no_architecture_allowlist(monkeypatch, arch):
    def unexpected_query(*_):
        pytest.fail(f"Portable Triton dispatch must not query architecture {arch}")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_query)
    monkeypatch.setattr(layers, "FUSED_MOE_DEC", True)
    module = MoEFeedForward(16, 8, 4, 2)
    x = torch.randn(1, 3, 16)
    fused = torch.randn(3, 16)
    monkeypatch.setattr(module, "_forward_fused_solo", lambda _: fused)
    assert torch.equal(module(x), fused.view_as(x))


def test_solo_fusion_preserves_cached_router_path(monkeypatch):
    monkeypatch.setattr(layers, "FUSED_MOE_DEC", True)
    torch.manual_seed(42)
    module = MoEFeedForward(16, 8, 4, 2)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.02)
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        expected = module(x, router_rows=(0, 3, 5))

    def unexpected_fusion(_):
        pytest.fail("Cached routing must preserve the full router GEMM shape")

    monkeypatch.setattr(module, "_forward_fused_solo", unexpected_fusion)
    with torch.no_grad():
        assert torch.equal(module(x, router_rows=(0, 3, 5)), expected)
