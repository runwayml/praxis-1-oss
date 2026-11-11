"""CPU fallbacks retain rotary normalization outputs, gradients and hooks."""

import copy
from types import SimpleNamespace

import pytest
import torch

from praxis.layers import RMSNorm, apply_rope, apply_rope_norm


@pytest.mark.parametrize("with_rope", [False, True])
@pytest.mark.parametrize("mode", [False, True, None])
def test_rope_norm_cpu_outputs_and_gradients(monkeypatch, with_rope, mode):
    from praxis import layers

    monkeypatch.setattr(layers, "FUSED_ROPE_QK_NORM", mode)
    q = torch.randn(1, 2, 5, 8, requires_grad=True)
    k = torch.randn(1, 1, 5, 8, requires_grad=True)
    qq = q.detach().clone().requires_grad_()
    kk = k.detach().clone().requires_grad_()
    norms = [RMSNorm(8), RMSNorm(8)]
    reference = copy.deepcopy(norms)
    table = torch.randn(1, 5, 4, 2) if with_rope else None
    actual = apply_rope_norm(q, k, table, *norms)
    qr, kr = apply_rope(qq, kk, table) if with_rope else (qq, kk)
    expected = reference[0](qr), reference[1](kr)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    sum(x.square().sum() for x in actual).backward()
    sum(x.square().sum() for x in expected).backward()
    assert torch.equal(q.grad, qq.grad)
    assert torch.equal(k.grad, kk.grad)
    for a, b in zip(norms, reference, strict=True):
        assert torch.equal(a.weight.grad, b.weight.grad)


@pytest.mark.parametrize("mode", [False, True, None])
def test_rope_norm_cpu_preserves_hooks(monkeypatch, mode):
    from praxis import layers

    monkeypatch.setattr(layers, "FUSED_ROPE_QK_NORM", mode)
    norms = [RMSNorm(8), RMSNorm(8)]
    calls = []
    handle = norms[0].register_forward_hook(lambda *_: calls.append(1))
    try:
        with torch.no_grad():
            apply_rope_norm(
                torch.randn(1, 2, 5, 8), torch.randn(1, 1, 5, 8), None, *norms
            )
        assert calls == [1]
    finally:
        handle.remove()


@pytest.mark.parametrize(
    ("arch", "dtype", "cuda", "expected"),
    [
        ((10, 0), torch.bfloat16, True, True),
        ((9, 0), torch.bfloat16, True, True),
        ((10, 3), torch.bfloat16, True, True),
        ((11, 0), torch.bfloat16, True, True),
        ((12, 0), torch.bfloat16, True, True),
        ((13, 0), torch.bfloat16, True, True),
        ((9, 0), torch.float16, True, True),
        ((9, 0), torch.float32, True, False),
        ((9, 0), torch.bfloat16, False, False),
        ((10, 0), torch.float16, True, True),
        ((10, 0), torch.float32, True, False),
        ((10, 0), torch.bfloat16, False, False),
    ],
)
def test_automatic_rope_norm_selection(monkeypatch, arch, dtype, cuda, expected):
    from praxis import layers

    monkeypatch.setattr(layers, "FUSED_ROPE_QK_NORM", None)

    def unexpected_capability_query(*_):
        pytest.fail(f"Fusion selection must not query GPU architecture {arch}")

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", unexpected_capability_query
    )
    q = SimpleNamespace(is_cuda=cuda, dtype=dtype, device=torch.device("cuda"))
    with torch.no_grad():
        assert layers._fused_rope_qk_norm_enabled(q) == expected
    with torch.enable_grad():
        assert not layers._fused_rope_qk_norm_enabled(q)


@pytest.mark.parametrize("enabled", [False, True])
def test_rope_norm_preserves_explicit_overrides(monkeypatch, enabled):
    from praxis import layers

    monkeypatch.setattr(layers, "FUSED_ROPE_QK_NORM", enabled)
    assert layers._fused_rope_qk_norm_enabled(torch.empty(1)) == enabled
