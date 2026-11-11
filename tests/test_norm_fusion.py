"""CPU checks for normalization fusion fallbacks and weight-cache invalidation."""

import copy
from types import SimpleNamespace

import pytest
import torch

from praxis.layers import RMSNorm, residual_norm_scale, rms_norm_scale


@pytest.mark.parametrize("affine", [False, True])
def test_cpu_norm_helpers_keep_outputs_and_gradients(affine):
    norm = RMSNorm(7, elementwise_affine=affine)
    reference = copy.deepcopy(norm)
    x = torch.randn(2, 3, 7, requires_grad=True)
    delta = torch.randn_like(x, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    dd = delta.detach().clone().requires_grad_()
    scale = torch.tensor(0.70710677)
    residual, actual = residual_norm_scale(x, delta, norm, scale)
    expected_residual = xx + dd
    expected = reference(expected_residual) * scale
    assert torch.equal(residual, expected_residual)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, xx.grad)
    assert torch.equal(delta.grad, dd.grad)
    if affine:
        assert torch.equal(norm.weight.grad, reference.weight.grad)
    with torch.no_grad():
        assert torch.equal(rms_norm_scale(norm, x, scale), norm(x) * scale)


def test_inference_weight_cache_follows_parameter_updates_and_apply():
    norm = RMSNorm(7)
    with torch.no_grad():
        first = norm._inference_weight()
        assert norm._inference_weight() is first
        norm.weight.add_(0.125)
        second = norm._inference_weight()
        assert second is not first
        assert torch.equal(second, 1 + norm.weight.float())
        state = copy.deepcopy(norm.state_dict())
        state["weight"].fill_(0.25)
        norm.load_state_dict(state)
        assert torch.equal(norm._inference_weight(), 1 + norm.weight.float())
        norm.to(torch.float64)
        assert norm._norm_weight_cache is None
        assert list(norm.state_dict()) == ["weight"]


def test_norm_helpers_preserve_module_hooks():
    norm = RMSNorm(7)
    calls = []
    handle = norm.register_forward_hook(lambda *_: calls.append(1))
    try:
        with torch.no_grad():
            x = torch.randn(2, 7)
            rms_norm_scale(norm, x, torch.tensor(0.5))
            residual_norm_scale(x, x, norm, torch.tensor(0.5))
        assert calls == [1, 1]
    finally:
        handle.remove()


def test_inference_tensor_weights_are_not_cached():
    with torch.inference_mode():
        norm = RMSNorm(7)
        first = norm._inference_weight()
        norm.weight.add_(0.25)
        assert torch.equal(norm._inference_weight(), 1 + norm.weight.float())
        assert not torch.equal(first, norm._inference_weight())
        assert norm._norm_weight_cache is None


def test_norm_affine_cache_observes_weight_updates():
    from praxis.kernels.norm_triton import _w1

    norm = RMSNorm(16)
    with torch.no_grad():
        first = _w1(norm)
        norm.weight.add_(0.125)
        actual = _w1(norm)
        assert torch.equal(actual, 1 + norm.weight.float())
        assert not torch.equal(actual, first)


def test_norm_scalar_cache_observes_buffer_updates():
    from praxis.kernels.norm_triton import _scale

    scale = torch.tensor(0.5, dtype=torch.bfloat16)
    assert _scale(scale) == 0.5
    scale.fill_(0.25)
    assert _scale(scale) == 0.25
    with torch.inference_mode():
        scale = torch.tensor(0.5, dtype=torch.bfloat16)
        assert _scale(scale) == 0.5
        scale.fill_(0.25)
        assert _scale(scale) == 0.25


@pytest.mark.parametrize(
    ("arch", "dtype", "cuda", "expected"),
    [
        ((10, 0), torch.bfloat16, True, True),
        ((9, 0), torch.bfloat16, True, True),
        ((10, 3), torch.bfloat16, True, True),
        ((11, 0), torch.bfloat16, True, True),
        ((12, 0), torch.bfloat16, True, True),
        ((13, 0), torch.bfloat16, True, True),
        ((9, 0), torch.float16, True, False),
        ((9, 0), torch.float32, True, False),
        ((9, 0), torch.bfloat16, False, False),
        ((10, 0), torch.float16, True, False),
        ((10, 0), torch.float32, True, False),
        ((10, 0), torch.bfloat16, False, False),
    ],
)
def test_automatic_norm_selection_is_limited_to_supported_inputs(
    monkeypatch, arch, dtype, cuda, expected
):
    from praxis import layers

    monkeypatch.setattr(layers, "FUSED_NORM", None)

    def unexpected_capability_query(*_):
        pytest.fail(f"Fusion selection must not query GPU architecture {arch}")

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", unexpected_capability_query
    )
    x = SimpleNamespace(is_cuda=cuda, dtype=dtype, device=torch.device("cuda"))
    with torch.no_grad():
        assert layers._fused_norm_enabled(x) == expected
    with torch.enable_grad():
        assert not layers._fused_norm_enabled(x)


def test_automatic_norm_respects_autocast(monkeypatch):
    from praxis import layers

    monkeypatch.setattr(layers, "FUSED_NORM", None)
    x = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    with torch.no_grad():
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda: True)
        assert not layers._fused_norm_enabled(x)


@pytest.mark.parametrize("enabled", [True, False])
def test_norm_selection_preserves_explicit_overrides(monkeypatch, enabled):
    from praxis import layers

    monkeypatch.setattr(layers, "FUSED_NORM", enabled)
    assert layers._fused_norm_enabled(torch.empty(1)) == enabled


def _small_transformer():
    from praxis.layers import Transformer

    class Projection(torch.nn.Module):
        backend = "triton"

        def forward(self, x, *args):
            return x * 0.25

    torch.manual_seed(17)
    model = Transformer(
        depth=3,
        num_dense=3,
        dim=16,
        head_dim=8,
        num_kv_heads=1,
        dense_hidden=24,
        rope_effective_dim=4,
    ).eval()
    for block in model.blocks:
        block.mha = Projection()
    return model


def test_generator_preserves_residuals_and_taps_on_the_cpu_fallback(monkeypatch):
    from praxis import layers

    model = _small_transformer()
    x, positions = torch.randn(1, 7, 16), torch.arange(7)[None]
    with torch.no_grad():
        monkeypatch.setattr(layers, "_fused_norm_enabled", lambda _: False)
        expected = model(x, positions, tap_blocks=(0, 2))
        monkeypatch.setattr(layers, "_fused_norm_enabled", lambda _: True)
        actual = model(x, positions, tap_blocks=(0, 2))
    for a, b in zip([actual[0], *actual[1]], [expected[0], *expected[1]], strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("customization", ["hook", "pre_hook", "forward", "stack"])
def test_normalization_keeps_block_hooks_and_overrides(monkeypatch, customization):
    from praxis import layers

    model = _small_transformer()
    block = model.blocks[1]
    calls = []

    def change_output(_module, _args, output):
        calls.append(1)
        return output + 0.125

    if customization == "hook":
        block.register_forward_hook(change_output)
    elif customization == "pre_hook":
        block.register_forward_pre_hook(lambda *_: calls.append(1))
    elif customization == "forward":
        original = block.forward
        monkeypatch.setattr(
            block, "forward", lambda *a, **kw: original(*a, **kw) + 0.125
        )
    else:
        original = model.block_forward
        monkeypatch.setattr(
            model, "block_forward", lambda *a, **kw: original(*a, **kw) + 0.125
        )
    x, positions = torch.randn(1, 7, 16), torch.arange(7)[None]
    with torch.no_grad():
        monkeypatch.setattr(layers, "_fused_norm_enabled", lambda _: False)
        expected = model(x, positions)[0]
        calls.clear()
        monkeypatch.setattr(layers, "_fused_norm_enabled", lambda _: True)
        actual = model(x, positions)[0]
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    if customization in ("hook", "pre_hook"):
        assert calls == [1]


def test_global_hooks_still_observe_transformer_blocks(monkeypatch):
    from praxis import layers

    model = _small_transformer()
    calls = []
    handle = torch.nn.modules.module.register_module_forward_hook(
        lambda module, *_: calls.append(module)
    )
    try:
        monkeypatch.setattr(layers, "_fused_norm_enabled", lambda _: True)
        with torch.no_grad():
            model(torch.randn(1, 7, 16), torch.arange(7)[None])
        assert all(block in calls for block in model.blocks)
    finally:
        handle.remove()
