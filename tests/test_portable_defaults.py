"""CPU checks for portable tuning defaults and numerical compatibility guards."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from praxis import layers, single_arm
from praxis.kernels import moe_gemm_triton, nvfp4


def no_architecture_query(*args):
    pytest.fail("Portable defaults must not query GPU architecture")


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
@pytest.mark.parametrize("rows", [(8,), (4, 4)])
@pytest.mark.parametrize("arch", [(8, 0), (9, 0), (10, 0), (10, 3), (11, 0), (12, 0)])
def test_fused_router_support_requires_blackwell_tcgen05(
    monkeypatch, device, rows, arch
):
    from praxis.kernels import router_fused

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: arch)
    common = dict(dtype=torch.float32, device=torch.device(device))
    logits = SimpleNamespace(
        **common,
        is_cuda=True,
        shape=(8, 128),
        dim=lambda: 2,
        is_contiguous=lambda: True,
    )
    bias = SimpleNamespace(**common, is_contiguous=lambda: True, numel=lambda: 128)
    assert router_fused.supported(logits, bias, rows) is (arch[0] in (10, 11))


@pytest.mark.parametrize("arch", [(8, 0), (9, 0), (12, 0)])
def test_fused_router_rejects_unsupported_arch_before_routing(monkeypatch, arch):
    monkeypatch.setattr(layers, "ROUTER_FUSED", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: arch)
    layer = SimpleNamespace(
        routing_hook=None, top_k=8, num_experts=128, expert_bias=SimpleNamespace()
    )
    x = SimpleNamespace(
        is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda:0")
    )
    with torch.no_grad():
        assert layers.MoEFeedForward._route_fused(layer, [x]) is None


def test_fused_router_compilation_cache_is_per_device(monkeypatch):
    from praxis.kernels import router_fused

    first, second = object(), object()
    tree = (0, 1)
    monkeypatch.setattr(
        router_fused,
        "_COMPILED",
        {
            (0, 1, False, "ptx", repr(tree), 1): first,
            (1, 1, False, "ptx", repr(tree), 1): second,
        },
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert router_fused._compiled(1, tree=tree) is first
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert router_fused._compiled(1, tree=tree) is second


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_fused_router_keeps_other_gate_dtypes_on_stock_path(dtype):
    layer = SimpleNamespace(
        routing_hook=None,
        top_k=8,
        num_experts=128,
        expert_bias=SimpleNamespace(),
    )
    x = SimpleNamespace(is_cuda=True, dtype=dtype)
    with torch.no_grad():
        assert layers.MoEFeedForward._route_fused(layer, [x]) is None


@pytest.mark.parametrize("arch", [(9, 0), (10, 0), (10, 3), (11, 0), (12, 0)])
@pytest.mark.parametrize("setting", [None, False, True])
def test_fused_moe_device_default_and_overrides(monkeypatch, arch, setting):
    calls = []

    def capability(device):
        calls.append(device)
        return arch

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    x = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    expected = arch != (9, 0) if setting is None else setting
    assert layers._fused_moe_enabled(setting, x) is expected
    assert calls == ([x.device] if setting is None else [])


def test_fused_moe_cpu_default_never_queries_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_architecture_query)
    assert not layers._fused_moe_enabled(None, torch.zeros(1))


@pytest.mark.parametrize("name", ["FUSED_MOE_SOLO"])
@pytest.mark.parametrize("arch", [(9, 0), (10, 0)])
def test_fused_moe_defaults_disable_only_sm90(monkeypatch, name, arch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: arch)
    x = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    assert layers._fused_moe_enabled(getattr(layers, name), x) is (arch != (9, 0))


@pytest.mark.parametrize("arch", [(9, 0), (10, 0), (10, 3), (11, 0), (12, 0)])
@pytest.mark.parametrize("name", ["FUSED_MOE", "FUSED_MOE_DEC"])
def test_expert_fusion_is_opt_in_on_every_architecture(monkeypatch, arch, name):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: arch)
    x = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    assert not layers._fused_moe_enabled(getattr(layers, name), x)
    assert layers._fused_moe_enabled(True, x)


@pytest.mark.parametrize("rows", [39_999, 40_000, 47_999, 48_000])
def test_encoder_tile_selection_uses_rows_and_preserves_overrides(monkeypatch, rows):
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_architecture_query)
    monkeypatch.setattr(layers, "ENC_GATE_UP_AUTO", True)
    monkeypatch.setattr(layers, "CUTE_EXPERTS", False)
    monkeypatch.setattr(layers, "ROUTED_QUACK_DOWN", False)
    model = layers.MoEFeedForward(16, 8, 4, 2)
    seen = []

    def gate_up(*args, **kwargs):
        seen.append(kwargs["config"])
        return torch.zeros(1, 8)

    monkeypatch.setattr(moe_gemm_triton, "grouped_gate_up_swiglu", gate_up)
    monkeypatch.setattr(
        moe_gemm_triton, "grouped_mm", lambda *a, **kw: torch.zeros(1, 16)
    )
    args = (torch.zeros(1, 16), torch.arange(rows), torch.tensor([rows]))
    model._fused_expert_gemms(*args, None, None)
    assert seen[-1] is (
        layers.ENC_GATE_UP_BM128
        if 40_000 <= rows < 48_000
        else layers.ENC_GATE_UP_BM256
    )
    override = {"BM": 64}
    model._fused_expert_gemms(*args, gate_up_config=override, down_config=None)
    assert seen[-1] is override
    monkeypatch.setattr(layers, "ENC_GATE_UP_AUTO", False)
    model._fused_expert_gemms(*args, None, None)
    assert seen[-1] is None


@pytest.mark.parametrize("shape", [(234, 416), (288, 512)])
@pytest.mark.parametrize("antialias", [False, True])
def test_gpu_tile_default_requires_no_resize(monkeypatch, shape, antialias):
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_architecture_query)
    monkeypatch.setattr(single_arm, "TILE_ON_DEVICE", True)
    monkeypatch.setattr(single_arm, "THOR_TILE_RESIZE", False)
    frames = np.zeros((1, *shape, 3), dtype=np.uint8)
    assert single_arm._tile_on_device(
        frames, frames, 224, 416, 0.03, "cuda", antialias
    ) == (shape == (234, 416))
    monkeypatch.setattr(single_arm, "TILE_ON_DEVICE", False)
    assert not single_arm._tile_on_device(
        frames, frames, 224, 416, 0.03, "cuda", antialias
    )


def test_gpu_tile_default_checks_both_views(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_architecture_query)
    monkeypatch.setattr(single_arm, "TILE_ON_DEVICE", True)
    monkeypatch.setattr(single_arm, "THOR_TILE_RESIZE", False)
    fit = np.zeros((25, 234, 416, 3), dtype=np.uint8)
    resize = np.zeros((25, 288, 512, 3), dtype=np.uint8)
    assert not single_arm._tile_on_device(fit, resize, 224, 416, 0.03, "cuda", True)
    assert not single_arm._tile_on_device(resize, fit, 224, 416, 0.03, "cuda", True)


@pytest.mark.parametrize("capability", [(10, 0), (11, 0)])
def test_resize_experiment_remains_thor_only_and_opt_in(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    frames = np.zeros((1, 288, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(single_arm, "TILE_ON_DEVICE", True)
    monkeypatch.setattr(single_arm, "THOR_TILE_RESIZE", True)
    monkeypatch.setattr(single_arm, "TILE_ON_DEVICE_ANTIALIAS", False)
    assert single_arm._tile_on_device(
        frames, frames, 224, 416, 0.03, "cuda", False
    ) == (capability == (11, 0))
    assert not single_arm._tile_on_device(frames, frames, 224, 416, 0.03, "cuda", True)
    monkeypatch.setattr(single_arm, "TILE_ON_DEVICE_ANTIALIAS", True)
    assert single_arm._tile_on_device(frames, frames, 224, 416, 0.03, "cuda", True) == (
        capability == (11, 0)
    )


def test_wide_quantization_is_portable_and_reversible(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_architecture_query)
    monkeypatch.setattr(nvfp4, "QUANT_WIDE", True)
    assert nvfp4._quantize_launch() == (64, 8)
    monkeypatch.setattr(nvfp4, "QUANT_WIDE", False)
    assert nvfp4._quantize_launch() == (8, 4)


def _fake_cuda(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _=None: capability)
    monkeypatch.setattr("praxis.kernels.cute_experts._cfg", lambda *c: c)


@pytest.mark.parametrize(
    ("capability", "gated", "table"),
    [
        ((9, 0), False, False),  # H100: ordinary ops
        ((10, 0), True, False),  # B200: tcgen05, empty table -> ordinary ops
        ((10, 3), True, False),
        ((11, 0), True, True),  # Thor: measured table
        ((12, 0), False, False),
    ],
)
def test_cute_dense_ffn_gate(monkeypatch, capability, gated, table):
    from praxis.kernels import cute_dense

    assert layers.CUTE_DENSE_FFN is False
    _fake_cuda(monkeypatch, capability)
    monkeypatch.setattr(cute_dense, "quack_available", lambda: True)
    assert cute_dense.supported() is gated
    assert (cute_dense.gate_up_config(3470) is not None) is table
    assert cute_dense.gate_up_config(100) is None
    assert cute_dense.down_config(3470) is None


def test_cute_dense_ffn_needs_quack(monkeypatch):
    from praxis.kernels import cute_dense

    _fake_cuda(monkeypatch, (11, 0))
    monkeypatch.setattr(cute_dense, "quack_available", lambda: False)
    assert not cute_dense.supported()


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (11, 0)])
def test_cute_dense_ffn_falls_back_silently(monkeypatch, capability):
    """Unsupported devices, empty tables and non-CUDA inputs run the ordinary ops."""
    from praxis.kernels import cute_dense

    _fake_cuda(monkeypatch, capability)
    monkeypatch.setattr(cute_dense, "quack_available", lambda: True)
    monkeypatch.setattr(layers, "CUTE_DENSE_FFN", True)
    ffn = layers.SwiGLUFeedForward(16, 32)
    assert ffn._cute_hidden(torch.randn(1, 3470, 16)) is None


def test_cute_dense_ffn_falls_back_on_cpu(monkeypatch):
    monkeypatch.setattr(layers, "CUTE_DENSE_FFN", True)
    torch.manual_seed(0)
    ffn = layers.SwiGLUFeedForward(16, 32)
    for p in ffn.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(3, 300, 16)
    with torch.no_grad():
        a, b = ffn[0].linear(x)
        expected = ffn[1](torch.nn.functional.silu(a) * b)
        torch.testing.assert_close(ffn(x), expected, rtol=0, atol=0)
