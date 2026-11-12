"""Contracts for routed-only NVFP4.

Most tests run on the CPU without a checkpoint. The kernel tests need a CUDA
GPU and the checkpoint packing test a checkpoint directory; both skip without
them. Packing is stubbed where the test concerns ownership or strict loading. The
packing tests check that the FP4 codes and scales equal the QAT training
rounding bit for bit. No test here measures kernel speed or simulation quality.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tiny_model import tiny_config, tiny_policy
from torch import nn

from praxis.checkpoint import CheckpointError, save_checkpoint
from praxis.layers import MoEFeedForward, RoutedExperts, _routed_chunks
from praxis.model import PraxisPolicy
from praxis.quantization import loading, nvfp4
from praxis.quantization.loading import _cached_bank


def packed_state(e=2, c=128, h=128):
    return {
        "gate_up_q": torch.zeros(e, 2 * h, c // 2, dtype=torch.uint8),
        "gate_up_sf": torch.zeros(e, 2 * h, c // 16, dtype=torch.uint8),
        "gate_up_global": torch.ones(e),
        "down_q": torch.zeros(e, c, h // 2, dtype=torch.uint8),
        "down_sf": torch.zeros(e, c, h // 16, dtype=torch.uint8),
        "down_global": torch.ones(e),
        "_unit_alpha": torch.ones(e),
    }


def fake_matrix_pack(value):
    n, k = value.shape
    return (
        torch.zeros(n, k // 2, dtype=torch.uint8),
        torch.zeros(n, k // 16, dtype=torch.uint8),
        torch.ones(()),
    )


def test_packing_transposes_each_source_expert_without_retaining_weights(monkeypatch):
    gu = torch.arange(2 * 128 * 256).reshape(2, 128, 256).to(torch.bfloat16)
    down = torch.arange(2 * 128 * 128).reshape(2, 128, 128).to(torch.bfloat16)
    matrices = []

    def pack(value):
        assert value.device.type == "cpu" and value.dtype == torch.bfloat16
        assert value.is_contiguous() and not value.requires_grad
        matrices.append(value.clone())
        return fake_matrix_pack(value)

    monkeypatch.setattr(nvfp4, "_pack_matrix_cpu", pack)
    bank = nvfp4.pack_routed_experts(gu, down)
    assert len(matrices) == 4
    for actual, expected in zip(
        matrices, [gu[0].T, gu[1].T, down[0].T, down[1].T], strict=True
    ):
        assert torch.equal(actual, expected)
    del expected
    refs = weakref.ref(gu), weakref.ref(down)
    del gu, down
    assert all(ref() is None for ref in refs)
    assert list(bank.parameters()) == []
    assert set(bank.state_dict()) == set(packed_state())
    assert not any(value.dtype == torch.bfloat16 for value in bank.buffers())
    assert (bank.num_experts, bank.in_features, bank.hidden_features) == (2, 128, 128)


@pytest.mark.parametrize("kind", ["dtype", "alignment", "shape", "meta"])
def test_packing_rejects_invalid_source_before_optional_packer(monkeypatch, kind):
    def forbidden(_):
        pytest.fail("Invalid source reached the optional packer")

    monkeypatch.setattr(nvfp4, "_pack_matrix_cpu", forbidden)
    gu, down = (
        torch.zeros(2, 128, 256, dtype=torch.bfloat16),
        torch.zeros(2, 128, 128, dtype=torch.bfloat16),
    )
    if kind == "dtype":
        gu = gu.float()
    elif kind == "alignment":
        gu, down = gu[:, :64], down[:, :, :64]
    elif kind == "shape":
        down = down[:1]
    else:
        gu = gu.to("meta")
    with pytest.raises(ValueError):
        nvfp4.pack_routed_experts(gu, down)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "shape",
        "dtype",
        "global_nan",
        "global_zero",
        "scale_nan",
        "scale_negative",
        "alpha",
    ],
)
def test_reconstruction_rejects_corrupt_packed_state(kind):
    state = packed_state()
    if kind == "missing":
        del state["down_q"]
    elif kind == "extra":
        state["bf16_master"] = torch.ones(1)
    elif kind == "shape":
        state["down_q"] = state["down_q"][:, :, :1].contiguous()
    elif kind == "dtype":
        state["gate_up_global"] = state["gate_up_global"].to(torch.bfloat16)
    elif kind.startswith("global"):
        state["gate_up_global"][0] = float("nan") if kind == "global_nan" else 0
    elif kind.startswith("scale"):
        state["down_sf"].flatten()[0] = 127 if kind == "scale_nan" else 184
    else:
        state["_unit_alpha"][0] = 2
    with pytest.raises(ValueError):
        nvfp4.NVFP4RoutedExperts.from_packed(state)


def test_bank_is_inference_only_and_dtype_rejection_is_atomic():
    bank = nvfp4.NVFP4RoutedExperts.from_packed(packed_state())
    originals = dict(bank.named_buffers())
    assert bank.to("cpu") is bank
    with pytest.raises(TypeError, match="dtype"):
        bank.to(torch.bfloat16)
    assert all(bank.get_buffer(name) is value for name, value in originals.items())
    with pytest.raises(RuntimeError, match="inference"):
        bank.train()
    x = torch.ones(3, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA BF16"):
        bank.routed_forward(
            x,
            torch.zeros(3, 1, dtype=torch.int64),
            torch.ones(3, 1, dtype=torch.bfloat16),
            scaling=2.5,
            chunk_size=2,
        )
    with pytest.raises(RuntimeError, match="inference"):
        bank.routed_forward(
            x.requires_grad_(),
            torch.zeros(3, 1, dtype=torch.int64),
            torch.ones(3, 1, dtype=torch.bfloat16),
            scaling=2.5,
            chunk_size=2,
        )


def test_backend_protocol_keeps_router_shared_branch_and_scaling_exact():
    layer = next(
        value for value in tiny_policy().modules() if isinstance(value, MoEFeedForward)
    ).eval()
    original = layer.routed
    native = {
        name: value.clone()
        for name, value in layer.state_dict().items()
        if not name.startswith("routed.")
    }
    observed = []

    class OriginalOracle(nn.Module):
        def routed_forward(self, x, top_idx, gate, *, scaling, chunk_size):
            observed.append((x.dtype, gate.dtype, top_idx.clone(), scaling, chunk_size))
            return _routed_chunks(
                x,
                top_idx,
                gate,
                original.gate_up_weight,
                original.down_weight,
                scaling,
                chunk_size,
            )

    x = torch.randn(2, 43, 64).to(torch.bfloat16)
    with torch.no_grad():
        baseline = layer(x)
        layer.routed = OriginalOracle()
        actual = layer(x)
    assert torch.equal(actual, baseline)
    assert observed[0][:2] == (torch.bfloat16, torch.bfloat16)
    assert observed[0][3:] == (2.5, 40)
    assert observed[0][2].shape == (86, 2)
    assert all(
        torch.equal(layer.state_dict()[name], value) for name, value in native.items()
    )
    assert layer.expert_bias.dtype == torch.float32


def test_backend_can_add_shared_output_after_the_routed_bf16_boundary():
    layer = next(
        value for value in tiny_policy().modules() if isinstance(value, MoEFeedForward)
    ).eval()
    original = layer.routed
    x = torch.randn(2, 43, 64).to(torch.bfloat16)
    observed = []

    class SharedOracle(nn.Module):
        def routed_forward_with_shared(
            self, x, top_idx, gate, *, scaling, chunk_size, shared
        ):
            observed.append((shared.clone(), scaling, chunk_size))
            routed = _routed_chunks(
                x,
                top_idx,
                gate,
                original.gate_up_weight,
                original.down_weight,
                scaling,
                chunk_size,
            )
            return routed + shared

        def routed_forward(self, *args, **kwargs):
            pytest.fail("The shared-output backend used the separate addition path")

    with torch.no_grad():
        expected = layer(x)
        expected_shared = layer.shared(x.reshape(-1, x.shape[-1]))
        layer.routed = SharedOracle()
        actual = layer(x)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(observed[0][0], expected_shared)
    assert observed[0][1:] == (2.5, 40)


@pytest.mark.parametrize("kind", ["float32", "noncontiguous", "gradient"])
def test_shared_output_fallback_preserves_native_addition(monkeypatch, kind):
    bank = nvfp4.NVFP4RoutedExperts.from_packed(packed_state())
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    routed = torch.randn_like(x)
    shared = torch.randn_like(x)
    if kind == "float32":
        shared = shared.float()
    elif kind == "noncontiguous":
        shared = torch.randn(128, 2, dtype=torch.bfloat16).T
    else:
        shared.requires_grad_(True)
    monkeypatch.setattr(bank, "routed_forward", lambda *args, **kwargs: routed)
    actual = bank.routed_forward_with_shared(
        x,
        torch.zeros(2, 1, dtype=torch.int64),
        torch.ones(2, 1, dtype=torch.bfloat16),
        scaling=2.5,
        chunk_size=2,
        shared=shared,
    )
    expected = routed + shared
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    if kind == "gradient":
        actual.float().sum().backward()
        assert torch.equal(shared.grad, torch.ones_like(shared))


@pytest.fixture
def source_checkpoint(tmp_path):
    config = tiny_config()
    config["torch_dtype"] = "bfloat16"
    config["fp32_tensors"] = ["*.expert_bias"]
    config["backbone"]["dim"] = 128
    config["backbone"]["attention"]["num_heads"] = 8
    config["backbone"]["moe"]["expert_hidden"] = 128
    config["backbone"]["decoder"]["modulation_dim"] = 128
    config["action_expert"]["trunk_dim"] = 128
    model = PraxisPolicy.from_config(config).to(torch.bfloat16)
    with torch.no_grad():
        for value in model.parameters():
            value.normal_(0, 0.01)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "checkpoint"
    save_checkpoint(path, state, config)
    return path, state, config


def test_strict_load_quantizes_only_routed_weights_and_reuses_valid_cache(
    source_checkpoint, monkeypatch, tmp_path
):
    path, original, _ = source_checkpoint
    calls = []

    def pack(value):
        calls.append(value.shape)
        return fake_matrix_pack(value)

    monkeypatch.setattr(nvfp4, "_pack_matrix_cpu", pack)
    cache = tmp_path / "cache"
    model = PraxisPolicy.from_pretrained(
        path, device="cpu", routed_precision="nvfp4", nvfp4_cache=cache
    )
    assert len(calls) == 3 * 4 * 2
    for name, value in original.items():
        if ".routed." not in name:
            actual = model.state_dict()[name]
            assert actual.dtype == value.dtype and torch.equal(actual, value)
    banks = [
        layer.routed for layer in model.modules() if isinstance(layer, MoEFeedForward)
    ]
    assert len(banks) == 3 and all(
        isinstance(bank, nvfp4.NVFP4RoutedExperts) for bank in banks
    )
    assert not any(value.is_meta for value in model.state_dict().values())
    assert model.routed_quantization["non_routed_checkpoint_dtypes_preserved"]
    assert all(not row["cache_hit"] for row in model.routed_quantization["banks"])
    calls.clear()
    again = PraxisPolicy.from_pretrained(
        path, device="cpu", routed_precision="nvfp4", nvfp4_cache=cache
    )
    assert not calls and all(
        row["cache_hit"] for row in again.routed_quantization["banks"]
    )
    assert all(
        torch.equal(value, again.state_dict()[name])
        for name, value in model.state_dict().items()
    )
    (checkpoint_dir,) = cache.iterdir()
    assert len(checkpoint_dir.name) == 16
    assert {p.name for p in checkpoint_dir.iterdir()} == {
        f"{name}.pt"
        for name, layer in model.named_modules()
        if isinstance(layer, MoEFeedForward)
    }


def test_cache_keeps_checkpoints_apart(source_checkpoint, monkeypatch, tmp_path):
    path, state, config = source_checkpoint
    monkeypatch.setattr(nvfp4, "_pack_matrix_cpu", fake_matrix_pack)
    name = next(name for name in state if ".routed.down_weight" in name)
    state[name] = state[name] + 1
    other = tmp_path / "other"
    save_checkpoint(other, state, config)
    cache = tmp_path / "cache"
    for checkpoint in (path, other):
        PraxisPolicy.from_pretrained(
            checkpoint, device="cpu", routed_precision="nvfp4", nvfp4_cache=cache
        )
    assert len(list(cache.iterdir())) == 2
    again = PraxisPolicy.from_pretrained(
        other, device="cpu", routed_precision="nvfp4", nvfp4_cache=cache
    )
    assert all(row["cache_hit"] for row in again.routed_quantization["banks"])


def test_default_loader_preserves_original_bf16_path(source_checkpoint, monkeypatch):
    path, original, _ = source_checkpoint
    monkeypatch.setattr(
        nvfp4,
        "pack_routed_experts",
        lambda *_args, **_kwargs: pytest.fail("Default BF16 invoked packing"),
    )
    model = PraxisPolicy.from_pretrained(path, device="cpu")
    assert all(
        isinstance(layer.routed, RoutedExperts)
        for layer in model.modules()
        if isinstance(layer, MoEFeedForward)
    )
    assert all(
        value.dtype == original[name].dtype and torch.equal(value, original[name])
        for name, value in model.state_dict().items()
    )
    with pytest.raises(ValueError, match="routed_precision"):
        PraxisPolicy.from_pretrained("missing", device="cpu", routed_precision="fp4")


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype"])
def test_loader_checks_all_native_headers_before_packing(
    source_checkpoint, monkeypatch, tmp_path, corruption
):
    _, state, config = source_checkpoint
    name = next(name for name in state if name.endswith("expert_bias"))
    if corruption == "missing":
        del state[name]
    elif corruption == "extra":
        state["unexpected"] = torch.ones(1, dtype=torch.bfloat16)
    elif corruption == "shape":
        state[name] = state[name][:-1]
    else:
        state[name] = state[name].to(torch.bfloat16)
    broken = tmp_path / "broken"
    save_checkpoint(broken, state, config)
    monkeypatch.setattr(
        nvfp4,
        "pack_routed_experts",
        lambda *_args, **_kwargs: pytest.fail("Invalid checkpoint reached packing"),
    )
    with pytest.raises(CheckpointError):
        PraxisPolicy.from_pretrained(broken, device="cpu", routed_precision="nvfp4")


@pytest.mark.parametrize(
    "corruption", ["source", "torchao", "recipe", "packed_bytes", "checksum_keys"]
)
def test_cached_bank_rejects_wrong_source_or_corruption(
    tmp_path, monkeypatch, corruption
):
    monkeypatch.setattr(nvfp4, "_pack_matrix_cpu", fake_matrix_pack)
    gu, down = (
        torch.zeros(2, 128, 256, dtype=torch.bfloat16),
        torch.zeros(2, 128, 128, dtype=torch.bfloat16),
    )
    path = tmp_path / "bank.pt"
    _, hit, source = _cached_bank(gu, down, path)
    assert not hit
    assert "activation_scale" not in source
    if corruption == "source":
        gu[0, 0, 0] = 1
    elif corruption == "torchao":
        monkeypatch.setattr(loading, "_torchao_version", lambda: "0.0.0")
    elif corruption == "recipe":
        monkeypatch.setattr(nvfp4, "PACKING_RECIPE", "other-recipe")
    else:
        cached = torch.load(path, weights_only=True)
        if corruption == "packed_bytes":
            cached["buffers"]["down_q"][0, 0, 0] = 1
        else:
            del cached["sha256"]["down_q"]
        torch.save(cached, path)
    match = "delete this cache file" if corruption != "packed_bytes" else "cache"
    with pytest.raises(CheckpointError, match=match):
        _cached_bank(gu, down, path)


@pytest.mark.parametrize("quantized", [False, True])
def test_pipeline_forwards_precision_options_without_changing_native_loads(
    monkeypatch, quantized
):
    import praxis.pipeline as pipeline

    config = json.loads(
        (Path(__file__).parent / "data" / "praxis-1-droid.json").read_text()
    )
    calls = []
    monkeypatch.setattr(
        pipeline.PraxisPolicy,
        "from_pretrained",
        lambda *a, **kw: (calls.append((a, kw)), SimpleNamespace(config=config))[1],
    )
    monkeypatch.setattr(
        pipeline.Autoencoder,
        "from_pretrained",
        lambda *a, **kw: (
            calls.append((a, kw)),
            SimpleNamespace(enable_fused_decoder=lambda: None),
        )[1],
    )
    monkeypatch.setattr(pipeline, "TextEncoder", lambda *a, **kw: calls.append((a, kw)))
    options = (
        dict(
            routed_precision="nvfp4",
            nvfp4_cache=Path("cache"),
        )
        if quantized
        else {}
    )
    pipeline.PraxisPipeline("checkpoint", "vae", "text", device="cpu", **options)
    assert calls[0][1] == dict(
        device=torch.device("cpu"),
        # None keeps the checkpoint's attention backend.
        decoder_attention_backend=None,
        encoder_attention_backend=None,
        routed_precision="nvfp4" if quantized else "bf16",
        nvfp4_cache=Path("cache") if quantized else None,
    )
    assert calls[1] == (
        ("vae",),
        dict(device=torch.device("cpu"), norm_block_c=None, decoder_impl="reference"),
    )
    assert calls[2] == (("text",), dict(device=torch.device("cpu")))


def qat_weight_qdq_codes(weight):
    """The weight rounding of the QAT training forward, op for op.

    Global scale ``amax * (1/2688)`` (an FP32 reciprocal multiply), block
    scale ``block_amax * (1/6) / global``, payload
    ``x * ((1/global) / block_scale)`` clamped to +-6 with E2M1 RNE.
    """
    from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked

    n, k = weight.shape
    global_scale = weight.abs().amax().clamp_min(1e-12).float() * torch.tensor(
        1.0 / 2688.0
    )
    blocks = weight.float().reshape(n, k // 16, 16)
    block_scale = (
        ((blocks.abs().amax(-1) * torch.tensor(1.0 / 6.0)) / global_scale)
        .clamp(2.0**-6, 448.0)
        .to(torch.float8_e4m3fn)
    )
    reciprocal = (1.0 / global_scale) / block_scale.float()
    scaled = (blocks * reciprocal.unsqueeze(-1)).clamp(-6.0, 6.0).reshape(n, k)
    return f32_to_f4_unpacked(scaled), block_scale, global_scale


def unpack_matrix(q, sf, n, k):
    from torchao.prototype.mx_formats.utils import from_blocked

    codes = torch.stack([q & 0xF, q >> 4], -1).reshape(n, k)
    return codes, from_blocked(sf.view(torch.float8_e4m3fn), n, k // 16)


def assert_packing_matches_qat(weight):
    n, k = weight.shape
    q, sf, global_scale = nvfp4._pack_matrix_cpu(weight)
    codes, scales = unpack_matrix(q, sf, n, k)
    ref_codes, ref_scales, ref_global = qat_weight_qdq_codes(weight)
    assert torch.equal(codes, ref_codes)
    assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))
    assert global_scale.item() == ref_global.item()


@pytest.mark.parametrize("seed", [9, 10, 14, 28, 30, 34])
def test_packing_rounds_like_the_qat_training_kernel(seed):
    # These seeds round differently under true division by 2688 and 6.
    pytest.importorskip("torchao")
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(128, 256, generator=generator) * 0.02).to(torch.bfloat16)
    assert_packing_matches_qat(weight)


@pytest.mark.skipif(
    not os.environ.get("PRAXIS_NVFP4_TEST_CHECKPOINT"),
    reason="set PRAXIS_NVFP4_TEST_CHECKPOINT to a praxis checkpoint directory",
)
def test_packing_matches_qat_on_checkpoint_matrices():
    pytest.importorskip("torchao")
    from safetensors import safe_open

    root = Path(os.environ["PRAXIS_NVFP4_TEST_CHECKPOINT"])
    files = json.loads((root / "model.safetensors.index.json").read_text())
    routed = sorted(k for k in files["weight_map"] if ".ffn.routed." in k)[:2]
    for key in routed:
        with safe_open(str(root / files["weight_map"][key]), "pt") as reader:
            bank = reader.get_tensor(key)
        for expert in (0, bank.shape[0] // 2, bank.shape[0] - 1):
            assert_packing_matches_qat(bank[expert].t().contiguous())


def qat_grouped_global_scales(rows, offsets):
    """Global scales of the QAT grouped forward: per-group amax / 2688."""
    row_amax = rows.float().abs().amax(-1)
    full_offsets = torch.cat((offsets.new_zeros(1), offsets))
    group_amax = torch.segment_reduce(
        row_amax, reduce="max", offsets=full_offsets, initial=0.0
    )
    return group_amax.to(rows.dtype).clamp_min(1e-12).float() * torch.tensor(
        1.0 / 2688.0
    )


@pytest.mark.parametrize("gather", [False, True])
def test_activation_global_scales_follow_the_qat_formula(gather):
    pytest.importorskip("triton")
    from praxis.kernels.nvfp4 import expert_global_scales

    generator = torch.Generator().manual_seed(0)
    counts = torch.tensor([0, 5, 3, 0, 9, 1, 4, 0])  # empty first, middle, last
    offsets = counts.cumsum(0).to(torch.int32)
    csr = torch.nn.functional.pad(offsets, (1, 0))
    rows_total = int(counts.sum())
    x = (torch.randn(rows_total, 256, generator=generator) * 30).to(torch.bfloat16)
    tokens = torch.randint(0, rows_total, (rows_total,), generator=generator)
    rows = x[tokens] if gather else x
    actual = expert_global_scales(
        x if gather else rows, csr, tokens if gather else None
    )
    expected = qat_grouped_global_scales(rows, offsets)
    assert torch.equal(actual, expected)
    empty = actual[counts == 0]
    assert (empty > 0).all() and (empty == empty[0]).all()


def test_activation_global_scales_ignore_zero_padding_rows():
    # The hidden call quantizes padded rows; padding is zero and must not move
    # any expert's scale.
    pytest.importorskip("triton")
    from praxis.kernels.nvfp4 import expert_global_scales

    rows = torch.randn(6, 128).to(torch.bfloat16)
    csr = torch.tensor([0, 2, 6], dtype=torch.int32)
    padded = torch.zeros(12, 128, dtype=torch.bfloat16)
    padded[0:2], padded[4:8] = rows[0:2], rows[2:6]
    padded_csr = torch.tensor([0, 4, 12], dtype=torch.int32)
    assert torch.equal(
        expert_global_scales(padded, padded_csr), expert_global_scales(rows, csr)
    )


def test_activation_global_scale_clamp_rounds_to_bf16_before_float32_multiply():
    pytest.importorskip("triton")
    from praxis.kernels.nvfp4 import expert_global_scales

    rows = torch.zeros(3, 128, dtype=torch.bfloat16)
    rows[0] = -0.0
    csr = torch.tensor([0, 0, 2, 3], dtype=torch.int32)
    expected = torch.tensor(1e-12, dtype=torch.bfloat16).float() * (1.0 / 2688.0)
    actual = expert_global_scales(rows, csr)
    assert torch.equal(actual.view(torch.int32), expected.expand(3).view(torch.int32))


@pytest.mark.parametrize("gather", [False, True])
def test_activation_global_scales_assign_trailing_rows_to_last_expert(gather):
    pytest.importorskip("triton")
    from praxis.kernels.nvfp4 import expert_global_scales

    rows = torch.tensor([2, 4, 8, 16, 32, 64], dtype=torch.bfloat16)
    rows = rows[:, None].expand(-1, 128).contiguous()
    if gather:
        indices = torch.tensor([1, 0, 5, -2])
        csr = torch.tensor([0, 0, 2, 2], dtype=torch.int32)
        maxima = torch.tensor([0, 4, 64], dtype=torch.bfloat16)
    else:
        indices = None
        csr = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
        maxima = torch.tensor([2, 4, 64], dtype=torch.bfloat16)
    expected = maxima.clamp_min(1e-12).float() * (1.0 / 2688.0)
    assert torch.equal(expert_global_scales(rows, csr, indices), expected)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_precomputed_activation_globals_are_validated_before_kernel_launch(invalid):
    pytest.importorskip("triton")
    from praxis.kernels.nvfp4 import quantize_activations

    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    csr = torch.tensor([0, 2], dtype=torch.int32)
    scales = torch.ones(1)
    if invalid == "shape":
        scales = torch.ones(2)
    elif invalid == "dtype":
        scales = scales.bfloat16()
    else:
        scales = scales.to("meta")
    with pytest.raises(ValueError, match="FP32 activation global"):
        quantize_activations(
            x, csr, padded_csr=csr, padded_rows=2, activation_globals=scales
        )


class _StopAfterPipeline(Exception):
    pass


@pytest.mark.parametrize("quantized", [False, True])
def test_run_inference_passes_routed_precision(monkeypatch, quantized):
    import sys

    from scripts import run_inference

    seen = []

    def pipeline(**kwargs):
        seen.append(kwargs)
        raise _StopAfterPipeline

    argv = ["run_inference", "--config"]
    argv += [str(Path(__file__).resolve().parents[1] / "configs/inference_droid.yaml")]
    argv += ["--set", "model.device=cpu"]
    if quantized:
        argv += ["--set", "model.routed_precision=nvfp4"]
        argv += ["--set", "model.nvfp4_cache=cache"]
    monkeypatch.setattr(run_inference, "PraxisPipeline", pipeline)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_StopAfterPipeline):
        run_inference.main()
    assert seen[0]["routed_precision"] == ("nvfp4" if quantized else "bf16")
    assert seen[0]["nvfp4_cache"] == ("cache" if quantized else None)


@pytest.mark.parametrize(
    ("capability", "backend"),
    [
        ((10, 0), "flashinfer_sm100_contiguous_nvfp4"),
        ((10, 3), "flashinfer_sm100_contiguous_nvfp4"),
        ((12, 0), "flashinfer_sm120_grouped_nvfp4"),
    ],
)
def test_nvfp4_selects_native_backend_for_device(monkeypatch, capability, backend):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert nvfp4.backend_for_device("cuda:0") == backend


@pytest.mark.parametrize("capability", [(8, 0), (9, 0)])
def test_hopper_nvfp4_rejected_before_checkpoint_reads(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with pytest.raises(RuntimeError, match=r"bf16.*H100"):
        loading.load_routed_nvfp4(nn.Module(), "does-not-exist", device="cuda")


def test_nvfp4_refuses_a_gpu_without_nvfp4_before_loading(monkeypatch):
    from praxis.quantization import loading

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d=None: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d=None: "H100")
    monkeypatch.setattr(
        loading,
        "load_routed_nvfp4",
        lambda *_a, **_k: pytest.fail("loaded before the device check"),
    )
    monkeypatch.setattr(PraxisPolicy, "from_config", classmethod(lambda cls, _c: cls))
    monkeypatch.setattr("praxis.model.load_config", lambda _p: {})
    with pytest.raises(RuntimeError, match="SM110, or SM120"):
        PraxisPolicy.from_pretrained("unused", device="cuda", routed_precision="nvfp4")


@pytest.mark.parametrize(
    ("capability", "extra"), [((10, 0), "nvfp4"), ((11, 0), "thor"), ((12, 0), "nvfp4")]
)
def test_nvfp4_device_check_needs_the_backend_package(monkeypatch, capability, extra):
    from praxis.quantization import loading

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d=None: capability)
    monkeypatch.setattr(loading.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(ImportError, match=rf"\[{extra}\]"):
        loading.check_nvfp4_device("cuda")
    loading.check_nvfp4_device("cpu")


def test_default_imports_load_no_nvfp4_code():
    pytest.importorskip("uvicorn")  # praxis.serve.server needs the `serve` extra
    code = (
        "import sys\n"
        "import praxis.pipeline, praxis.serve.server\n"
        "loaded = [m for m in sys.modules if m.startswith("
        "('praxis.quantization', 'praxis.kernels.nvfp4', 'flashinfer'))]\n"
        "print(loaded)\n"
        "assert not loaded, loaded\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=env)


# The combine kernel of praxis/kernels/nvfp4.py runs on any CUDA GPU. The
# grouped GEMMs need a Blackwell GPU, and these tests do not call them.
cuda_kernels = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the NVFP4 Triton kernels need a CUDA GPU"
)
# The activation quantize kernel packs the FP4 codes with the e2m1x2 convert
# instruction, which exists from SM100 on: an older GPU cannot compile it.
sm100_kernels = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="the NVFP4 quantize kernel needs an SM100 or newer GPU",
)

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
KERNEL_SHAPES = [(8, 256, 2, 50), (16, 3072, 8, 300)]  # experts, width, top-k, tokens


def decode_codes(codes):
    magnitude = torch.tensor(FP4_VALUES, dtype=torch.float64, device=codes.device)
    value = magnitude[codes.long() & 7]
    return torch.where(codes.long() >= 8, -value, value)


def decode_fp4(q, k):
    return decode_codes(torch.stack([q & 0xF, q >> 4], -1).reshape(*q.shape[:-1], k))


def qat_activation_qdq(x, global_scale):
    """Activation rounding of the QAT forward for one expert, in float64."""
    from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked

    n, k = x.shape
    blocks = x.float().reshape(n, k // 16, 16)
    block_scale = (
        (blocks.abs().amax(-1) * (1.0 / 6.0) / global_scale)
        .clamp(2.0**-6, 448.0)
        .to(torch.float8_e4m3fn)
    )
    reciprocal = (1.0 / global_scale) / block_scale.float()
    scaled = (blocks * reciprocal.unsqueeze(-1)).clamp(-6.0, 6.0).reshape(n, k)
    value = decode_codes(f32_to_f4_unpacked(scaled))
    return (
        value * block_scale.double().repeat_interleave(16, -1) * global_scale.double()
    )


def dispatched_rows(experts, width, top_k, tokens):
    import torch.nn.functional as F

    from praxis.layers import _dispatch_rows

    device = torch.device("cuda")
    torch.manual_seed(0)
    x = (torch.randn(tokens, width) * 1.5).to(torch.bfloat16).to(device)
    logits = torch.randn(tokens, experts, device=device)
    logits[:, -1] = -1e9  # the last expert gets no rows
    top_idx = logits.topk(top_k, -1).indices.contiguous()
    gate = torch.softmax(torch.randn(tokens, top_k, device=device), -1).to(
        torch.bfloat16
    )
    order, rows, sorted_gate, offsets = _dispatch_rows(top_idx, gate, experts)
    return x, order, rows, sorted_gate, F.pad(offsets, (1, 0))


@sm100_kernels
@pytest.mark.parametrize("shape", KERNEL_SHAPES)
def test_activation_quantize_kernel_equals_the_qat_rounding(shape):
    pytest.importorskip("torchao")
    from torchao.prototype.mx_formats.utils import from_blocked

    from praxis.kernels.nvfp4 import quantize_activations

    experts, width, top_k, tokens = shape
    x, _, rows, _, csr = dispatched_rows(*shape)
    q, sf, global_scales, padded = quantize_activations(x, csr, row_indices=rows)
    bounds, padded_bounds = csr.tolist(), padded.tolist()
    flat = sf.reshape(-1)
    for e in range(experts):
        lo, hi, start = bounds[e], bounds[e + 1], padded_bounds[e]
        if hi == lo:
            continue
        m = hi - lo
        tile = (m + 127) // 128 * 128
        base = (start + e * 127) // 128 * 128
        scales = from_blocked(
            flat[base * (width // 16) : (base + tile) * (width // 16)],
            tile,
            width // 16,
        )[:m]
        decoded = (
            decode_fp4(q[start : start + m], width)
            * scales.contiguous()
            .view(torch.float8_e4m3fn)
            .double()
            .repeat_interleave(16, -1)
            * global_scales[e].double()
        )
        assert torch.equal(
            decoded, qat_activation_qdq(x[rows[lo:hi]], global_scales[e])
        )


@sm100_kernels
@pytest.mark.parametrize("projection", ["gate_up", "down"])
@pytest.mark.parametrize("shape", KERNEL_SHAPES)
def test_rescale_kernel_equals_torch(shape, projection):
    """gate/up rescales the padded rows in place; down also drops the padding."""
    from praxis.kernels.nvfp4 import quantize_activations, rescale

    experts, width, top_k, tokens = shape
    x, _, rows, _, csr = dispatched_rows(*shape)
    _, _, act_globals, padded = quantize_activations(x, csr, row_indices=rows)
    weight_globals = torch.rand(experts, device=x.device) * 1e-3
    padded_rows = int(padded[-1].item())
    columns = 2 * width if projection == "gate_up" else width
    raw = torch.randn(padded_rows, columns, device=x.device).to(torch.bfloat16)
    multiplier = (act_globals * weight_globals).to(torch.bfloat16).float()
    if projection == "gate_up":
        out = rescale(raw, padded, padded, act_globals, weight_globals, padded_rows)
        bounds = padded
        source = torch.arange(padded_rows, device=x.device)
    else:
        out = rescale(raw, csr, padded, act_globals, weight_globals, rows.numel())
        bounds = csr
        index = torch.arange(rows.numel(), device=x.device)
        expert = torch.bucketize(index.int(), csr[1:], right=True).clamp_max(
            experts - 1
        )
        source = padded[expert] + index - csr[expert]
    expert_of_row = torch.bucketize(
        torch.arange(out.shape[0], device=x.device, dtype=torch.int32),
        bounds[1:],
        right=True,
    ).clamp_max(experts - 1)
    expected = (raw[source].float() * multiplier[expert_of_row][:, None]).to(
        torch.bfloat16
    )
    assert torch.equal(out, expected)


@cuda_kernels
@pytest.mark.parametrize("shape", KERNEL_SHAPES)
def test_combine_kernel_equals_the_bf16_reduction(shape):
    from praxis.kernels.nvfp4 import combine_rows
    from praxis.layers import _sum_rows_of_tokens

    experts, width, top_k, tokens = shape
    x, order, rows, sorted_gate, _ = dispatched_rows(*shape)
    values = torch.randn(rows.numel(), width, device=x.device).to(torch.bfloat16)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=x.device)
    scale = sorted_gate.float() * 2.5
    out = combine_rows(values, inverse, scale, tokens, top_k)
    expected = _sum_rows_of_tokens(
        values, inverse.view(tokens, top_k), scale[inverse].view(tokens, top_k)
    )
    assert torch.equal(out, expected.to(torch.bfloat16))
