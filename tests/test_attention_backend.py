"""CPU checks for Flash attention fallback on SM110 and SM120."""

from unittest.mock import Mock

import pytest
import torch

from praxis import layers


@pytest.mark.parametrize(
    ("limit", "initial", "expected"),
    [
        (64, (8, 256), (64, 1024)),
        (32, (8, 256), (32, 512)),
        (64, (256, 4096), (256, 4096)),
        (None, (8, 256), (8, 256)),
    ],
)
def test_flash_sets_compile_limits_before_compiling(
    monkeypatch, limit, initial, expected
):
    import torch._dynamo.config as dynamo_config

    from praxis import attention

    monkeypatch.setattr(attention, "FLASH_RECOMPILE_LIMIT", limit)
    calls = []

    def compile_attention(fn, **kwargs):
        calls.append(kwargs)
        assert (
            dynamo_config.recompile_limit,
            dynamo_config.accumulated_recompile_limit,
        ) == expected
        return fn

    monkeypatch.setattr(torch, "compile", compile_attention)
    with dynamo_config.patch(
        recompile_limit=initial[0], accumulated_recompile_limit=initial[1]
    ):
        attention.FlashFlex()
    assert calls == [{"dynamic": False, "options": {"triton.cudagraphs": False}}]


@pytest.mark.parametrize(
    ("requested", "device", "major", "expected"),
    [
        ("flash", "cuda:0", 12, "triton"),
        ("flash", "cuda:0", 9, "flash"),
        ("flash", "cuda:0", 10, "flash"),
        ("flash", "cuda:0", 11, "triton"),
        ("flash", "cpu", 12, "flash"),
        ("triton", "cuda:0", 12, "triton"),
        ("triton", "cpu", 12, "triton"),
    ],
)
def test_requested_backend_and_architecture_select_explicit_effective_backend(
    monkeypatch, requested, device, major, expected
):
    capability = Mock(return_value=(major, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    module = layers.Attention(32, 16, 2, requested)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(0.125)
    state = {k: v.clone() for k, v in module.state_dict().items()}
    metadata = module.backend_metadata(device)
    assert metadata == {
        "requested": requested,
        "effective": expected,
        "reason": "Flash block sparsity is unsupported for this architecture or block size"
        if expected != requested
        else None,
    }
    assert module.backend == requested
    assert all(torch.equal(state[k], v) for k, v in module.state_dict().items())
    if requested == "flash" and device.startswith("cuda"):
        capability.assert_called_once_with(torch.device(device))
    else:
        capability.assert_not_called()


def test_fallback_forward_uses_exact_existing_triton_call(monkeypatch):
    # Only backend selection is substituted on CPU. The same projection,
    # normalization, attention inputs and output math run in both modules.
    requested = layers.Attention(32, 16, 2, "flash").eval()
    explicit = layers.Attention(32, 16, 2, "triton").eval()
    with torch.no_grad():
        for parameter in requested.parameters():
            parameter.fill_(0.125)
    explicit.load_state_dict(requested.state_dict())
    monkeypatch.setattr(
        requested, "effective_backend", lambda device, block_mask=None: "triton"
    )
    calls = []

    def attention(q, k, v, **kwargs):
        calls.append((q.clone(), k.clone(), v.clone(), kwargs))
        return v.clone()

    monkeypatch.setattr(layers, "_compiled_flex_attention", attention)
    x = torch.randn(1, 7, 32)
    with torch.no_grad():
        actual, reference = requested(x, None, None), explicit(x, None, None)
    assert torch.equal(actual, reference)
    assert len(calls) == 2
    assert calls[0][3] == calls[1][3]
    assert calls[0][3]["kernel_options"] is None
    assert all(
        torch.equal(a, b) for a, b in zip(calls[0][:3], calls[1][:3], strict=True)
    )


def test_unchanged_flash_forward_keeps_optional_backend_dispatch(monkeypatch):
    from praxis import attention

    module = layers.Attention(32, 16, 2, "flash").eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(0.125)
    calls = []

    def optional(backend, q, k, v, **kwargs):
        calls.append((backend, kwargs))
        return v

    monkeypatch.setattr(attention, "optional_attention", optional)
    with torch.no_grad():
        module(torch.randn(1, 7, 32), None, None)
    assert calls[0][0] == "flash"
    assert "kernel_options" not in calls[0][1]


@pytest.mark.parametrize(
    ("device", "capability", "expected"),
    [
        ("cpu", (10, 3), 128),
        ("cuda:0", (9, 0), 128),
        ("cuda:0", (10, 0), 256),
        ("cuda:0", (10, 3), 256),
        ("cuda:0", (12, 0), 128),
    ],
)
def test_flash_mask_tiles_preserve_hopper_and_fit_blackwell(
    monkeypatch, device, capability, expected
):
    from praxis.attention import flash_mask_block_size

    probe = Mock(return_value=capability)
    monkeypatch.setattr(torch.cuda, "get_device_capability", probe)
    assert flash_mask_block_size(device) == expected
    if device == "cpu":
        probe.assert_not_called()


@pytest.mark.parametrize(
    ("capability", "expected"), [((11, 0), 128), ((10, 0), 256), ((9, 0), 128)]
)
def test_sm110_flash_mask_256_off_changes_only_thor(monkeypatch, capability, expected):
    from praxis import attention

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    assert attention.flash_mask_block_size("cuda:0") == (
        256 if capability[0] in (10, 11) else 128
    )
    monkeypatch.setattr(attention, "SM110_FLASH_MASK_256", False)
    assert attention.flash_mask_block_size("cuda:0") == expected


def test_mask_cache_uses_effective_block_size_in_its_key(monkeypatch):
    from praxis import backbone

    architecture = [9]
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _: (architecture[0], 0)
    )
    built = []

    def mask(slots, chunk_size, clean_past, block_size, device, full_blocks):
        result = object()
        built.append((block_size, result))
        return result

    monkeypatch.setattr(backbone, "teacher_forcing_mask", mask)
    cache = backbone._TeacherForcingMaskCache(128)
    args = ([("global", 1, 7), ("noised", 6, 43)], 6, True, torch.device("cuda"), True)
    hopper = cache.get(*args)
    architecture[0] = 10
    blackwell = cache.get(*args)
    assert hopper is not blackwell
    assert cache.get(*args) is blackwell
    assert [size for size, _ in built] == [128, 256]


@pytest.mark.parametrize(
    ("block_size", "expected"), [(128, "triton"), (256, "flash"), (512, "flash")]
)
def test_thor_flash_requires_compatible_sparse_tiles(monkeypatch, block_size, expected):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (11, 0))
    module = layers.Attention(32, 16, 2, "flash")
    mask = SimpleNamespace(BLOCK_SIZE=(block_size, block_size))
    assert module.effective_backend("cuda:0", mask) == expected
