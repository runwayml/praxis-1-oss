"""CPU checks for Flash attention's explicit SM120 Triton fallback."""

import copy
from unittest.mock import Mock

import pytest
import torch

from praxis import layers


@pytest.mark.parametrize(
    ("requested", "device", "major", "expected"),
    [
        ("flash", "cuda:0", 12, "triton"),
        ("flash", "cuda:0", 9, "flash"),
        ("flash", "cuda:0", 10, "flash"),
        ("flash", "cuda:0", 11, "flash"),
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
    original_options = copy.deepcopy(module._kernel_options)
    metadata = module.backend_metadata(device)
    assert metadata == {
        "requested": requested,
        "effective": expected,
        "reason": "Flash block sparsity is unsupported on SM120"
        if expected != requested
        else None,
    }
    assert module.backend == requested
    assert module._kernel_options == original_options
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
    monkeypatch.setattr(requested, "effective_backend", lambda device: "triton")
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


def test_unchanged_flash_forward_keeps_original_options(monkeypatch):
    module = layers.Attention(32, 16, 2, "flash").eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(0.125)
    calls = []

    def attention(q, k, v, **kwargs):
        calls.append(kwargs)
        return v

    monkeypatch.setattr(layers, "_compiled_flex_attention", attention)
    with torch.no_grad():
        module(torch.randn(1, 7, 32), None, None)
    assert calls[0]["kernel_options"] is module._kernel_options
    assert calls[0]["kernel_options"] == {"BACKEND": "FLASH"}
