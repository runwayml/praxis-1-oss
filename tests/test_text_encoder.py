"""Tests of the fixed launch settings of the text encoder kernels (CPU only)."""

import sys
import types

import pytest
import torch

from praxis import text_encoder


def test_encode_accepts_new_processor_metadata() -> None:
    encoder = text_encoder.TextEncoder.__new__(text_encoder.TextEncoder)
    encoder.max_batch_size = 2
    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "mm_token_type_ids": torch.zeros((2, 3), dtype=torch.long),
    }
    seen = []

    def tokenize(prompts):
        seen.extend(prompts)
        return batch

    def encode_tokens(input_ids, attention_mask):
        assert input_ids is batch["input_ids"]
        assert attention_mask is batch["attention_mask"]
        return input_ids.unsqueeze(-1).float()

    encoder.tokenize = tokenize
    encoder._encode_tokens = encode_tokens
    result = encoder.encode(["pick", ""])
    assert seen == ["pick", " "]
    assert [x.shape for x in result] == [(1, 2, 1), (1, 1, 1)]
    assert result[0].flatten().tolist() == [1, 2]


def test_pin_fla_kernels_keeps_one_launch_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")

    @triton.autotune(
        configs=[
            triton.Config({"BT": bt}, num_warps=w) for bt in (8, 16) for w in (2, 4)
        ],
        key=["D"],
    )
    @triton.jit
    def kernel(x, D, BT: tl.constexpr):
        pass

    module = types.ModuleType("fake_kernels")
    vars(module)["kernel"] = kernel
    monkeypatch.setitem(sys.modules, "fake_kernels", module)
    monkeypatch.setattr(
        text_encoder,
        "FLA_KERNEL_CONFIGS",
        {
            "fake_kernels.kernel": ({"BT": 16}, 4, 3),
            "fake_kernels.absent": ({}, 1, 3),
            "absent_package.kernel": ({}, 1, 3),
        },
    )
    pinned = text_encoder.pin_fla_kernels()
    assert list(pinned) == ["kernel"]
    assert [(c.kwargs, c.num_warps, c.num_stages) for c in kernel.configs] == [
        ({"BT": 16}, 4, 3)
    ]
    # A second call keeps the same setting.
    assert text_encoder.pin_fla_kernels() == pinned


_STATE_KERNEL = (
    "fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
)


def _install_state_kernel(monkeypatch, settings, *, path=_STATE_KERNEL):
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")
    configs = [
        triton.Config({"BV": bv}, num_warps=warps, num_stages=stages)
        for bv, warps, stages in settings
    ]

    @triton.autotune(configs=configs, key=["D"])
    @triton.jit
    def kernel(x, D, BV: tl.constexpr):
        pass

    module_path, name = path.rsplit(".", 1)
    module = types.ModuleType(module_path)
    vars(module)[name] = kernel
    monkeypatch.setitem(sys.modules, module_path, module)
    monkeypatch.setattr(text_encoder, "FLA_KERNEL_CONFIGS", {path: ({"BV": 32}, 4, 3)})
    return kernel, configs, name


def test_state_kernel_prefers_original_three_stages(monkeypatch, caplog):
    kernel, configs, name = _install_state_kernel(monkeypatch, [(32, 4, 2), (32, 4, 3)])
    with caplog.at_level("INFO", logger="praxis.text_encoder"):
        pinned = text_encoder.pin_fla_kernels()
    assert kernel.configs == [configs[1]]
    assert kernel.configs[0] is configs[1]
    assert pinned == {name: str(configs[1])}
    assert "fallback" not in caplog.text


def test_state_kernel_fixes_supported_two_stages_idempotently(monkeypatch, caplog):
    kernel, configs, name = _install_state_kernel(
        monkeypatch, [(32, 4, 1), (64, 4, 2), (32, 2, 2), (32, 4, 2)]
    )
    with caplog.at_level("INFO", logger="praxis.text_encoder"):
        pinned = text_encoder.pin_fla_kernels()
        assert text_encoder.pin_fla_kernels() == pinned
    assert kernel.configs == [configs[3]]
    assert kernel.configs[0] is configs[3]
    assert pinned == {name: str(configs[3])}
    assert "using fixed supported fallback" in caplog.text


@pytest.mark.parametrize("settings", [[(32, 4, 1)], [(64, 4, 2)], [(32, 2, 2)]])
def test_state_kernel_does_not_invent_unsupported_setting(
    monkeypatch, caplog, settings
):
    kernel, configs, _name = _install_state_kernel(monkeypatch, settings)
    assert text_encoder.pin_fla_kernels() == {}
    assert kernel.configs is configs
    assert "no launch setting" in caplog.text


def test_two_stage_fallback_is_not_used_for_other_kernels(monkeypatch):
    kernel, configs, _name = _install_state_kernel(
        monkeypatch, [(32, 4, 2)], path="fake_kernels.other_state_kernel"
    )
    assert text_encoder.pin_fla_kernels() == {}
    assert kernel.configs is configs


def test_two_stage_fallback_requires_the_original_requested_setting(monkeypatch):
    kernel, configs, _name = _install_state_kernel(monkeypatch, [(32, 4, 2)])
    monkeypatch.setattr(
        text_encoder, "FLA_KERNEL_CONFIGS", {_STATE_KERNEL: ({"BV": 32}, 4, 4)}
    )
    assert text_encoder.pin_fla_kernels() == {}
    assert kernel.configs is configs
