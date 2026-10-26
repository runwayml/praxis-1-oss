"""Tests of the fixed launch settings of the text encoder kernels (CPU only)."""

import sys
import types

import pytest

from praxis import text_encoder


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
