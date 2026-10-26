"""Unit tests for the sampler grids and checks (CPU only)."""

import pytest
import torch

from praxis.sampler import (
    action_substep_grid,
    check_guidance,
    guidance_combine,
    sampler_num_evals,
    sampler_timesteps,
)


def test_timestep_grid() -> None:
    assert torch.equal(sampler_timesteps(2), torch.tensor([1.0, 0.5, 0.0]))
    t = sampler_timesteps(4, t_shift=3.0)
    assert t[0] == 1.0 and t[-1] == 0.0 and bool((t[1:] < t[:-1]).all())


def test_substep_grid_keeps_trunk_points() -> None:
    m, fine = action_substep_grid(2, 6)
    assert m == 3
    assert torch.allclose(fine[::m], sampler_timesteps(2))
    with pytest.raises(ValueError, match="multiple"):
        action_substep_grid(2, 3)


def test_num_evals() -> None:
    assert sampler_num_evals(2) == 2
    assert sampler_num_evals(3, "heun") == 5
    with pytest.raises(ValueError, match="solver"):
        sampler_num_evals(2, "rk4")


def test_guidance() -> None:
    assert not check_guidance(1.0, 1.0) and check_guidance(3.0, 1.0)
    with pytest.raises(ValueError, match="cfg_scale"):
        check_guidance(-1.0, 1.0)
    v_c, v_u = torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0])
    assert guidance_combine(v_c, v_u, 1.0) is v_c
    assert torch.equal(guidance_combine(v_c, v_u, 3.0), torch.tensor([3.0, 4.0]))
