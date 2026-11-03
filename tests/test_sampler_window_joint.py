"""The window joint sampler of the checkpoint (CPU only).

The time grid and the noise are pinned to values of the original sampler.
The Euler loop runs on a stub backbone and a stub expert, and must give the
same bits as the reference loop written out below.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import cast

import pytest
import torch
from torch import Tensor, nn

import praxis.model as model_mod
from praxis.action_expert import ActionExpert
from praxis.backbone import VideoBackbone
from praxis.model import PraxisPolicy, SamplerSettings
from praxis.pipeline import PraxisPipeline, Session
from praxis.sampler import (
    ACTION_SALT,
    VIDEO_SALT,
    guidance_combine,
    seeded_normal,
    window_time_grid,
)
from praxis.single_arm import SingleArmChunkInputs


def _sha(t: Tensor) -> str:
    raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Time grid and noise (values of the original sampler)
# ---------------------------------------------------------------------------

GRIDS: dict[tuple[int, float], list[float]] = {
    (1, 1.0): [1.0, 0.0],
    (2, 1.0): [1.0, 0.5, 0.0],
    (2, 8.0): [1.0, 0.8888888888888888, 0.0],
    (3, 8.0): [1.0, 0.9411764705882354, 0.8, 0.0],
    (4, 3.0): [1.0, 0.9, 0.75, 0.5, 0.0],
    (5, 8.0): [
        1.0,
        0.9696969696969697,
        0.923076923076923,
        0.8421052631578947,
        0.6666666666666666,
        0.0,
    ],
}


@pytest.mark.parametrize(("steps", "shift"), list(GRIDS))
def test_time_grid_matches_the_original(steps: int, shift: float) -> None:
    grid = window_time_grid(steps, shift)
    assert grid == GRIDS[(steps, shift)]
    assert all(type(t) is float for t in grid)


def test_time_grid_hash_matches_the_original_trace() -> None:
    """The float64 grid of 2 steps at shift 8 has the hash of the original trace."""
    grid = torch.tensor(window_time_grid(2, 8.0), dtype=torch.float64)
    assert _sha(grid) == (
        "a6401df10d39bf354afd5b3916ece65170b83b13d3d7350948c4eb68f56faa6b"
    )
    with pytest.raises(ValueError, match="num_steps"):
        window_time_grid(0)


NOISE_VALUES: dict[tuple[int, int, tuple[int, ...]], list[float]] = {
    (0, 0, (2, 3)): [
        1.5409960746765137,
        -0.293428897857666,
        -2.1787893772125244,
        0.5684312582015991,
        -1.0845223665237427,
        -1.3985954523086548,
    ],
    (0, 1, (2, 3)): [
        0.6613521575927734,
        0.266924113035202,
        0.06167725846171379,
        0.6213173270225525,
        -0.4519059658050537,
        -0.16613022983074188,
    ],
    (7, 0, (4,)): [
        -0.32943111658096313,
        -0.4009939730167389,
        1.6553479433059692,
        -0.740679919719696,
    ],
    (123456789, 1, (3,)): [
        0.09768254309892654,
        -1.4723442792892456,
        0.7480131387710571,
    ],
    (2**40 + 5, 0, (3,)): [
        -0.004918505437672138,
        -0.1052783727645874,
        0.5255449414253235,
    ],
}

# Hashes of full-size draws: the video and action noise of the original
# traces at depth 0 (seed 0) and depth 1 (seed 1).
NOISE_SHA: dict[tuple[int, int, tuple[int, ...]], str] = {
    (0, 0, (1, 192, 6, 8, 26)): (
        "fdd278862c193980e4bf3811fff0f5afef9a968143b139e279ff84f071eb52d6"
    ),
    (0, 1, (6, 40)): (
        "b47e44eb787c9eeb337fcfbe6de890fb5b33a44bee94a07c478cc5a1b1a806f9"
    ),
    (1, 0, (1, 192, 12, 8, 26)): (
        "df0225d7070ebe28bc8390a69fd674976cf32f240257f6c7309d04c4d117c93f"
    ),
    (1, 1, (12, 40)): (
        "c783d55a5232520ff96ab4f752a55b57bcbeaad518832e5e99e8efd9954637ff"
    ),
}


@pytest.mark.parametrize("key", list(NOISE_VALUES))
def test_seeded_normal_values(key: tuple[int, int, tuple[int, ...]]) -> None:
    seed, salt, shape = key
    got = seeded_normal(shape, seed, salt)
    assert got.dtype == torch.float32 and got.device.type == "cpu"
    assert got.reshape(-1).tolist() == NOISE_VALUES[key]


@pytest.mark.parametrize("key", list(NOISE_SHA))
def test_seeded_normal_hashes(key: tuple[int, int, tuple[int, ...]]) -> None:
    seed, salt, shape = key
    assert _sha(seeded_normal(shape, seed, salt)) == NOISE_SHA[key]


def test_salts() -> None:
    assert (VIDEO_SALT, ACTION_SALT) == (0, 1)
    # A seed past 2**63 wraps into the positive int64 range.
    assert torch.equal(seeded_normal((3,), 2**63, 0), seeded_normal((3,), 0, 0))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_window_joint_settings() -> None:
    SamplerSettings(layout="window_joint", num_steps=2, t_shift=8.0)
    SamplerSettings(num_steps=2, action_steps=2)
    with pytest.raises(ValueError, match="euler"):
        SamplerSettings(layout="window_joint", solver="heun")
    with pytest.raises(ValueError, match="sub-steps"):
        SamplerSettings(layout="window_joint", num_steps=2, action_steps=4)
    for layout in ("chunk", "full"):
        with pytest.raises(ValueError, match="layout"):
            SamplerSettings(layout=layout)


def test_sampler_defaults_are_the_recommended_settings() -> None:
    """The defaults are the recommended settings (configs/*_droid.yaml)."""
    s = SamplerSettings()
    assert (s.num_steps, s.t_shift, s.solver) == (2, 12.5, "euler")
    assert s.action_steps is None
    assert (s.layout, s.cfg_scale, s.cfg_video_scale) == ("window_joint", 1.0, 4.0)
    assert s.cfg_batched is False and s.guided


# ---------------------------------------------------------------------------
# The Euler loop on stubs
# ---------------------------------------------------------------------------

C, H, W, LAT, A_DIM, P_DIM = 192, 2, 3, 6, 40, 10
TPF = H * W
MD = torch.bfloat16


@dataclass
class StubTaps:
    taps: list[Tensor]
    embeddings: Tensor
    rope_ids: Tensor
    global_spans: dict[str, tuple[int, int]]
    noised_span: tuple[int, int]
    clean_span: tuple[int, int] | None
    frames_per_chunk: int = LAT
    tokens_per_frame: int = TPF


@dataclass
class StubOutput:
    velocity: Tensor | None
    taps: StubTaps


@dataclass
class Calls:
    trunk: list[dict] = field(default_factory=list)
    expert: list[dict] = field(default_factory=list)


def _feature(text: Tensor, noised: Tensor, clean: Tensor | None, t: Tensor) -> Tensor:
    """A deterministic summary of the trunk inputs, in the model dtype."""
    s = text.float().sum() * 1e-3 + t.float().sum()
    if clean is not None:
        s = s + clean.float().mean()
    return (noised.float() * 0.5 + s).to(MD)


class StubBackbone(nn.Module):
    def __init__(self, calls: Calls) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.registers = nn.Parameter(torch.zeros(1, 4, 8, dtype=MD))
        self.calls = calls

    def forward_chunk_causal(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        decode: bool = True,
    ) -> StubOutput:
        self.calls.trunk.append(
            {
                "text": text,
                "anchor": anchor,
                "clean": clean,
                "noised": noised,
                "t": timesteps,
                "decode": decode,
            }
        )
        kk = timesteps.shape[1]
        feat = _feature(text, noised, clean, timesteps)
        velocity = (feat.float() - 0.25 * noised.float()).to(MD) if decode else None
        n_noised = kk * LAT * TPF
        n_clean = (kk - 1) * LAT * TPF
        n_text = text.shape[1]
        seq = 4 + TPF + n_text + n_noised + n_clean
        emb = torch.full((1, seq, 8), float(feat.float().mean()), dtype=MD)
        taps = StubTaps(
            taps=[emb],
            embeddings=emb,
            rope_ids=torch.arange(seq)[None],
            global_spans={
                "register": (0, 4),
                "anchor": (4, TPF),
                "text": (4 + TPF, n_text),
            },
            noised_span=(4 + TPF + n_text, n_noised),
            clean_span=(4 + TPF + n_text + n_noised, n_clean) if kk > 1 else None,
        )
        return StubOutput(velocity, taps)


def _stub_expert(calls: Calls):
    def run(expert, taps, embeddings, rope_ids, **kw):
        calls.expert.append({"rope_ids": rope_ids, **kw})
        dtype = embeddings.dtype
        x = kw["noised_actions"].to(dtype)
        p = kw["proprio_per_chunk"].to(dtype).repeat_interleave(LAT, dim=1)
        t = kw["action_t_per_chunk"].repeat_interleave(LAT, dim=1)
        out = (
            x.float() * 0.75
            - embeddings.float().mean()
            + p.float().mean(-1, True) * t[..., None]
        )
        return out.to(dtype)

    return run


def _policy(calls: Calls) -> PraxisPolicy:
    config = {
        "chunking": {"latents_per_chunk": LAT, "n_chunks": 4},
        "action_expert": {"action_patch_dim": A_DIM},
    }
    backbone = cast(VideoBackbone, StubBackbone(calls))
    return PraxisPolicy(backbone, cast(ActionExpert, nn.Module()), config)


def _inputs(depth: int) -> dict:
    g = torch.Generator().manual_seed(10 + depth)
    return {
        "text": torch.randn(1, 5, 2048, generator=g).to(MD),
        "text_uncond": torch.randn(1, 1, 2048, generator=g).to(MD),
        "anchor": torch.randn(1, C, 1, H, W, generator=g).to(MD),
        "cleans": [
            torch.randn(1, C, LAT, H, W, generator=g).to(MD) for _ in range(depth)
        ],
        "proprio_all": torch.randn(1, depth + 1, P_DIM, generator=g),
    }


def _reference_loop(
    inp: dict, seed: int, steps: int, shift: float, cfg: float, cfg_video: float
) -> tuple[Tensor, Tensor]:
    """The reference window sampler written out with the stub velocity."""
    kk = len(inp["cleans"]) + 1
    clean = torch.cat(inp["cleans"], 2) if inp["cleans"] else None
    sink = Calls()
    backbone, expert = StubBackbone(sink), _stub_expert(sink)

    def velocity(text: Tensor, xv: Tensor, xa: Tensor, t_cur: float) -> tuple:
        t = torch.full((1, kk), float(t_cur), dtype=MD)
        out = backbone.forward_chunk_causal(text, inp["anchor"], clean, xv.to(MD), t)
        tp = out.taps
        a = expert(
            None,
            tp.taps,
            tp.embeddings,
            tp.rope_ids,
            noised_actions=xa[None],
            action_t_per_chunk=torch.full((kk,), float(t_cur))[None],
            proprio_per_chunk=inp["proprio_all"],
        )
        assert out.velocity is not None
        return out.velocity.float(), a.float()[0]

    xv = seeded_normal((1, C, kk * LAT, H, W), seed, 0)
    xa = seeded_normal((kk * LAT, A_DIM), seed, 1)
    ts = window_time_grid(steps, shift)
    guided = cfg != 1.0 or cfg_video != 1.0
    for step in range(steps):
        t_cur, t_next = ts[step], ts[step + 1]
        v, a = velocity(inp["text"], xv, xa, t_cur)
        if guided:
            v_u, a_u = velocity(inp["text_uncond"], xv, xa, t_cur)
            v = guidance_combine(v, v_u, cfg_video)
            a = guidance_combine(a, a_u, cfg)
        d = t_next - t_cur
        xv = xv + d * v
        xa = xa + d * a
    return xv[:, :, (kk - 1) * LAT :], xa[(kk - 1) * LAT :][None]


CASES = [
    # depth, seed, steps, shift, cfg_scale, cfg_video_scale
    (0, 0, 2, 8.0, 1.0, 1.0),
    (1, 3, 2, 8.0, 1.0, 1.0),
    (2, 5, 3, 8.0, 1.0, 1.0),
    (3, 1, 2, 8.0, 1.0, 1.0),
    (0, 0, 2, 1.0, 3.0, 1.0),
    (2, 7, 2, 1.0, 3.0, 2.0),
]


@pytest.mark.parametrize(("depth", "seed", "steps", "shift", "cfg", "cfg_v"), CASES)
def test_window_joint_equals_the_reference_loop(
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
    seed: int,
    steps: int,
    shift: float,
    cfg: float,
    cfg_v: float,
) -> None:
    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    policy = _policy(calls)
    inp = _inputs(depth)
    settings = SamplerSettings(
        layout="window_joint",
        num_steps=steps,
        t_shift=shift,
        cfg_scale=cfg,
        cfg_video_scale=cfg_v,
    )
    video, actions = policy.sample_window_joint(seed=seed, settings=settings, **inp)
    want_v, want_a = _reference_loop(inp, seed, steps, shift, cfg, cfg_v)
    assert video.shape == (1, C, LAT, H, W) and actions.shape == (1, LAT, A_DIM)
    assert video.dtype == torch.float32 and actions.dtype == torch.float32
    assert torch.equal(video, want_v)
    assert torch.equal(actions, want_a)

    kk = depth + 1
    guided = cfg != 1.0 or cfg_v != 1.0
    assert len(calls.trunk) == steps * (2 if guided else 1)
    assert len(calls.expert) == len(calls.trunk)
    ts = window_time_grid(steps, shift)
    for i, call in enumerate(calls.trunk):
        step, uncond = (i // 2, i % 2) if guided else (i, 0)
        assert call["text"] is (inp["text_uncond"] if uncond else inp["text"])
        assert call["t"].dtype == MD and call["t"].shape == (1, kk)
        assert torch.equal(call["t"], torch.full((1, kk), ts[step], dtype=MD))
        assert call["noised"].dtype == MD
        assert call["noised"].shape == (1, C, kk * LAT, H, W)
        if depth:
            assert torch.equal(call["clean"], torch.cat(inp["cleans"], 2))
        else:
            assert call["clean"] is None
        # The unguided video is decoded only when its scale needs it.
        assert call["decode"] is (not uncond or cfg_v != 1.0)
    for i, call in enumerate(calls.expert):
        step = i // 2 if guided else i
        assert call["n_chunks"] == kk and call["frames_per_chunk"] == LAT
        assert call["noised_actions"].dtype == torch.float32
        assert call["noised_actions"].shape == (1, kk * LAT, A_DIM)
        assert torch.equal(call["action_t_per_chunk"], torch.full((1, kk), ts[step]))
        assert call["action_t_per_chunk"].dtype == torch.float32
        assert torch.equal(call["proprio_per_chunk"], inp["proprio_all"])
    # Both passes of a step read the same states.
    if guided:
        for j in range(0, len(calls.trunk), 2):
            assert torch.equal(calls.trunk[j]["noised"], calls.trunk[j + 1]["noised"])
            assert torch.equal(
                calls.expert[j]["noised_actions"], calls.expert[j + 1]["noised_actions"]
            )


def test_window_joint_trace_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe gives the names of the original trace, in its order."""
    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    policy = _policy(calls)
    names: list[str] = []
    settings = SamplerSettings(layout="window_joint", num_steps=2, cfg_scale=3.0)
    policy.sample_window_joint(
        seed=0, settings=settings, trace=lambda n, t: names.append(n), **_inputs(1)
    )
    one_pass = [
        "velocity.in.xt_v",
        "velocity.in.xt_a",
        "velocity.in.t_cur",
        "velocity.out.video",
        "velocity.out.action",
    ]
    combine = [
        f"cfg_combine.{kind}.{part}"
        for kind in ("video", "action")
        for part in ("cond", "uncond", "out")
    ]
    step = ["pass.marker", *one_pass, "pass.marker", *one_pass, *combine]
    assert names == [
        "noise.salt0",
        "noise.salt1",
        "sampler.time_grid",
        *step,
        *step,
        "out.video",
        "out.action",
        "out.video_last",
        "out.action_last",
    ]


def test_window_joint_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    policy = _policy(calls)
    inp = _inputs(1)
    guided = SamplerSettings(layout="window_joint", cfg_scale=3.0)
    with pytest.raises(ValueError, match="blank caption"):
        policy.sample_window_joint(
            inp["text"],
            None,
            inp["anchor"],
            inp["cleans"],
            inp["proprio_all"],
            seed=0,
            settings=guided,
        )
    with pytest.raises(ValueError, match="proprio_all"):
        policy.sample_window_joint(
            **{**inp, "proprio_all": inp["proprio_all"][:, :1]},
            seed=0,
            settings=SamplerSettings(layout="window_joint"),
        )
    with pytest.raises(ValueError, match="out of range"):
        policy.sample_window_joint(
            **_inputs(4), seed=0, settings=SamplerSettings(layout="window_joint")
        )


# ---------------------------------------------------------------------------
# The session call
# ---------------------------------------------------------------------------


@dataclass
class StubInputs:
    prompt: str
    anchor: Tensor
    cleans: list[Tensor]
    proprio: Tensor
    proprio_all: Tensor
    chunk_index: int


class StubPipe:
    def __init__(self) -> None:
        self.policy = self
        self.calls: list[dict] = []
        self.texts: list[str] = []

    def encode_text(self, prompt: str) -> Tensor:
        self.texts.append(prompt)
        return torch.full((1, 2, 4), float(len(prompt)))

    def encode_uncond_text(self) -> Tensor:
        self.texts.append("<uncond>")
        return torch.zeros(1, 1, 4)

    def sample_window_joint(self, **kw: object) -> tuple[Tensor, Tensor]:
        self.calls.append(kw)
        return torch.zeros(1), torch.ones(1)


@pytest.mark.parametrize("cfg", [1.0, 3.0])
def test_session_samples_the_window(cfg: float) -> None:
    pipe = StubPipe()
    session = object.__new__(Session)
    session.pipe = cast(PraxisPipeline, pipe)
    session.sampler = SamplerSettings(
        layout="window_joint", cfg_scale=cfg, cfg_video_scale=1.0
    )
    session.seed = 100
    inputs = StubInputs(
        prompt="Put the cube on the plate.",
        anchor=torch.zeros(1),
        cleans=[torch.zeros(1)],
        proprio=torch.zeros(1, 1, 10),
        proprio_all=torch.ones(1, 2, 10),
        chunk_index=7,
    )
    _video, _actions, text = session._sample(cast(SingleArmChunkInputs, inputs))
    (call,) = pipe.calls
    assert call["seed"] == 107
    assert call["proprio_all"] is inputs.proprio_all
    assert call["cleans"] is inputs.cleans and call["anchor"] is inputs.anchor
    assert call["settings"] is session.sampler
    if cfg == 1.0:
        assert call["text_uncond"] is None and pipe.texts == [inputs.prompt]
    else:
        assert pipe.texts == [inputs.prompt, "<uncond>"]
        assert torch.equal(call["text_uncond"], torch.zeros(1, 1, 4))
    assert torch.equal(text, call["text"])
