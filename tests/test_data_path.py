"""The data path of the single-arm checkpoint (CPU only, no weights).

The pinned values come from the original implementation: its tile canvas,
its action and proprio codec on the stats of the checkpoint, and its pose
frame conversion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from praxis import single_arm as sa
from praxis.robot_io import (
    NormStats,
    aspect_crop_box,
    denormalize_rows,
    pose8_to_10d,
    proprio_normalize,
    q99_normalize,
    unit_quat_wxyz_to_matrix,
)

# The normalization block of the checkpoint config.
Q01 = [-0.1485469937324524, -0.19603200256824493, -0.2031259983778]
Q01 += [0.6827049851417542, -0.5835009813308716, -0.3863309919834137]
Q01 += [-0.5673199892044067, 0.6868450045585632, -0.39503800868988037, 0.0]
Q99 = [0.18577100336551666, 0.19784800708293915, 0.19871099293231964, 1.0]
Q99 += [0.5795639753341675, 0.4474309980869293, 0.5877500176429749, 1.0]
Q99 += [0.39563798904418945, 1.0]
P_MEAN = [0.5195770263671875, -0.006078170146793127, 0.3168239891529083]
P_MEAN += [0.7413139939308167, -0.04902319982647896, 0.08595360070466995]
P_MEAN += [-0.06293799728155136, -0.7607250213623047, 0.008901899680495262]
P_MEAN += [0.39361199736595154]
P_STD = [0.1195869967341423, 0.17830400168895721, 0.1687529981136322]
P_STD += [0.32760199904441833, 0.4835300147533417, 0.3154970109462738]
P_STD += [0.4926550090312958, 0.33570900559425354, 0.24869099259376526]
P_STD += [0.4135879874229431]
Q99_DIMS = np.array([0, 1, 2, 9])

CHECKPOINT_CONFIG = {
    "video": {
        "height": 256,
        "width": 832,
        "views": ["exterior", "wrist"],
        "canvas": "tiles",
        "tile_height": 256,
        "tile_width": 416,
        "aspect_tol": 0.03,
    },
}


def _stats() -> NormStats:
    return NormStats(
        action_q01=np.asarray(Q01, dtype=np.float32),
        action_q99=np.asarray(Q99, dtype=np.float32),
        proprio_mean=np.asarray(P_MEAN, dtype=np.float32),
        proprio_std=np.asarray(P_STD, dtype=np.float32),
    )


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _f32(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tile canvas
# ---------------------------------------------------------------------------

# SHA-256 of the uint8 canvas bytes (1, 3, 2, 256, 832) and a few pixels
# (channel, frame, row, column) of the canvas of the original implementation.
CANVAS_REF = {
    (720, 1280): (
        "82ab2a59e5177cf4ac791bd44b514b8ba3c4bc1fba0a5ad76dec4cc808bbccca",
        {(0, 0, 0, 0): 90, (1, 1, 100, 200): 80, (2, 0, 255, 415): 209},
        {(0, 1, 17, 416): 38, (2, 1, 200, 831): 143},
    ),
    (288, 512): (
        "8a6e0391dede7681e5c389e7a1e095cb1676678cfcf6c6e88cc21b932de49d54",
        {(0, 0, 0, 0): 198, (1, 1, 100, 200): 125, (2, 0, 255, 415): 100},
        {(0, 1, 17, 416): 123, (2, 1, 200, 831): 121},
    ),
}


@pytest.mark.parametrize("size", sorted(CANVAS_REF))
def test_tile_clip_equals_the_original_canvas(size: tuple[int, int]) -> None:
    h, w = size
    rng = np.random.default_rng(0)
    ext = rng.integers(0, 256, size=(2, h, w, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(2, h, w, 3), dtype=np.uint8)
    clip = sa.tile_clip(ext, wrist)
    assert clip.shape == (1, 3, 2, 256, 832)
    assert clip.dtype == torch.float32
    u8 = clip.to(torch.uint8)
    assert torch.equal(u8.float(), clip)
    sha, ext_px, wrist_px = CANVAS_REF[size]
    assert _sha(u8.numpy()) == sha
    for (c, t, y, x), v in {**ext_px, **wrist_px}.items():
        assert float(clip[0, c, t, y, x]) == v


# SHA-256 of the uint8 canvas bytes (1, 3, 2, 224, 832) of the praxis-1-droid training
# data path and serve: the fit-416 rule (a centre crop to the native 720x1280
# aspect, an antialiased bilinear resize to 234x416 and a centre trim to
# 224x416), computed with the original serve and training code, on the frames of
# ``test_tile_clip_equals_the_original_canvas``. A 416x234 frame, the size the
# client sends, is trimmed with no resize.
ANTIALIAS_CANVAS_REF = {
    (720, 1280): "e9a050a35e7c8d6fff1940109412f0375d5772b33293327c5cbf3c14dc47db48",
    (234, 416): "8046acf194e1b67ebedc19e5169035f9a3bf772192986c362ac706cadfbd5495",
    (288, 512): "cbec75a15f3995b104e092cc460303d5f4a3bdaa893e15db8aa01912106017ad",
}


@pytest.mark.parametrize("size", sorted(ANTIALIAS_CANVAS_REF))
def test_antialias_clip_equals_the_training_canvas(size: tuple[int, int]) -> None:
    h, w = size
    rng = np.random.default_rng(0)
    ext = rng.integers(0, 256, size=(2, h, w, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(2, h, w, 3), dtype=np.uint8)
    clip = sa.tile_clip(ext, wrist, 224, 416, 0.03, antialias=True)
    assert clip.shape == (1, 3, 2, 224, 832)
    assert _sha(clip.to(torch.uint8).numpy()) == ANTIALIAS_CANVAS_REF[size]


def test_tile_clip_on_a_gradient() -> None:
    yy, xx = np.meshgrid(np.arange(720), np.arange(1280), indexing="ij")
    grad = np.stack([(xx * 255) // 1279, (yy * 255) // 719, (xx + yy) % 256], -1)
    grad = grad.astype(np.uint8)
    clip = sa.tile_clip(grad[None], grad[::-1][None].copy())
    want = "6b26d1868b1454ee34957d887eb595d456859dea3b05b4e3c0597fbb7010be0a"
    assert _sha(clip.to(torch.uint8).numpy()) == want


def test_aspect_crop_boxes() -> None:
    # (source size) -> the crop box (y0, y1, x0, x1) to 256x416 of the original implementation.
    want = {
        (720, 1280): (0, 720, 55, 1225),
        (288, 512): (0, 288, 22, 490),
        (256, 416): (0, 256, 0, 416),
        (480, 640): (43, 437, 0, 640),
        (300, 480): (0, 300, 0, 480),
        (1000, 400): (377, 623, 0, 400),
        (261, 416): (0, 261, 0, 416),
    }
    for (h, w), box in want.items():
        assert aspect_crop_box(h, w, 256, 416, 0.03) == box


def test_tile_clip_layout_and_wrist() -> None:
    ext = np.zeros((1, 256, 416, 3), np.uint8)
    wrist = np.full((1, 256, 416, 3), 200, np.uint8)
    clip = sa.tile_clip(ext, wrist, device="cpu")
    assert clip[..., :416].max() == 0
    assert clip[..., 416:].min() == 200
    one = sa.tile_clip(ext[0], wrist[0])
    assert torch.equal(one, clip)
    with pytest.raises(ValueError):
        sa.tile_clip(ext, None)
    with pytest.raises(ValueError):
        sa.tile_clip(ext, np.concatenate([wrist, wrist]))


def test_canvas_size_of_the_tile_canvas() -> None:
    assert sa.canvas_size(CHECKPOINT_CONFIG) == (256, 832)
    joint = {"video": {"height": 288, "width": 1024, "canvas": "joint"}}
    with pytest.raises(ValueError, match="single-arm checkpoint"):
        sa.canvas_size(joint)
    with pytest.raises(ValueError):
        sa.canvas_size({"video": {"canvas": "mosaic"}})


# ---------------------------------------------------------------------------
# Action and proprio codec
# ---------------------------------------------------------------------------


def test_original_floors_do_not_change_the_stats() -> None:
    """The original implementation widens q99 ranges below 0.02 and floors the
    proprio std at 0.01 (praxis: no range floor, std floor 1e-6). Neither floor
    touches the stats of the checkpoint."""
    q01, q99 = _f32(Q01), _f32(Q99)
    assert float((q99 - q01)[Q99_DIMS].min()) == pytest.approx(0.334318, abs=1e-6)
    assert ((q99 - q01)[Q99_DIMS] >= np.float32(0.02)).all()
    std = _f32(P_STD)
    assert np.array_equal(np.maximum(std, np.float32(0.01)), std)
    assert np.array_equal(np.maximum(std, np.float32(1e-6)), std)


def test_codec_equals_the_original_codec() -> None:
    stats = _stats()
    rng = np.random.default_rng(11)
    rel = rng.normal(scale=0.15, size=(24, 10)).astype(np.float32)
    rel[:, 9] = rng.uniform(-0.2, 1.2, size=24)
    toks = rng.uniform(-1.3, 1.3, size=(24, 10)).astype(np.float32)
    pose8 = np.concatenate(
        [
            rng.normal(scale=0.2, size=(4, 3)) + np.array([0.5, 0.0, 0.3]),
            rng.normal(size=(4, 4)),
            rng.uniform(-0.3, 1.3, size=(4, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    assert _sha(rel) == (
        "3333fbb24e88516ce60098aa0db0d6e87c785a8811a1f5780f0f33a497cce387"
    )
    assert _sha(toks) == (
        "ac9631dd4c3bec0ecdb0249cb6a00bb41330e54413321b318c96efc3fde76ee2"
    )

    normed = q99_normalize(rel, stats.action_q01, stats.action_q99, Q99_DIMS)
    assert _sha(normed) == (
        "d8fef32df8c274ed1c8ee6f956c14cff1e161312f9f1bdfc90163099dc8f1ed6"
    )
    row0 = [-0.08066022396087646, 1.0, 0.9253287315368652, -0.07654605805873871]
    assert normed[0, :4].tolist() == row0
    assert normed[0, 9] == -1.0

    denorm = denormalize_rows(toks, stats, Q99_DIMS)
    assert _sha(denorm) == (
        "60d4c72a0ca6de15a4ddf04919899354f364c2729f9b4d2f467aa6e6aea7d9af"
    )
    row3 = [-0.1485469937324524, 0.15391825139522552, -0.13488346338272095]
    assert denorm[3, :3].tolist() == row3
    assert denorm[3, 9] == np.float32(0.06171119213104248)

    p8 = pose8.copy()
    p8[:, 7] = np.clip(p8[:, 7], 0.0, 1.0)
    prop = proprio_normalize(pose8_to_10d(p8), stats.proprio_mean, stats.proprio_std)
    assert _sha(prop) == (
        "4244aca3a4ed1b4a1a158ffc4c2197c5cac413a63c021820ee64665b6e374920"
    )
    assert prop[1, [0, 3, 9]].tolist() == [
        0.24321340024471283,
        -4.749327182769775,
        0.05717268958687782,
    ]


# ---------------------------------------------------------------------------
# Pose frame
# ---------------------------------------------------------------------------

CLIENT = np.array(
    [
        [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.45, -0.12, 0.41, 0.1, 0.9, -0.3, 0.2, 0.7],
        [0.6, 0.2, 0.2, 0.92388, 0.0, 0.0, -0.38268, 1.4],
        [0.3, 0.1, 0.5, -0.2, 0.5, 0.5, 0.6, -0.1],
        [0.55, -0.05, 0.25, 0.0, 0.0, 0.0, 1.0, 0.5],
    ],
    dtype=np.float32,
)
# The conversion of CLIENT to the flange frame by the original implementation (5 rows of 8).
MODEL = np.array(
    [
        0.5,
        0.0,
        0.30000001192092896,
        0.0,
        0.9238795042037964,
        -0.3826834261417389,
        0.0,
        0.0,
        0.44999998807907104,
        -0.11999999731779099,
        0.4099999964237213,
        0.01626305654644966,
        0.7353048920631409,
        -0.6377265453338623,
        0.22883857786655426,
        0.699999988079071,
        0.6000000238418579,
        0.20000000298023224,
        0.20000000298023224,
        1.0,
        0.0,
        0.0,
        3.3437793263146887e-06,
        1.0,
        0.30000001192092896,
        0.10000000149011612,
        0.5,
        -0.43680116534233093,
        0.6886191368103027,
        0.28523537516593933,
        0.5036360025405884,
        0.0,
        0.550000011920929,
        -0.05000000074505806,
        0.25,
        -0.3826834261417389,
        0.0,
        0.0,
        0.9238795042037964,
        0.5,
    ],
    dtype=np.float32,
).reshape(5, 8)


def test_pose8_to_model_equals_the_original() -> None:
    model = sa.pose8_to_model(CLIENT)
    assert model.dtype == np.float32
    assert np.array_equal(model, MODEL)
    assert _sha(model) == (
        "532a50c9a0347acffbc6b986b28e26415548ec5aab22ed55efc2de4a3f70ee73"
    )
    for i, row in enumerate(CLIENT):
        assert np.array_equal(sa.pose8_to_model(row), MODEL[i])


def test_rotation_port_equals_the_command_path() -> None:
    """The anchor rotation of the commands (unit_quat_wxyz_to_matrix) has the
    bits of the scipy conversion of the original implementation."""
    rng = np.random.default_rng(5)
    for q in rng.normal(size=(2000, 4)).astype(np.float32):
        want = sa.quat_xyzw_to_matrix(q.astype(np.float64)[[1, 2, 3, 0]])
        assert np.array_equal(unit_quat_wxyz_to_matrix(q), want)


def test_matrix_to_quat_branches() -> None:
    s = np.sqrt(0.5)
    cases = {
        (0.0, 0.0, 0.0, 1.0): np.eye(3),
        (1.0, 0.0, 0.0, 0.0): np.diag([1.0, -1.0, -1.0]),
        (0.0, 1.0, 0.0, 0.0): np.diag([-1.0, 1.0, -1.0]),
        (0.0, 0.0, 1.0, 0.0): np.diag([-1.0, -1.0, 1.0]),
        (s, s, 0.0, 0.0): 2 * np.outer([s, s, 0], [s, s, 0]) - np.eye(3),
    }
    for want, m in cases.items():
        assert np.allclose(sa.matrix_to_quat_xyzw(m), want, atol=1e-15)
    rng = np.random.default_rng(6)
    q = rng.normal(size=(64, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    back = sa.matrix_to_quat_xyzw(sa.quat_xyzw_to_matrix(q))
    assert np.allclose(np.abs(np.sum(back * q, axis=1)), 1.0, atol=1e-12)


def test_commands_equal_the_original() -> None:
    rng = np.random.default_rng(12)
    rel = rng.normal(scale=0.05, size=(24, 10)).astype(np.float32)
    rel[:, 3:9] = (
        np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (24, 1)) + rel[:, 3:9]
    )
    rel[:, 9] = rng.uniform(0, 1, size=24)
    assert _sha(rel) == (
        "d156b703c1c72ff19fc31c1dac9cbe7ace4777078704eb44b3b76ff3a8fbd85e"
    )
    wire = sa.rel10_to_wire(rel, MODEL[1])
    poses, grips = sa.wire_to_arrays(wire)
    assert _sha(poses) == (
        "d3ccf3a5f9e55bbc3b5b6faa8f094e7c136ad7c0081e021ee43b6a9a7c8cb7ea"
    )
    assert poses[0, :4].tolist() == [
        0.44965866208076477,
        -0.067692831158638,
        0.4470794200897217,
        0.7893879413604736,
    ]
    assert poses[23, 9:].tolist() == [
        0.4115254282951355,
        0.038934919983148575,
        -0.9105662107467651,
    ]
    assert np.array_equal(grips, rel[:, 9])


def test_commands_round_trip_to_the_rig_pose() -> None:
    """A zero relative action at the model-frame anchor gives back the rig pose."""
    rel = np.zeros((1, 10), np.float32)
    rel[0, 3:9] = [1, 0, 0, 0, 1, 0]
    for row, model in zip(CLIENT, MODEL, strict=True):
        pose12, _ = sa.wire_to_arrays(sa.rel10_to_wire(rel, model))
        assert np.array_equal(pose12[0, :3], row[:3])
        want = unit_quat_wxyz_to_matrix(row[3:7])
        assert np.allclose(pose12[0, 3:].reshape(3, 3), want, atol=2e-6)


# ---------------------------------------------------------------------------
# Session with the tile canvas and the flange frame
# ---------------------------------------------------------------------------


def _pose8(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    xyz = rng.normal(size=(n, 3)) * 0.2
    grip = rng.uniform(-0.2, 1.2, size=(n, 1))
    return np.concatenate([xyz, q, grip], axis=1).astype(np.float32)


def _obs(frames: dict, pose8: np.ndarray, lo: int, hi: int) -> dict:
    return {
        "image.exterior": frames["exterior"][lo:hi],
        "image.wrist": frames["wrist"][lo:hi],
        "robot_state.ee_pose": pose8[lo:hi, :7],
        "grip": pose8[lo:hi, 7:8],
    }


class _Encoder:
    """A stand-in encoder that keeps its input clips."""

    def __init__(self) -> None:
        self.clips: list[torch.Tensor] = []

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        self.clips.append(clip)
        t = clip.shape[2]
        idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
        return torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 32, 32))


def _session(enc: _Encoder) -> sa.SingleArmServeSession:
    return sa.SingleArmServeSession(
        encode=enc,
        stats=_stats(),
        device="cpu",
        prompt="Put the marker in the bowl",
        control_mode="droid_impedance",
        exec_horizon=24,
        tile_size=(256, 416),
        aspect_tol=0.03,
    )


def _frames(rng: np.random.Generator, n: int) -> dict:
    return {
        v: rng.integers(0, 256, size=(n, 36, 64, 3), dtype=np.uint8)
        for v in ("exterior", "wrist")
    }


def test_session_tiles_and_flange_frame() -> None:
    rng = np.random.default_rng(8)
    n = 70
    frames, p = _frames(rng, n), _pose8(rng, n)
    enc = _Encoder()
    sess = _session(enc)
    stats = _stats()
    depths = []
    for f in (0, 21, 45, 69):
        lo = max(0, f - 24)
        enc.clips = []
        inputs = sess.prepare(_obs(frames, p, lo, f + 1))
        depths.append(inputs.depth)
        assert all(c.shape[-2:] == (256, 832) for c in enc.clips)
        if inputs.depth == 0:
            cur = sa.tile_clip(frames["exterior"][f], frames["wrist"][f])
            assert torch.equal(enc.clips[0], cur)
        # Warm calls encode only the executed stream and reuse its anchor.
        assert len(enc.clips) == 1
        model_now = sa.pose8_to_model(p[f], sa.FR3_HAND)  # no descriptor: the FR3 rig
        assert np.array_equal(inputs.pose8_now, model_now)
        offs = [*[0, 21, 45][: inputs.depth], f]
        want = sa.pose8_to_model(p[offs], sa.FR3_HAND)
        assert np.array_equal(inputs.boundary_poses, want)
        assert inputs.boundary_poses[:, 7].min() >= 0.0
        assert inputs.boundary_poses[:, 7].max() <= 1.0
        want = proprio_normalize(
            pose8_to_10d(inputs.boundary_poses), stats.proprio_mean, stats.proprio_std
        )
        assert np.array_equal(inputs.proprio_all[0].numpy(), want)
        assert np.array_equal(inputs.proprio[0].numpy(), want[-1:])
        wire = sess.finish(inputs, torch.zeros(1, 6, 40))
        _rows, rel = sess.decode_actions(inputs, torch.zeros(1, 6, 40))
        again = sess.rel_to_wire(rel, inputs.pose8_now)
        pw, gw = sa.wire_to_arrays(wire)
        pa, ga = sa.wire_to_arrays(again)
        assert np.array_equal(pw, pa) and np.array_equal(gw, ga)
        ref = sa.rel10_to_wire(
            rel, model_now, 1.0 / 15.0, sa.DROID_IMPEDANCE, sa.FR3_HAND
        )
        assert np.array_equal(pw, sa.wire_to_arrays(ref)[0])
        assert wire[0]["robot_command"]["mode"] == sa.DROID_IMPEDANCE
    assert depths == [0, 1, 2, 3]
    assert sess.restart_count == 0
    # The executed frames of the window, in one tile clip.
    stream = enc.clips[0]
    assert stream.shape == (1, 3, 69, 256, 832)
    want = sa.tile_clip(frames["exterior"][:69], frames["wrist"][:69])
    assert torch.equal(stream, want)


def test_session_tiles_restart_on_a_short_wrist_stack() -> None:
    rng = np.random.default_rng(9)
    frames, p = _frames(rng, 30), _pose8(rng, 30)
    sess = _session(_Encoder())
    sess.finish(sess.prepare(_obs(frames, p, 0, 1)), torch.zeros(1, 6, 40))
    obs = _obs(frames, p, 0, 22)
    obs["image.wrist"] = frames["wrist"][21:22]
    inputs = sess.prepare(obs)
    assert inputs.depth == 0 and sess.restart_count == 1
    sess.finish(inputs, torch.zeros(1, 6, 40))
    obs = _obs(frames, p, 0, 22)
    del obs["image.wrist"]
    with pytest.raises(ValueError):
        sess.prepare(obs)
    obs = _obs(frames, p, 0, 22)
    obs["image.wrist"] = obs["image.exterior"][-1]
    assert sess.prepare(obs).depth == 0
    assert sess.restart_count == 2


@pytest.mark.parametrize("antialias", [False, True])
def test_session_clips_follow_the_checkpoint_antialias(antialias: bool) -> None:
    """The pipeline session passes ``video.antialias`` to the current and executed clips."""
    from types import SimpleNamespace

    from praxis.pipeline import Session

    rng = np.random.default_rng(10)
    frames, p = _frames(rng, 30), _pose8(rng, 30)
    frames = {
        v: rng.integers(0, 256, size=(30, 288, 512, 3), dtype=np.uint8) for v in frames
    }
    config = json.loads(
        (Path(__file__).resolve().parent / "data" / "praxis-1-droid.json").read_text()
    )
    config["video"]["antialias"] = antialias
    enc = _Encoder()
    pipe = SimpleNamespace(
        policy=SimpleNamespace(config=config),
        vae=SimpleNamespace(encode=enc),
        stats=_stats(),
        device="cpu",
        sampler=None,
    )
    sess = Session(pipe)  # ty: ignore[invalid-argument-type]
    state = sess.state
    state.finish(state.prepare(_obs(frames, p, 0, 1)), torch.zeros(1, 6, 40))
    cur = enc.clips[0]
    enc.clips = []
    state.prepare(_obs(frames, p, 0, 22))
    (stream,) = enc.clips
    assert cur.shape[-2:] == stream.shape[-2:] == (224, 832)
    want_cur = sa.tile_clip(
        frames["exterior"][0], frames["wrist"][0], 224, 416, 0.03, antialias=antialias
    )
    want_stream = sa.tile_clip(
        frames["exterior"][:21],
        frames["wrist"][:21],
        224,
        416,
        0.03,
        antialias=antialias,
    )
    assert torch.equal(cur, want_cur) and torch.equal(stream, want_stream)
    other = sa.tile_clip(
        frames["exterior"][0],
        frames["wrist"][0],
        224,
        416,
        0.03,
        antialias=not antialias,
    )
    assert not torch.equal(cur, other)


def test_pinned_stats_are_the_checkpoint_stats() -> None:
    """The pinned codec statistics equal the ``normalization`` block of
    praxis-1-droid."""
    config = json.loads(
        (Path(__file__).resolve().parent / "data" / "praxis-1-droid.json").read_text()
    )
    norm = config["normalization"]
    assert norm["action_q01"] == Q01 and norm["action_q99"] == Q99
    assert norm["proprio_mean"] == P_MEAN and norm["proprio_std"] == P_STD
