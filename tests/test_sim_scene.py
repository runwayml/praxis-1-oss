"""Unit tests for the sim assets and scene (CPU physics, no model weights).

The tests need the ``mujoco`` package and the pinned Menagerie files. They get
the files from the cache, or download them. Without the files and without a
network, the tests skip.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

# praxis.sim sets the OpenGL backend. It must load before mujoco.
import praxis.sim  # noqa: F401

mujoco = pytest.importorskip("mujoco")

from praxis.sim import assets, gl  # noqa: E402
from praxis.sim.robot import DROID_HOME  # noqa: E402
from praxis.sim.scene import (  # noqa: E402
    CAMERAS,
    EE_SITE,
    FLANGE_SITE,
    GRIP_ACTUATOR,
    PINCH_OFFSET,
    PINCH_SITE,
    SceneConfig,
    build_model,
)
from praxis.sim.tasks import TaskConfig, make_task  # noqa: E402

# Denavit-Hartenberg parameters of the Franka arm (modified convention):
# (a, d, alpha) per joint. Public values of the Franka documentation.
FRANKA_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -math.pi / 2),
    (0.0, 0.316, math.pi / 2),
    (0.0825, 0.0, math.pi / 2),
    (-0.0825, 0.384, -math.pi / 2),
    (0.0, 0.0, math.pi / 2),
    (0.088, 0.0, math.pi / 2),
    (0.0, 0.107, 0.0),  # flange, fixed
)


def _assets_or_skip() -> None:
    try:
        assets.ensure_assets()
    except assets.AssetError as exc:
        pytest.skip(f"The pinned Menagerie files are not available: {exc}")


def _franka_flange(q: np.ndarray) -> np.ndarray:
    """Return the 4x4 flange pose of the Franka arm (modified DH)."""
    t = np.eye(4)
    for (a, d, alpha), theta in zip(FRANKA_DH, [*q, 0.0], strict=True):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        t = t @ np.array(
            [
                [ct, -st, 0.0, a],
                [st * ca, ct * ca, -sa, -sa * d],
                [st * sa, ct * sa, ca, ca * d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return t


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    _assets_or_skip()
    task = make_task(TaskConfig())
    return build_model(SceneConfig(), task.objects(), task.fixtures())


def test_pinned_file_table() -> None:
    paths = [p for p, _, _ in assets.FILES]
    assert len(paths) == len(set(paths)) == 88
    for model_source in assets.MODELS.values():
        assert model_source.xml in paths
        assert f"{model_source.directory}/LICENSE" in paths
    for _, sha, size in assets.FILES:
        assert len(sha) == 64 and int(sha, 16) >= 0 and size > 0
    assert len(assets.MENAGERIE_COMMIT) == 40


def test_assets_check_and_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    good, bad = b"good file", b"other bytes"
    (src / "d" / "a.txt").write_bytes(good)
    table = (("d/a.txt", hashlib.sha256(good).hexdigest(), len(good)),)
    monkeypatch.setattr(assets, "FILES", table)
    cache = tmp_path / "cache"
    assert not assets.assets_ready(cache)
    with pytest.raises(assets.AssetError, match="missing"):
        assets.ensure_assets(cache, download=False)
    root = assets.ensure_assets(cache, source=src)
    assert (root / "d" / "a.txt").read_bytes() == good
    assert assets.assets_ready(cache)
    # A changed file in the cache is found and replaced.
    (root / "d" / "a.txt").write_bytes(bad[: len(good)].ljust(len(good), b"x"))
    assert assets.missing_files(root) == ["d/a.txt"]
    assets.ensure_assets(cache, source=src)
    assert assets.assets_ready(cache)
    # A source file with a wrong checksum is refused.
    (src / "d" / "a.txt").write_bytes(bad)
    assets.clear_cache(cache)
    with pytest.raises(assets.AssetError, match="SHA-256"):
        assets.ensure_assets(cache, source=src)


def test_gl_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert gl.is_software_renderer("llvmpipe (LLVM 20.1.2, 256 bits)")
    assert not gl.is_software_renderer("NVIDIA H100 80GB HBM3/PCIe/SSE2")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/x.json")
    assert gl.add_nvidia_vendor(tmp_path) is None
    with pytest.raises(ValueError):
        gl.prepare(backend="bogus")


def test_scene_builds(model: mujoco.MjModel) -> None:
    assert model.opt.timestep == pytest.approx(1.0 / (15 * 34))
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for name in CAMERAS:
        assert model.camera(name).id >= 0
    for name in (EE_SITE, FLANGE_SITE, PINCH_SITE):
        assert model.site(name).id >= 0
    assert model.nu == 8  # 7 arm servos and the gripper
    assert model.actuator(GRIP_ACTUATOR).ctrlrange.tolist() == [0.0, 255.0]
    assert model.body_gravcomp[model.body("link3").id] == 1.0
    assert model.body_gravcomp[model.body("red_cube").id] == 0.0


def test_flange_matches_franka_dh(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    qadr = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
    ranges = np.array([model.joint(f"joint{i}").range for i in range(1, 8)])
    flange, ee = model.site(FLANGE_SITE).id, model.site(EE_SITE).id
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(ranges[:, 0], ranges[:, 1])
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        want = _franka_flange(q)
        assert np.abs(data.site_xpos[flange] - want[:3, 3]).max() < 1e-6
        assert np.abs(data.site_xmat[flange].reshape(3, 3) - want[:3, :3]).max() < 1e-6
        np.testing.assert_allclose(
            data.site_xpos[ee], data.site_xpos[flange], atol=1e-9
        )


def test_home_pose_and_finger_line(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    qadr = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
    data.qpos[qadr] = DROID_HOME
    mujoco.mj_kinematics(model, data)
    ee = model.site(EE_SITE).id
    np.testing.assert_allclose(data.site_xpos[ee], [0.3597, 0.0, 0.4899], atol=1e-4)
    r_ee = data.site_xmat[ee].reshape(3, 3)
    assert math.degrees(math.atan2(r_ee[1, 0], r_ee[0, 0])) == pytest.approx(
        45.0, abs=1e-3
    )
    np.testing.assert_allclose(r_ee[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
    # The fingers open along the flange y axis: (-0.71, 0.71, 0) in the ee frame.
    r_flange = data.site_xmat[model.site(FLANGE_SITE).id].reshape(3, 3)
    left = data.geom_xpos[model.geom("g/left_pad1").id]
    right = data.geom_xpos[model.geom("g/right_pad1").id]
    line = (right - left) / np.linalg.norm(right - left)
    assert abs(float(np.dot(line, r_flange[:, 1]))) > 0.999
    line_ee = r_ee.T @ line
    assert (
        abs(float(np.dot(line_ee, [-1 / math.sqrt(2), 1 / math.sqrt(2), 0.0]))) > 0.999
    )
    pinch = data.site_xpos[model.site(PINCH_SITE).id]
    mid = (left + right) / 2
    assert np.linalg.norm(pinch - data.site_xpos[ee]) == pytest.approx(PINCH_OFFSET)
    assert np.linalg.norm((pinch - mid)[:2]) < 0.01


def test_cameras_render_shapes() -> None:
    _assets_or_skip()
    from praxis.sim.env import EnvConfig, SimEnv

    try:
        env = SimEnv(EnvConfig(scene=SceneConfig(msaa_samples=0)))
    except Exception as exc:  # no OpenGL backend on this machine
        pytest.skip(f"MuJoCo cannot make an offscreen OpenGL context: {exc}")
    try:
        obs = env.reset(seed=0)
        assert obs.exterior is not None and obs.wrist is not None
        assert obs.overview is not None
        assert obs.exterior.shape == obs.wrist.shape == (288, 512, 3)
        assert obs.overview.shape == (360, 640, 3)
        for img in (obs.exterior, obs.wrist, obs.overview):
            assert img.dtype == np.uint8 and img.std() > 5
            red = (img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)
            assert red.sum() > 20  # the red cube is in view
        again = env.observe()
        assert again.wrist is not None and again.wrist is not obs.wrist
        assert np.array_equal(again.wrist, obs.wrist)
    finally:
        env.close()
