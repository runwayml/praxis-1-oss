"""scripts/run_inference.py records a call from the pipeline's StepResult.

A stub StepResult with the keys the pipeline fills in, so a renamed key in
``StepResult.inputs`` fails here and not on the first call of a replay.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from praxis import single_arm
from praxis.pipeline import StepResult

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "run_inference_script", ROOT / "scripts" / "run_inference.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_call_record_reads_the_step_result() -> None:
    pose8 = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2], np.float32)
    rel = np.zeros((21, 10), np.float32)
    rel[:, 3], rel[:, 7] = 1.0, 1.0  # identity rot6d
    commands = single_arm.rel10_to_wire(rel, pose8)
    latent = torch.zeros(1, 48, 1, 2, 2)
    res = StepResult(
        commands=commands,
        action_rows_norm=np.zeros((21, 10), np.float32),
        action_rel=rel,
        pred_video_latent=torch.zeros(1, 48, 6, 2, 2),
        pred_action_norm=torch.zeros(1, 6, 40),
        depth=0,
        seed=3,
        prompt="pick",
        inputs={
            "text_embedding": torch.zeros(1, 4, 8),
            "anchor": latent,
            "cleans": [],
            "proprio": torch.zeros(1, 1, 10),
            "proprio_all": torch.zeros(1, 1, 10),
            "boundary_poses": pose8[None],
            "pose_now": pose8,
        },
    )
    clip = torch.zeros(1, 3, 1, 32, 32)
    rec = _script().call_record(res, [(clip, latent)])
    assert torch.equal(rec["inputs"]["pose8_now"], torch.from_numpy(pose8))
    assert rec["command_pose"].shape == (21, 12) and rec["seed"] == 3
    assert torch.equal(rec["inputs"]["latent_boundary"], latent)
