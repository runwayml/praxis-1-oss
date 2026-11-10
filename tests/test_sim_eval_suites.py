"""The fixed evaluation suites and the paired comparison (CPU only, no weights)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("praxis_sim")  # the `sim` extra

from praxis.sim.eval_suites import (
    SUITES,
    mcnemar_p,
    paired_difference,
    sub_dir,
)

ROOT = Path(__file__).resolve().parents[1]


def test_suites_have_the_documented_size() -> None:
    assert SUITES["droid"].episodes == 224
    assert SUITES["droid_guard"].episodes == 56
    assert SUITES["hard"].episodes == 352


def test_every_suite_task_is_registered() -> None:
    pytest.importorskip("mujoco")
    from praxis.sim.tasks import TaskConfig, make_task

    for suite in SUITES.values():
        for name in suite.tasks:
            assert make_task(TaskConfig(name=name)).name == name


def test_units_are_seed_major_and_complete() -> None:
    suite = SUITES["hard"]
    units = suite.units(1)
    assert len(units) == len(suite.tasks) * suite.scene_seeds
    assert units[: len(suite.tasks)] == [(t, 0) for t in suite.tasks]
    assert len(set(units)) == len(units)
    with pytest.raises(ValueError):
        SUITES["droid_guard"].units(1)


def test_eval_doc_lists_every_suite_task() -> None:
    doc = (ROOT / "praxis" / "sim" / "EVAL.md").read_text()
    for suite in SUITES.values():
        assert f"`{suite.name}`" in doc
        for name in suite.tasks:
            assert f"`{name}`" in doc, name


def test_sub_dir_names() -> None:
    assert sub_dir(0) == "suite"
    assert sub_dir(1) == "suite_s1"


def test_mcnemar_p_is_the_exact_binomial_test() -> None:
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    # 0 of 6 discordant pairs on one side: 2 * 0.5**6.
    assert mcnemar_p(6, 0) == pytest.approx(2 * 0.5**6)
    want = 2 * sum(math.comb(138, i) for i in range(51)) / 2**138
    assert mcnemar_p(88, 50) == pytest.approx(want)


def test_paired_difference_counts_wins_and_losses() -> None:
    keys = [("t", s, 0) for s in range(10)]
    a = {k: i < 6 for i, k in enumerate(keys)}  # 6 of 10
    b = {k: 2 <= i < 10 for i, k in enumerate(keys)}  # 8 of 10
    res = paired_difference(b, a)
    assert res["n"] == 10 and res["wins"] == 4 and res["losses"] == 2
    assert res["diff"] == pytest.approx(0.2)
    assert res["lo"] <= res["diff"] <= res["hi"]
    with pytest.raises(ValueError):
        paired_difference({("t", 0, 0): True}, {("u", 0, 0): True})


def _write_run(root: Path, successes: dict[tuple[str, int, int], bool]) -> None:
    for (task, seed, ss), ok in successes.items():
        d = root / sub_dir(ss) / task
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "task": task,
            "seed": seed,
            "sampler_seed": ss,
            "success": ok,
            "progress": 1.0 if ok else 0.5,
        }
        (d / f"episode_{seed:03d}.json").write_text(json.dumps(rec))


def test_compare_script_pairs_the_episodes(tmp_path: Path) -> None:
    keys = [(t, s, ss) for t in ("x", "y") for s in range(4) for ss in (0, 1)]
    _write_run(tmp_path / "a", {k: k[1] < 2 for k in keys})
    _write_run(tmp_path / "b", {k: k[1] < 3 for k in keys})
    out = tmp_path / "cmp.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_sim_eval.py"),
            str(tmp_path / "a"),
            str(tmp_path / "b"),
            "--json",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    res = json.loads(out.read_text())
    assert res["total"]["n"] == 16
    assert (res["total"]["a"], res["total"]["b"]) == (8, 12)
    assert (res["total"]["wins"], res["total"]["losses"]) == (4, 0)
    assert res["tasks"]["x"]["wins"] == 2
