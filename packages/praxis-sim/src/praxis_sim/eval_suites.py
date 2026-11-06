"""The fixed evaluation scenarios of the single-arm checkpoint.

An evaluation suite is a list of tasks, a number of scene seeds and a list of
sampler seeds. One episode is one ``(task, scene seed, sampler seed)`` unit.
The scene seed places the objects and selects the target of the prompt. The
sampler seed is ``session.seed`` of the policy. Each suite uses the default
prompt and the default time limit of each task.

``praxis/sim/EVAL.md`` describes the scenarios, the run command and the
comparison of two checkpoints. ``scripts/run_sim_eval.py`` runs a suite and
``scripts/compare_sim_eval.py`` compares two runs on the same episodes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SUITES",
    "EvalSuite",
    "mcnemar_p",
    "paired_difference",
    "sub_dir",
]


@dataclass(frozen=True)
class EvalSuite:
    """One fixed set of episodes: tasks x scene seeds x sampler seeds."""

    name: str
    tasks: tuple[str, ...]
    scene_seeds: int  # scene seeds 0 .. scene_seeds - 1
    sampler_seeds: tuple[int, ...]
    note: str

    @property
    def episodes(self) -> int:
        """Return the number of episodes of the suite."""
        return len(self.tasks) * self.scene_seeds * len(self.sampler_seeds)

    def units(self, sampler_seed: int) -> list[tuple[str, int]]:
        """Return the ``(task, scene seed)`` units of one sampler seed.

        The order is seed-major: scene seed 0 of every task, then scene seed
        1, and so on. A run that stops early thus covers all tasks evenly.
        """
        if sampler_seed not in self.sampler_seeds:
            raise ValueError(
                f"suite {self.name!r} uses the sampler seeds {self.sampler_seeds}"
            )
        return [(t, s) for s in range(self.scene_seeds) for t in self.tasks]


#: The 7 DROID task families. The single-arm checkpoint solves most of these
#: episodes, so the suite checks for regressions.
DROID_TASKS = (
    "lift_cube",
    "cube_on_plate",
    "block_in_bowl",
    "block_out_of_bowl",
    "move_block",
    "stack_block",
    "press_button",
)

#: The harder tasks. The single-arm checkpoint solves about 20 to 60 % of the
#: episodes of each task, so the suite can show a gain or a loss.
HARD_TASKS = (
    "cube_in_tray",
    "cube_on_color",
    "pick_color_block",
    "color_block_in_bowl",
    "stack_color_blocks",
    "peg_in_hole_6mm",
    "block_in_small_bowl",
    "cube_on_small_plate",
    "stack_small_block",
    "two_blocks_in_bowl",
    "stack_three_blocks",
)

SUITES: dict[str, EvalSuite] = {
    s.name: s
    for s in (
        EvalSuite(
            "droid",
            DROID_TASKS,
            16,
            (0, 1),
            "the 7 DROID task families, 224 episodes",
        ),
        EvalSuite(
            "droid_guard",
            DROID_TASKS,
            8,
            (0,),
            "the 7 DROID task families, 56 episodes: a fast regression check",
        ),
        EvalSuite(
            "hard",
            HARD_TASKS,
            16,
            (0, 1),
            "the 11 harder tasks, 352 episodes",
        ),
    )
}


def sub_dir(sampler_seed: int) -> str:
    """Return the output sub-directory of a sampler seed."""
    return "suite" if sampler_seed == 0 else f"suite_s{sampler_seed}"


def mcnemar_p(wins: int, losses: int) -> float:
    """Return the exact two-sided McNemar p of the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * p)


def paired_difference(
    a: Mapping[tuple[str, int, int], bool],
    b: Mapping[tuple[str, int, int], bool],
    *,
    keys: Iterable[tuple[str, int, int]] | None = None,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Compare the success of ``a`` and ``b`` on the same episodes.

    The keys are ``(task, scene seed, sampler seed)``. The function uses the
    episodes in both runs (or ``keys``). It returns the mean difference
    ``a - b`` of the success (1 or 0), the 95 % bootstrap interval of the mean
    (``lo``, ``hi``), the number of pairs, the wins (``a`` succeeds and ``b``
    fails), the losses and the exact McNemar p of the wins and losses.
    """
    ks = sorted(set(a) & set(b) if keys is None else keys)
    if not ks:
        raise ValueError("the two runs have no common episodes")
    d = np.array([float(a[k]) - float(b[k]) for k in ks])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    wins = sum(1 for k in ks if a[k] and not b[k])
    losses = sum(1 for k in ks if b[k] and not a[k])
    return {
        "diff": float(d.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": len(ks),
        "wins": wins,
        "losses": losses,
        "mcnemar_p": mcnemar_p(wins, losses),
    }
