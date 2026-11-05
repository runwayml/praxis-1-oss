"""MuJoCo simulation of the single-arm rig: the model loop, the video writer and the live viewer.

The simulator itself (scene, robot, tasks, environment, the evaluation suites, the
scripted policy, the run settings) is the ``praxis-sim`` package of
``packages/praxis-sim``, installable without torch. This package keeps the parts
that need the model: :mod:`praxis.sim.loop` (the closed-loop sampling loop and
the policies), :mod:`praxis.sim.video` (the MP4 writer) and
:mod:`praxis.sim.viewer` (the live view in a web browser).

The simulator modules stay reachable under their old names (``praxis.sim.scene``,
``praxis.sim.env``, ...): they are the modules of ``praxis_sim``.

Run it with ``scripts/run_sim.py``. See ``praxis/sim/README.md``.

MuJoCo reads the OpenGL backend when Python imports it. ``praxis_sim`` sets the
default backend before any sub-module imports MuJoCo. To select a different
backend, call :func:`praxis_sim.gl.prepare` before the first import of a module
that uses MuJoCo.
"""

import importlib
import sys

import praxis_sim  # noqa: F401  (sets the OpenGL backend)

# The simulator modules under their former names.
for _name in (
    "assets",
    "config",
    "env",
    "eval_suites",
    "gl",
    "robot",
    "scene",
    "scripted",
    "suite",
    "suite_hard",
    "suite_insert",
    "suite_multi",
    "suite_place",
    "tasks",
):
    _module = importlib.import_module(f"praxis_sim.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
del _module, _name
