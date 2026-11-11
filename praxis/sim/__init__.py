"""MuJoCo simulation of the single-arm rig: the model loop, the video writer and the live viewer.

The simulator itself (scene, robot, tasks, environment, the evaluation suites, the
scripted policy, the run settings) is the ``praxis-sim`` package of
``packages/praxis-sim``, installable without torch (the ``sim`` extra of ``praxis``
pulls it in). This package keeps the parts that need the model:
:mod:`praxis.sim.loop` (the closed-loop sampling loop and the policies),
:mod:`praxis.sim.video` (the MP4 writer) and :mod:`praxis.sim.viewer` (the live
view in a web browser).

The simulator modules stay reachable under their old names (``praxis.sim.scene``,
``praxis.sim.env``, ...): each is the module of ``praxis_sim`` of the same name,
imported on first use, so that ``import praxis.sim`` does not import MuJoCo.

Run it with ``scripts/run_sim.py``. See ``praxis/sim/README.md``.

MuJoCo reads the OpenGL backend when Python imports it. ``praxis_sim`` sets the
default backend before any sub-module imports MuJoCo. To select a different
backend, call :func:`praxis_sim.gl.prepare` (``praxis.sim.gl.prepare``) before the
first import of a module that uses MuJoCo.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

try:
    import praxis_sim  # noqa: F401  (sets the OpenGL backend)
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        'praxis.sim needs the praxis-sim package: pip install -e ".[sim]" '
        "(or, in a checkout, pip install -e packages/praxis-sim)"
    ) from e

# The simulator modules under their former names, imported on first use.
_MOVED = frozenset(
    {
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
    }
)


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import ``praxis.sim.<name>`` as the module ``praxis_sim.<name>`` (the same object)."""

    def __init__(self) -> None:
        self._specs: dict[str, importlib.machinery.ModuleSpec | None] = {}

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        prefix, _, name = fullname.rpartition(".")
        if prefix != __name__ or name not in _MOVED:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        module = importlib.import_module(f"praxis_sim.{spec.name.rpartition('.')[2]}")
        self._specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        # The import system has set ``module.__spec__`` to the alias spec; the module
        # keeps its own (``praxis_sim.<name>``), as ``importlib.reload`` and tools expect.
        alias = module.__spec__.name if module.__spec__ is not None else ""
        module.__spec__ = self._specs.pop(alias, module.__spec__)


# Once per interpreter: a reload of this package must not add a second finder.
if not any(
    type(f).__name__ == "_Alias" and type(f).__module__ == __name__
    for f in sys.meta_path
):
    sys.meta_path.insert(0, _Alias())


def __getattr__(name: str) -> types.ModuleType:
    if name in _MOVED:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
