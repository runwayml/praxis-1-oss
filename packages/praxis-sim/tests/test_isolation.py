"""The package installs without the model: no torch, no core praxis module, Python 3.10+."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import praxis_sim

PACKAGE_DIR = Path(praxis_sim.__file__).resolve().parent
# What the package may import besides the standard library and itself.
ALLOWED = {"numpy", "mujoco", "OpenGL", "yaml", "gymnasium", "PIL"}


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
        elif isinstance(node, ast.ImportFrom):
            names.add("praxis_sim")  # a relative import stays in the package
    return names


def test_sources_import_only_the_simulator_dependencies() -> None:
    files = sorted(PACKAGE_DIR.glob("*.py"))
    assert len(files) >= 14
    for path in files:
        for name in _imports(path):
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top in ALLOWED or top == "praxis_sim":
                continue
            raise AssertionError(f"{path.name} imports {name}")


def test_importing_every_module_loads_no_model_code() -> None:
    # A fresh interpreter: the test may run next to tests that import praxis already.
    code = (
        "import importlib, pkgutil, sys, praxis_sim\n"
        "for info in pkgutil.iter_modules(praxis_sim.__path__):\n"
        "    importlib.import_module(f'praxis_sim.{info.name}')\n"
        "print(' '.join(sorted({name.split('.')[0] for name in sys.modules})))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
    )
    loaded = set(out.stdout.split())
    assert "praxis_sim" in loaded
    assert "torch" not in loaded and "praxis" not in loaded
