"""Checks that the simulation package stays apart from the core package (AST only).

Rules:

* Only ``praxis/sim``, ``tests/test_sim_*.py``, ``scripts/run_sim.py``,
  ``scripts/run_sim_eval.py`` and ``scripts/compare_sim_eval.py`` import
  ``praxis.sim``.
* ``praxis.sim`` imports only the standard library, numpy, Pillow, PyYAML,
  MuJoCo (with PyOpenGL, a MuJoCo dependency), imageio-ffmpeg, torch and the
  core ``praxis`` modules (not ``praxis.serve``). Relative imports count too.
* The core modules do not import the packages of the ``sim`` extra.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "praxis" / "sim"
ALLOWED_THIRD_PARTY = {
    "numpy",
    "PIL",
    "yaml",
    "mujoco",
    "OpenGL",
    "imageio",
    "imageio_ffmpeg",
    "torch",
}
SIM_ONLY = {"mujoco", "OpenGL", "imageio_ffmpeg"}
# The server package is not part of the core: the sim must not import it.
SERVER_PACKAGE = ["praxis", "serve"]


def _imports(path: Path) -> set[str]:
    """Return the absolute module names that a file imports."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(_resolve_relative(path, node))
    return names


def _resolve_relative(path: Path, node: ast.ImportFrom) -> set[str]:
    """Return the absolute names of a relative import in ``path``."""
    package = list(path.relative_to(ROOT).parts[:-1])  # also for __init__.py
    base = package[: len(package) - node.level + 1]
    module = ".".join(base + ([node.module] if node.module else []))
    return {module} | {f"{module}.{alias.name}" for alias in node.names}


def _python_files() -> list[Path]:
    skip = {".git", ".venv", "__pycache__", "build", "dist"}
    return [p for p in ROOT.rglob("*.py") if not skip & set(p.relative_to(ROOT).parts)]


def _may_import_sim(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel.parts[:2] == ("praxis", "sim"):
        return True
    if rel.parts[0] == "tests" and rel.name.startswith("test_sim_"):
        return True
    return rel in (
        Path("scripts/run_sim.py"),
        Path("scripts/run_sim_eval.py"),
        Path("scripts/compare_sim_eval.py"),
    )


def test_only_sim_files_import_praxis_sim() -> None:
    bad = []
    for path in _python_files():
        if _may_import_sim(path):
            continue
        hits = [
            n
            for n in _imports(path)
            if n == "praxis.sim" or n.startswith("praxis.sim.")
        ]
        if hits:
            bad.append(f"{path.relative_to(ROOT)}: {sorted(hits)}")
    assert not bad, bad


def test_sim_imports_only_allowed_packages() -> None:
    files = sorted(SIM_DIR.glob("*.py"))
    assert len(files) >= 10
    bad = []
    for path in files:
        for name in _imports(path):
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY:
                continue
            if top == "praxis" and name.split(".")[:2] != SERVER_PACKAGE:
                continue
            bad.append(f"{path.name}: {name}")
    assert not bad, bad


def test_core_does_not_need_the_sim_extra() -> None:
    bad = []
    for path in sorted((ROOT / "praxis").rglob("*.py")):
        if path.is_relative_to(SIM_DIR):
            continue
        hits = {n.split(".")[0] for n in _imports(path)} & SIM_ONLY
        if hits:
            bad.append(f"{path.relative_to(ROOT)}: {sorted(hits)}")
    assert not bad, bad


def test_relative_imports_resolve() -> None:
    loop = ROOT / "praxis" / "sim" / "loop.py"
    tree = ast.parse(
        "from . import env\nfrom .. import model\nfrom ..pipeline import x\n"
    )
    names = set()
    for node in tree.body:
        assert isinstance(node, ast.ImportFrom)
        names |= _resolve_relative(loop, node)
    assert {"praxis.sim.env", "praxis.model", "praxis.pipeline.x"} <= names
