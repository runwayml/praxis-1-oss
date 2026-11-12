"""The policy server stays a separate part of the repository.

A release without the server deletes ``praxis/serve/``, ``docker/``,
``.dockerignore``, ``configs/serve_*.yaml``, ``tests/serve/`` and the
clients of the server (``eval/so101/``), and the ``[serve]`` extra of
``pyproject.toml`` (see ``praxis/serve/README.md``). These tests make sure
that the rest of the code then still works.
"""

import ast
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The paths that a release without the server deletes.
SERVER_PARTS = ("praxis/serve/", "docker/", "tests/serve/", "eval/so101/")
# Packages that only the server may need. The core must not list them.
SERVER_ONLY = {"fastapi", "starlette", "uvicorn", "websockets", "msgpack", "modal"}


def _python_files() -> list[Path]:
    """Return the Python files of the repository, without build output."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others",
             "--exclude-standard", "*.py"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        files = [ROOT / p for p in out.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        files = list(ROOT.rglob("*.py"))

    def build_output(path: Path) -> bool:
        return any(
            part.startswith(".")
            or part in ("build", "dist")
            or part.endswith(".egg-info")
            for part in path.relative_to(ROOT).parts
        )

    return sorted(p for p in files if p.is_file() and not build_output(p))


def _server_part(path: Path) -> bool:
    return path.relative_to(ROOT).as_posix().startswith(SERVER_PARTS)


def _imported_modules(path: Path) -> set[str]:
    """Return the absolute names of all modules that ``path`` imports."""
    package = path.relative_to(ROOT).with_suffix("").parts[:-1]
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = list(package[: len(package) - node.level + 1]) if node.level else []
            module = ".".join([*base, node.module] if node.module else base)
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("praxis.serve"):
                names.add(node.value)  # for example importlib.import_module(...)
    return names


def _names(reqs: list[str]) -> set[str]:
    return {re.split(r"[<>=!~;\[ ]", r, maxsplit=1)[0].strip().lower() for r in reqs}


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def _server_packages() -> set[str]:
    """Return the packages of the [serve] extra that no other extra lists."""
    project = _project()
    others = _names(project["dependencies"])
    for name, reqs in project["optional-dependencies"].items():
        if name != "serve":
            others |= _names(reqs)
    serve = _names(project["optional-dependencies"]["serve"]) - others
    return {n.replace("-", "_") for n in serve} | {"starlette"}


def test_no_file_outside_the_server_imports_it() -> None:
    files = [p for p in _python_files() if not _server_part(p)]
    assert any(p.name == "pipeline.py" for p in files)
    bad = {
        str(p.relative_to(ROOT)): sorted(
            m for m in _imported_modules(p) if m.split(".")[:2] == ["praxis", "serve"]
        )
        for p in files
    }
    assert {k: v for k, v in bad.items() if v} == {}


def test_the_rest_needs_no_server_package() -> None:
    packages = _server_packages()
    assert SERVER_ONLY <= packages
    files = [p for p in _python_files() if not _server_part(p)]
    bad = {
        str(p.relative_to(ROOT)): sorted(
            m for m in _imported_modules(p) if m.split(".")[0] in packages
        )
        for p in files
    }
    assert {k: v for k, v in bad.items() if v} == {}


def test_server_packages_are_an_optional_extra() -> None:
    project = _project()
    assert not _names(project["dependencies"]) & SERVER_ONLY
    assert SERVER_ONLY - {"starlette"} <= _names(
        project["optional-dependencies"]["serve"]
    )
