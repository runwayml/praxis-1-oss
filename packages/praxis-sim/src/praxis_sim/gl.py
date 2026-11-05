"""OpenGL backend of the MuJoCo renderer. This module does not import MuJoCo.

MuJoCo reads ``MUJOCO_GL`` when Python imports it. On a Linux machine without
a display, ``egl`` is the backend that works. Thus :func:`prepare` sets
``MUJOCO_GL=egl`` on Linux when the variable is not set.

libglvnd selects the EGL vendor libraries when the EGL library loads. Some GPU
containers have the NVIDIA EGL library but not its glvnd vendor file. Then
EGL uses the Mesa software renderer (llvmpipe) and does not tell you. The
software renderer is approximately 20 times slower. In this condition,
:func:`prepare` writes a vendor file for the NVIDIA library to the cache
directory. Then it sets ``__EGL_VENDOR_LIBRARY_FILENAMES`` to that file and
to the vendor files of the system. The NVIDIA devices come first, and the
software device comes last.

``MUJOCO_EGL_DEVICE_ID`` selects the EGL device. It counts all physical GPUs.
It does not use ``CUDA_VISIBLE_DEVICES``. MuJoCo reads it when it makes the
first renderer. Thus you can set it after the import.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import sys
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

BACKENDS: tuple[str, ...] = ("egl", "osmesa", "glfw", "glx")
GLVND_VENDOR_DIRS: tuple[str, ...] = (
    "/etc/glvnd/egl_vendor.d",
    "/usr/share/glvnd/egl_vendor.d",
    "/usr/local/share/glvnd/egl_vendor.d",
)
NVIDIA_EGL_LIBRARY = "libEGL_nvidia.so.0"
NVIDIA_VENDOR_FILE = "10_nvidia_egl.json"
SOFTWARE_RENDERERS: tuple[str, ...] = ("llvmpipe", "softpipe", "swrast")
# MuJoCo and libglvnd read these variables when Python imports mujoco.
LOAD_TIME_VARIABLES: tuple[str, ...] = (
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
)


def _default_cache_dir() -> Path:
    """Return the cache directory of the simulation (no import of ``assets``)."""
    if os.environ.get("PRAXIS_SIM_CACHE"):
        return Path(os.environ["PRAXIS_SIM_CACHE"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "praxis" / "sim"


def _set_env(name: str, value: str, override: bool) -> None:
    """Set an environment variable. Keep a value that is set, unless ``override``."""
    current = os.environ.get(name)
    if current == value or (current is not None and not override):
        return
    if "mujoco" in sys.modules and name in LOAD_TIME_VARIABLES:
        warnings.warn(
            f"MuJoCo is already imported, so {name}={value!r} has no effect. "
            "Call praxis_sim.gl.prepare before the first import of mujoco.",
            stacklevel=3,
        )
        return
    os.environ[name] = value


def _system_vendor_files() -> list[Path]:
    """Return the EGL vendor files that libglvnd finds in its standard directories."""
    files: list[Path] = []
    for directory in GLVND_VENDOR_DIRS:
        path = Path(directory)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
    return files


def _nvidia_egl_loads() -> bool:
    """Return ``True`` when the dynamic loader finds the NVIDIA EGL library."""
    try:
        ctypes.CDLL(NVIDIA_EGL_LIBRARY)
    except OSError:
        return False
    return True


def add_nvidia_vendor(cache_dir: str | Path | None = None) -> Path | None:
    """Make libglvnd find the NVIDIA EGL library when its vendor file is missing.

    The function does nothing in these conditions:

    * ``__EGL_VENDOR_LIBRARY_FILENAMES`` or ``__EGL_VENDOR_LIBRARY_DIRS`` has
      a value.
    * The system has an NVIDIA vendor file.
    * The NVIDIA library does not load.

    It returns the path of the vendor file that it wrote, or ``None``.
    """
    if os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") or os.environ.get(
        "__EGL_VENDOR_LIBRARY_DIRS"
    ):
        return None
    if "mujoco" in sys.modules:
        return None  # too late: the EGL library is already loaded
    system = _system_vendor_files()
    if any("nvidia" in path.name.lower() for path in system):
        return None
    if not _nvidia_egl_loads():
        return None
    directory = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    path = directory / NVIDIA_VENDOR_FILE
    content = {
        "file_format_version": "1.0.0",
        "ICD": {"library_path": NVIDIA_EGL_LIBRARY},
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or json.loads(path.read_text()) != content:
            path.write_text(json.dumps(content))
    except OSError:
        logger.warning("Cannot write the EGL vendor file %s", path, exc_info=True)
        return None
    _set_env(
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        ":".join([str(path), *(str(p) for p in system)]),
        override=False,
    )
    return path


def prepare(
    backend: str | None = None,
    egl_device: int | None = None,
    fix_nvidia_vendor: bool = True,
    cache_dir: str | Path | None = None,
    override: bool = False,
) -> str | None:
    """Set the environment variables of the MuJoCo OpenGL backend.

    ``backend`` is one of :data:`BACKENDS`. ``None`` keeps ``MUJOCO_GL`` when
    it is set, and selects ``egl`` on Linux. A value that is set stays the
    same, unless ``override`` is ``True``. ``egl_device`` is the physical GPU
    index of the EGL renderer. The function returns the backend in use.
    """
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"Unknown OpenGL backend {backend!r}. Use one of {BACKENDS}.")
    if backend is None:
        backend = os.environ.get("MUJOCO_GL") or (
            "egl" if platform.system() == "Linux" else None
        )
        override = False
    if backend is None:
        return None
    _set_env("MUJOCO_GL", backend, override)
    if egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(egl_device))
    if os.environ.get("MUJOCO_GL") == "egl" and fix_nvidia_vendor:
        add_nvidia_vendor(cache_dir)
    return os.environ.get("MUJOCO_GL")


def renderer_name() -> str:
    """Return ``GL_RENDERER`` of the current OpenGL context.

    Call it after you make a ``mujoco.Renderer``, because that call makes its
    context current.
    """
    from OpenGL import GL

    try:
        name = GL.glGetString(GL.GL_RENDERER)
    except Exception:
        logger.warning("Cannot read GL_RENDERER", exc_info=True)
        return "unknown"
    if isinstance(name, bytes):
        return name.decode(errors="replace")
    return str(name) if name else "unknown"


def is_software_renderer(name: str) -> bool:
    """Return ``True`` when ``name`` is a CPU (software) OpenGL renderer."""
    return any(key in name.lower() for key in SOFTWARE_RENDERERS)
