"""The evaluation sets of the autoregressive eval: load, verify, describe.

The sets are in the Hugging Face dataset ``runwayml/praxis-1-eval`` under
``autoregressive/<set>/``. :data:`REVISION` pins the commit of the dataset.
:func:`load_set` reads a set from a local copy or downloads that commit, and
verifies every file before it returns:

* ``rollouts.jsonl`` and ``scenes.jsonl`` must have the SHA-256 of
  :data:`SETS`.
* Every first frame (``frames/<scene>.png`` and the second exterior view
  ``frames_ext2/`` of ``droid``) must have the SHA-256 of its scene row.

``frames_ext2/<scene>.png`` is optional data of ``droid``: the episode's
real second exterior frame with the same objects inserted, one 416x224
tile. Praxis-1 does not read it; the baselines with a second exterior
camera input were given it (:meth:`EvalSet.second_exterior`).

A mismatch raises :class:`SetError`. Thus a result names one exact set.

Each set fixes its rig, its views and its canvas (:class:`SetSpec`). The
first frame of a scene is one canvas: the views side by side, one tile each.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ID = "runwayml/praxis-1-eval"
REVISION = "2e95455bc8497d6fe2c00da759e863e186a7d9f2"


class SetError(ValueError):
    """A file of a set is missing or does not have its pinned SHA-256."""


@dataclass(frozen=True)
class SetSpec:
    """The fixed description of one set.

    ``tile_wh`` is the ``(width, height)`` of one tile of the frames of the
    set. ``judge_views`` are the camera names that the judge reads, left to
    right. ``judge`` selects the judge prompt (``droid``: two views).
    ``praxis_tile_wh`` is
    the tile size of the Praxis-1 canvas of the set. It is also the tile
    size of the judged videos (:attr:`judge_canvas_wh`).
    """

    name: str
    rig: str
    views: tuple[str, ...]
    judge_views: tuple[str, ...]
    tile_wh: tuple[int, int]
    praxis_tile_wh: tuple[int, int]
    judge: str
    rollouts_sha256: str
    scenes_sha256: str
    n_rollouts: int
    n_scenes: int

    @property
    def canvas_wh(self) -> tuple[int, int]:
        return self.tile_wh[0] * len(self.views), self.tile_wh[1]

    @property
    def judge_canvas_wh(self) -> tuple[int, int]:
        """The canvas of the judged videos (832x224 on ``droid``)."""
        return self.praxis_tile_wh[0] * len(self.judge_views), self.praxis_tile_wh[1]


SETS: dict[str, SetSpec] = {
    "droid": SetSpec(
        name="droid",
        rig="single_arm",
        views=("exterior", "wrist"),
        judge_views=("exterior", "wrist"),
        tile_wh=(416, 224),
        praxis_tile_wh=(416, 224),
        judge="droid",
        rollouts_sha256="661296e8bbc06c6c718080fdadaf9f0d87907bdd6cacb083a819cd1ef76e5277",
        scenes_sha256="c400c99fe03e9d1c9b85efb7de24a000bc3470626616f10ade0f5acf41dd9e68",
        n_rollouts=104,
        n_scenes=8,
    ),
}


@dataclass
class EvalSet:
    """One verified set: its spec, its directory, its scenes and rollouts."""

    spec: SetSpec
    root: Path
    scenes: dict[str, dict]
    rollouts: list[dict]

    def frame_path(self, scene: str) -> Path:
        return self.root / self.scenes[scene]["frame"]

    def first_frame(self, scene: str) -> np.ndarray:
        """Return the first-frame canvas ``(H, W, 3)`` uint8 of a scene."""
        return load_png(self.frame_path(scene))

    def second_exterior(self, scene: str) -> np.ndarray | None:
        """Return the second exterior view ``(H, W, 3)`` uint8 (``droid``), or None."""
        f = self.scenes[scene].get("frame_ext2")
        return None if f is None else load_png(self.root / f)

    def rollout(self, rollout_id: str) -> dict:
        for r in self.rollouts:
            if r["id"] == rollout_id:
                return r
        raise KeyError(rollout_id)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _check(path: Path, want: str) -> None:
    if not path.is_file():
        raise SetError(f"{path} is missing")
    got = sha256_file(path)
    if got != want:
        raise SetError(f"{path}: SHA-256 {got}, expected {want}")


def download_set(
    name: str, revision: str = REVISION, local_dir: str | Path | None = None
) -> Path:
    """Download ``autoregressive/<name>/`` at ``revision`` and return its folder.

    The download uses the normal Hugging Face login (``HF_TOKEN`` or
    ``hf auth login``).
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"autoregressive/{name}/*"],
        local_dir=None if local_dir is None else str(local_dir),
    )
    return Path(str(path)) / "autoregressive" / name


def load_set(name: str, root: str | Path | None = None) -> EvalSet:
    """Return the verified set ``name``.

    ``root`` is a local folder: the set folder itself (it holds
    ``rollouts.jsonl``), or a copy of the dataset (it holds
    ``autoregressive/<name>/``). ``None`` downloads the pinned revision.
    """
    if name not in SETS:
        raise SetError(f"unknown set {name!r}; the sets are {sorted(SETS)}")
    spec = SETS[name]
    if root is None:
        folder = download_set(name)
    else:
        folder = Path(root)
        if not (folder / "rollouts.jsonl").exists():
            folder = folder / "autoregressive" / name
    _check(folder / "rollouts.jsonl", spec.rollouts_sha256)
    _check(folder / "scenes.jsonl", spec.scenes_sha256)
    scenes = {s["scene"]: s for s in _read_jsonl(folder / "scenes.jsonl")}
    rollouts = _read_jsonl(folder / "rollouts.jsonl")
    if len(scenes) != spec.n_scenes or len(rollouts) != spec.n_rollouts:
        raise SetError(
            f"{name}: {len(scenes)} scenes and {len(rollouts)} rollouts, expected "
            f"{spec.n_scenes} and {spec.n_rollouts}"
        )
    for s in scenes.values():
        _check(folder / s["frame"], s["sha256"])
        for view in (s.get("frames_640") or {}).values():
            _check(folder / view["file"], view["sha256"])
        if s.get("frame_ext2"):
            _check(folder / s["frame_ext2"], s["sha256_ext2"])
    for r in rollouts:
        if r["scene"] not in scenes:
            raise SetError(f"rollout {r['id']} names the unknown scene {r['scene']}")
    return EvalSet(spec, folder, scenes, rollouts)


def praxis_canvas(es: EvalSet, scene: str) -> np.ndarray:
    """Return the first-frame canvas of a scene at the Praxis-1 tile size.

    A set whose tiles differ from the Praxis-1 tiles gets each tile resized
    (PIL bicubic); ``droid`` keeps its frames.
    """
    return resize_tiles(es.first_frame(scene), es.spec)


def resize_tiles(frame: np.ndarray, spec: SetSpec) -> np.ndarray:
    """Resize each ``tile_wh`` tile of a canvas to ``praxis_tile_wh`` (PIL bicubic).

    A set whose two tile sizes are equal returns ``frame`` as it is.
    """
    if spec.tile_wh == spec.praxis_tile_wh:
        return frame
    tw, _ = spec.tile_wh
    tiles = [
        np.asarray(
            Image.fromarray(
                np.ascontiguousarray(frame[:, i * tw : (i + 1) * tw])
            ).resize(spec.praxis_tile_wh, Image.Resampling.BICUBIC)
        )
        for i in range(len(spec.views))
    ]
    return np.concatenate(tiles, axis=1)
