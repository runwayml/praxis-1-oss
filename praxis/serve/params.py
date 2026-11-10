"""Session parameters, rollout ids, seeds, the rig-side pipeline declaration and
the code version of the handshake.

A client sets the parameters of one session in the query of the session URL,
for example ``/api/v1/session?sample_seed=0&num_steps=4``. Each value is a
JSON literal: ``5`` is an int, ``3.0`` a float, ``true`` a bool and ``null``
None. A value that is not valid JSON stays a string. When a key occurs two
times, the last value applies.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path

# The keys that a client can set per session.
SESSION_PARAMS: frozenset[str] = frozenset(
    {
        "history_frames",
        "stride",
        "pad_start",
        "stack_keys",
        "return_server_timing",
        "exec_horizon",
        "max_chunks_per_episode",
        "context_mode",
        "max_context_chunks",
        "carry_mode",
        "smoothing",
        "sg_window",
        "sg_order",
        "cfg_scale",
        "cfg_video_scale",
        "cfg_batched",
        "num_steps",
        "action_steps",
        "t_shift",
        "solver",
        "sample_seed",
        "prompt",
        "prompt_style",
        "declare_stop_on_fault",
        "control_mode",
        "record_denoised",
        "compile",
        "skip_video_decoder",
        "schedule",
        "sample_fps",
        "max_in_flight",
        "rtc_stride",
        "rtc_rebase_decay",
        "rtc_vel_fade",
        "rtc_grip",
        "rtc_grip_rows",
        "rtc_reach_max_m",
    }
)
# Keys that the server reads itself. They do not go to the policy.
RESERVED_PARAMS: frozenset[str] = frozenset({"rollout_id"})

# The version of the Positronic wire protocol that this server speaks.
POSITRONIC_WIRE_REF = "d5f2a47"

_ROLLOUT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def literal_value(raw: str) -> object:
    """Return the JSON value of ``raw``, or ``raw`` if it is not valid JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_session_params(raw_params: Mapping[str, str]) -> dict[str, object]:
    """Convert the query parameters to session overrides.

    An unknown key raises ``ValueError``. ``stack_keys`` becomes a tuple: from
    a JSON list, or from a comma list.
    """
    unknown = sorted(set(raw_params) - SESSION_PARAMS - RESERVED_PARAMS)
    if unknown:
        raise ValueError(
            f"Unknown session params {unknown}. Tunable per session: "
            f"{sorted(SESSION_PARAMS)}. A different checkpoint goes in the path "
            "(/api/v1/session/<model_id>)."
        )
    overrides: dict[str, object] = {}
    for key, raw in raw_params.items():
        if key in RESERVED_PARAMS:
            continue
        value = literal_value(raw)
        if key == "stack_keys":
            value = (
                tuple(value)
                if isinstance(value, list)
                else tuple(k.strip() for k in str(value).split(",") if k.strip())
            )
        overrides[key] = value
    return overrides


def parse_on_off(v: object) -> bool | None:
    """Return the bool of a flag value, or None for an absent value.

    The value can be a bool, a number, or one of ``on``, ``off``, ``true``,
    ``false``, ``yes``, ``no``, ``1``, ``0``, ``none`` and an empty string.
    Other values raise ``ValueError``.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int | float):
        return bool(v)
    t = str(v).strip().lower()
    if t in ("on", "true", "1", "yes"):
        return True
    if t in ("off", "false", "0", "no", "none", ""):
        return False
    raise ValueError(f"expected on|off, got {v!r}")


def parse_opt_int(v: str | float | None) -> int | None:
    """Return an int, or None for None, ``""``, ``none``, ``auto`` and ``null``."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("", "none", "auto", "null"):
        return None
    return int(v)


def parse_skip_mode(v: object) -> bool | None:
    """Return the ``skip_video_decoder`` value: None for ``auto``, else a bool."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("auto", "default"):
        return None
    return parse_on_off(v)


def parse_reach(value: object, default_m: float, arms: int = 2) -> tuple[float, ...]:
    """Return the ``rtc_reach_max_m`` radii: one per arm, or empty for no clamp.

    ``auto``, ``on`` and None give ``default_m`` on every arm. ``off``, ``0``
    and an empty string turn the clamp off. ``R`` gives R m on every arm and
    ``L,R`` one radius per arm. A radius out of 0.2..1.0 m raises
    ``ValueError``.
    """
    text = "auto" if value is None else str(value).strip().lower()
    if text in ("auto", "on", "true", "1"):
        return (float(default_m),) * arms
    if text in ("", "0", "off", "none", "false", "null"):
        return ()
    vals = tuple(float(v) for v in text.split(","))
    radii = vals * arms if len(vals) == 1 else vals
    if len(radii) != arms or not all(
        math.isfinite(r) and 0.2 <= r <= 1.0 for r in radii
    ):
        raise ValueError(
            "rtc_reach_max_m must be auto, off, one radius or one per arm "
            f"(left,right), 0.2..1.0 m, got {value!r}"
        )
    return radii


def resolve_rollout_id(value: str | None) -> tuple[str, str]:
    """Return the rollout id of a session and its source.

    A missing or empty id gives a new id (``bridge_generated``). An id that is
    not one safe path component gives a new id
    (``bridge_generated_after_reject``). Else the client id applies
    (``client_supplied``).
    """
    if not value:
        return f"praxis-{uuid.uuid4().hex[:12]}", "bridge_generated"
    if value in (".", "..") or _ROLLOUT_ID_RE.fullmatch(value) is None:
        return f"praxis-{uuid.uuid4().hex[:12]}", "bridge_generated_after_reject"
    return value, "client_supplied"


def resolve_sample_seed(
    session_value: str | float | None, env_value: str | None
) -> tuple[int, str]:
    """Return the base seed of a session and its source.

    The session parameter ``sample_seed`` applies first (``session_param``),
    then the deploy value ``PRAXIS_SAMPLE_SEED`` (``env``). Without both, each
    session draws a new 31-bit seed (``rotated``). Chunk ``k`` of the session
    samples with the seed ``base + k``.
    """
    if session_value is not None and str(session_value).strip().lower() not in (
        "",
        "none",
        "null",
    ):
        return int(session_value), "session_param"
    if env_value is not None and env_value.strip() != "":
        return int(env_value), "env"
    return secrets.randbits(31), "rotated"


def git_commit(root: Path) -> str:
    """Return the git commit of the checkout at ``root``, or ``unknown``.

    ``-dirty`` marks changes that are not committed. A directory that is not
    the top of a checkout gives ``unknown``, also when a parent directory is
    in another repository.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
            return "unknown"
        sha = git("rev-parse", "HEAD")
        changes = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return sha + ("-dirty" if changes else "")


def frame_offsets_sec(
    history_frames: int, stride: int, fps: float, half_frame: bool = False
) -> list[float]:
    """Return the time offsets of the frame stack in seconds, oldest first, 0.0 last.

    With ``half_frame``, every past offset is half a frame period later. The
    client stack picks the latest sample at or before each offset, and frames
    arrive on the control ticks: an offset on a tick picks the previous tick
    when the clock jitters early, half a period after it picks that tick.
    """
    frames = [*range(0, history_frames, stride), history_frames]
    half = 0.5 / fps if half_frame else 0.0
    return [0.0 if f == 0 else -f / fps + half for f in reversed(frames)]


def local_stack(
    keys: tuple[str, ...],
    history_frames: int,
    fps: float,
    width: int,
    height: int,
    stop_on_fault: bool,
    restrict_first: bool = False,
    half_frame: bool | None = None,
) -> dict:
    """Return the ``local_stack`` value of the handshake: the rig-side pipeline.

    The entries apply from the outside in. ``stop_on_fault`` stops the
    observations while an arm reports a fault. ``temporal_stack`` sends the
    last ``history_frames`` frames plus the current frame of each key (only
    with observed context). ``chunked_schedule`` plays the commands at their
    time offsets. ``restrict_image_size`` scales down each image to fit
    ``width`` x ``height``.

    ``restrict_first`` (the antialiased canvas of praxis-1-droid) puts
    ``restrict_image_size`` right after ``stop_on_fault``, so each frame is
    resized once on its own tick and the stack holds the small frames, and it
    targets each past stack frame half a frame period after its tick
    (:func:`frame_offsets_sec`). Without it the restrict comes last.
    ``half_frame`` (default: ``restrict_first``) sets the half-frame offsets
    on their own: the bimanual rig keeps the restrict last, as the YAM
    client does (the restrict costs the client a resize per tick), with
    half-frame offsets.
    """
    if half_frame is None:
        half_frame = restrict_first
    seq: list[dict] = [{"name": "stop_on_fault"}] if stop_on_fault else []
    restrict = {
        "name": "restrict_image_size",
        "args": {"width": int(width), "height": int(height)},
    }
    if restrict_first:
        seq.append(restrict)
    if history_frames > 0:
        seq.append(
            {
                "name": "temporal_stack",
                "args": {
                    "keys": list(keys),
                    "offsets_sec": frame_offsets_sec(
                        history_frames, 1, fps, half_frame=half_frame
                    ),
                    "pad_start": False,
                },
            }
        )
    seq.append({"name": "chunked_schedule"})
    if not restrict_first:
        seq.append(restrict)
    return {"seq": seq}
