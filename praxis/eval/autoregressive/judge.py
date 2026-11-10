"""The VLM judge of the autoregressive eval (GPT-6 Astra, OpenAI-compatible API).

The judge reads one rollout video and answers the questions of a fixed JSON
schema. The fields of the score are ``target_object_acted_on``,
``target_destination_used`` and ``action_correct`` (full adherence, see
:mod:`praxis.eval.autoregressive.score`).

One request holds:

* the system prompt of the set: ``judge_system_droid.txt`` (two views);
* the question (:func:`question`): the scene, the cameras, the task and the
  adherence targets of the rollout row;
* 12 evenly spaced frames, each a labelled grid of the camera tiles at 1.5x
  (:func:`labelled_grid`, JPEG quality 85), then the last frame at 2.25x.

The judge reads 189 frames at 15 fps on the judged canvas of the set
(:func:`judged_frames`): 832x224, one 416-px tile per camera.

The request is the request of the published results, byte for byte, less
the temperature (``tests/test_eval_autoregressive.py``). It sets no
temperature: the current judge model takes only its default, and the
published runs used 0. The answer is strict JSON (``response_format``
json_schema, ``judge_schema_<set>.json``), at most 8000 completion tokens.

The time-to-success judge (:func:`tts_request`) runs on the fully adherent
clips only. It reads 24 evenly spaced frames and gives ``success_time_s``,
the first frame time at which the task is visibly done.

The client reads ``ASTRA_API_KEY``, ``ASTRA_BASE_URL`` (an OpenAI-compatible
endpoint, for example ``https://api.openai.com/v1``) and ``ASTRA_MODEL``
(default ``gpt-6-astra``).

The tile labels use the font DejaVu Sans Bold. The published requests used
``/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`` of fonts-dejavu-core
2.37 (SHA-256 :data:`FONT_SHA256`). ``PRAXIS_JUDGE_FONT`` names another font
file. Another font gives other image bytes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import os
import re
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from praxis.eval.autoregressive import sets

HERE = Path(__file__).parent
N_FRAMES = 12
FRAME_SCALE = 1.5
LAST_SCALE = 2.25
JPEG_QUALITY = 85
TTS_FRAMES = 24
MAX_TOKENS = 8000
TTS_MAX_TOKENS = 4000
RETRIES = 6
#: HTTP statuses that a retry does not fix: a bad key (401), no access (403),
#: an unknown model or endpoint (404).
FATAL_STATUS = (401, 403, 404)
#: The judged clip: 189 frames at 15 fps.
CLIP_FRAMES = 189
CLIP_FPS = 15.0
DEFAULT_MODEL = "gpt-6-astra"
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
FONT_SHA256 = "5c1247acef7f2b8522a31742c76d6adcb5569bacc0be7ceaa4dc39dd252ce895"

#: The ``Source:`` token of the question of each set. The tokens are the
#: ones of the published requests; they stay as they are so that the
#: requests stay byte-identical. The camera names, the judge prompt and the
#: tile sizes come from :data:`praxis.eval.autoregressive.sets.SETS`. The YAM
#: sets add their tokens when they ship.
SOURCES: dict[str, str] = {
    "droid": "droid",
}
ADHERENCE_KEYS = ("target_object", "destination", "action", "distractors")
#: The rollout fields of the feature section.
FEATURE_KEYS = (
    "feature",
    "level",
    "required_count",
    "count_mode",
    "relation",
    "container",
    "distance_confounded",
    "target_instances",
    "count_filter",
    "n_present",
    "relation_applies_to",
)


def _features(set_name: str) -> bool:
    """True if the question of the set has the feature section (``yam``)."""
    return sets.SETS[set_name].judge == "yam"


def system_prompt(set_name: str) -> str:
    kind = sets.SETS[set_name].judge
    return (HERE / f"judge_system_{kind}.txt").read_text()


def schema(set_name: str) -> dict:
    kind = sets.SETS[set_name].judge
    return json.loads((HERE / f"judge_schema_{kind}.json").read_text())


def tts_system_prompt() -> str:
    return (HERE / "judge_tts_system.txt").read_text()


def tts_schema() -> dict:
    return json.loads((HERE / "judge_tts_schema.json").read_text())


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def read_video(
    path: str | Path, fps_hint: float = 15.0
) -> tuple[list[np.ndarray], float]:
    """Decode all frames of an mp4 to RGB uint8 ``(H, W, 3)``. Return frames and fps."""
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate) if st.average_rate else fps_hint
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return frames, fps


@cache
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    env = os.environ.get("PRAXIS_JUDGE_FONT")
    for p in (env, *FONT_PATHS) if env else FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def labelled_grid(
    frame: np.ndarray,
    widths: list[int],
    labels: tuple[str, ...],
    caption: str,
    scale: float,
) -> Image.Image:
    """Return one time step as labelled camera tiles with white separators.

    ``widths`` are the tile widths of the canvas, left to right. A canvas of
    another width is one tile labelled ``view``. Each tile is resized by
    ``scale`` (bicubic) and gets the header ``CAMERA k: name``. The caption
    goes under the tiles.
    """
    h, w, _ = frame.shape
    if sum(widths) != w:
        widths, labels = [w], ("view",)
    fs = max(9, round(8 * scale))
    bar, top = max(3, round(4 * scale)), fs + max(3, round(3 * scale))
    font = _font(fs)
    xs = np.cumsum([0, *widths])
    tw = [round(x * scale) for x in widths]
    th = round(h * scale)
    canvas = Image.new(
        "RGB", (sum(tw) + bar * (len(widths) - 1), th + 2 * top), (255, 255, 255)
    )
    d = ImageDraw.Draw(canvas)
    x = 0
    for i, (x0, x1) in enumerate(itertools.pairwise(xs)):
        t = Image.fromarray(frame[:, x0:x1])
        if scale != 1:
            t = t.resize((tw[i], th), Image.Resampling.BICUBIC)
        canvas.paste(t, (x, top))
        label = labels[i] if i < len(labels) else ""
        d.text((x + 4, 1), f"CAMERA {i + 1}: {label}", fill=(0, 0, 0), font=font)
        x += tw[i] + bar
    d.text((4, th + top + 1), caption, fill=(0, 0, 0), font=font)
    return canvas


def _grid(frame: np.ndarray, set_name: str, caption: str, scale: float) -> Image.Image:
    spec = sets.SETS[set_name]
    views = spec.judge_views
    return labelled_grid(
        frame, [spec.praxis_tile_wh[0]] * len(views), views, caption, scale
    )


def judged_frames(
    frames: list[np.ndarray], fps: float, set_name: str
) -> list[np.ndarray]:
    """Check a clip and return its frames on the judged canvas of the set.

    The clip must have :data:`CLIP_FRAMES` frames at :data:`CLIP_FPS`. A
    frame on the judged canvas (``SetSpec.judge_canvas_wh``) is kept as it
    is. A frame on the native canvas of a set whose tiles differ from the
    Praxis-1 tiles gets each tile resized (:func:`sets.resize_tiles`). Any
    other clip raises ``ValueError``.
    """
    spec = sets.SETS[set_name]
    if len(frames) != CLIP_FRAMES or abs(fps - CLIP_FPS) > 1e-3:
        raise ValueError(
            f"{set_name}: a clip of {len(frames)} frames at {fps} fps; the judge "
            f"reads {CLIP_FRAMES} frames at {CLIP_FPS:g} fps"
        )
    h, w = frames[0].shape[:2]
    if any(f.shape[:2] != (h, w) for f in frames):
        raise ValueError(f"{set_name}: the frames of a clip differ in size")
    if (w, h) == spec.judge_canvas_wh:
        return frames
    if (w, h) == spec.canvas_wh:
        return [sets.resize_tiles(f, spec) for f in frames]
    jw, jh = spec.judge_canvas_wh
    raise ValueError(f"{set_name}: a {w}x{h} clip; the judge reads {jw}x{jh}")


def jpeg_url(im: Image.Image, quality: int = JPEG_QUALITY) -> str:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _image(im: Image.Image) -> dict:
    return {"type": "image_url", "image_url": {"url": jpeg_url(im), "detail": "high"}}


def frame_indices(n_total: int, n: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_total - 1, n).round().astype(int))


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------


def split_subtasks(task: str) -> list[str]:
    parts = re.split(r",?\s+(?:and\s+)?then\s+", task.strip(), flags=re.IGNORECASE)
    return [p.strip().rstrip(".") for p in parts if p.strip()]


def adherence_spec(rollout: dict, set_name: str) -> dict:
    """Return the adherence targets (and, on the YAM sets, the feature fields).

    The destination distractor joins the distractors as
    ``"<name> [destination distractor]"``. ``distance_confounded`` is true for
    the medium and hard spatial prompts (the instances sit at different
    distances from the arms).
    """
    rec = dict(rollout)
    ds = [dict(d) for d in rec.get("distractors") or []]
    dd = rec.get("destination_distractor")
    if isinstance(dd, dict) and dd.get("name"):
        ds.append(
            {
                "name": f"{dd['name']} [destination distractor]",
                "differs_by": dd.get("differs_by", "unknown"),
            }
        )
    rec["distractors"] = ds
    spec = {k: rec[k] for k in ADHERENCE_KEYS if rec.get(k) not in (None, "", [])}
    if not _features(set_name):
        return spec
    rec["distance_confounded"] = rec.get("feature") == "spatial" and rec.get(
        "level"
    ) in ("medium", "hard")
    for k in FEATURE_KEYS:
        if rec.get(k) not in (None, "", []):
            spec[k] = rec[k]
    spec.setdefault("distractors", [])
    return spec


def _base_adherence_text(spec: dict) -> str:
    if not spec:
        return (
            "Adherence targets: not provided; derive the target object, the "
            "destination (if any) and the action verb from the task text."
        )
    lines = ["Adherence targets (check the video against EXACTLY these):"]
    if "target_object" in spec:
        lines.append(f"- target object: {spec['target_object']}")
    if "destination" in spec:
        lines.append(f"- destination: {spec['destination']}")
    if "action" in spec:
        lines.append(f"- action: {spec['action']}")
    if "distractors" in spec:
        ds = [
            d if isinstance(d, dict) else {"name": str(d)} for d in spec["distractors"]
        ]
        lines.append(
            "- distractors present: "
            + "; ".join(
                f"{d.get('name')} (differs by {d.get('differs_by', 'unknown')})"
                for d in ds
            )
        )
    return "\n".join(lines)


def _instances(spec: dict, scene: dict) -> list[dict]:
    """The identical instances with their top-view centre px, from the scene row."""
    ctr = scene.get("top_center_px") or {}
    names = list(scene.get("twins") or [])
    if spec.get("relation_applies_to") == "destination":
        names = [n for n in (spec.get("destination"),) if n]
        names += [
            d["name"].replace(" [destination distractor]", "")
            for d in spec.get("distractors", [])
            if isinstance(d, dict) and "[destination distractor]" in str(d.get("name"))
        ]
        names = list(dict.fromkeys(names))
    by = {o["name"]: o for o in scene.get("objects") or []}
    return [
        {
            "name": n,
            "top_px": ctr.get(n),
            "rests_in": by.get(n, {}).get("rests_in"),
            "state": by.get(n, {}).get("state"),
        }
        for n in names
        if n in ctr
    ]


def feature_text(spec: dict, scene: dict) -> str:
    """Return the feature section of a YAM question ("" without a feature)."""
    f = spec.get("feature")
    if not f:
        return ""
    lines = [f"FEATURE: {f} (difficulty {spec.get('level', '?')})."]
    if f == "count":
        n = spec.get("required_count")
        mode = spec.get("count_mode", "exact")
        filt = spec.get("count_filter") or {}
        kind = str(spec.get("target_object", "")).replace("_", " ")
        dest = spec.get("destination")
        if mode == "all":
            lines.append(
                f"Count question: the task asks for ALL {spec.get('n_present', n)} "
                f"{kind} items to end in/on {dest} (required count {n}, mode all)."
            )
        else:
            lines.append(
                f"Count question: the task asks for EXACTLY {n} {kind} item(s) in/on "
                f"{dest} (mode exact; {spec.get('n_present', '?')} present). More "
                f"than {n} is wrong."
            )
        if filt:
            lines.append(
                f"Only count items of colour {filt.get('colour')}; items of other "
                "colours do not count."
            )
        if spec.get("target_instances"):
            lines.append(
                "Countable instances: " + ", ".join(spec["target_instances"]) + "."
            )
        lines.append(
            "In the LAST frame, how many of these items are in/on the destination? "
            "Fill count_in_destination and count_correct; moved_instance and "
            "relation_correct = null."
        )
        return "\n".join(lines)
    rel, cont = spec.get("relation"), spec.get("container")
    if f in ("spatial", "containment") or rel or cont:
        if rel:
            lines.append(
                f"Identifying relation: {rel} (applies to the "
                f"{spec.get('relation_applies_to', 'object')})."
            )
        if cont:
            where = ("on the " + cont) if cont == "table" else "in/on " + cont
            lines.append(f"The target is the instance resting {where}.")
        inst = _instances(spec, scene)
        if inst:
            lines.append(
                "Instances that look identical apart from this (top-camera pixel "
                "centre in the 416x256 top tile, x right / y down):"
            )
            for i in inst:
                extra = ", ".join(
                    x
                    for x in (
                        f"resting {i['rests_in']}" if i["rests_in"] else "",
                        f"state {i['state']}" if i["state"] else "",
                    )
                    if x
                )
                lines.append(
                    f"  - {i['name']} at {i['top_px']}"
                    + (f" ({extra})" if extra else "")
                )
        if spec.get("distance_confounded"):
            lines.append(
                "Note: the instances are at different distances from the arms "
                "(distance-confounded prompt)."
            )
        lines.append(
            "Which instance did the robot move (moved_instance)? Is it the one the "
            "relation/state names (relation_correct)? count_in_destination and "
            "count_correct = null."
        )
        return "\n".join(lines)
    lines.append(
        "No extra feature question: moved_instance, relation_correct, "
        "count_in_destination, count_correct = null."
    )
    return "\n".join(lines)


def adherence_text(rollout: dict, scene: dict, set_name: str) -> str:
    spec = adherence_spec(rollout, set_name)
    base = _base_adherence_text({k: v for k, v in spec.items() if k in ADHERENCE_KEYS})
    if not _features(set_name):
        return base
    ft = feature_text(spec, scene)
    return base + ("\n" + ft if ft else "")


def _cameras(set_name: str) -> str:
    views = sets.SETS[set_name].judge_views
    return (
        f"Each frame tiles {len(views)} cameras, left to right: "
        + "; ".join(f"camera {i + 1} = {v}" for i, v in enumerate(views))
        + "."
    )


def question(
    rollout: dict, scene: dict, set_name: str, n_shown: int, n_total: int
) -> str:
    task = rollout["prompt"]
    sub = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(split_subtasks(task)))
    desc = scene.get("description") or "(not described)"
    return (
        f"Scene: {desc}\nSource: {SOURCES[set_name]}. {_cameras(set_name)}\n"
        f"Task given to the robot: {task}\nSubtasks in order:\n{sub}\n"
        f"{adherence_text(rollout, scene, set_name)}\n\n"
        f"You get {n_shown} evenly spaced frames of the {n_total}-frame clip in "
        "order, then the last frame at higher resolution. Judge each subtask, the "
        "overall success, physical plausibility, and prompt adherence."
    )


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def request(
    rollout: dict,
    scene: dict,
    set_name: str,
    frames: list[np.ndarray],
    fps: float,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Return the chat-completions arguments of the adherence judge.

    ``frames`` go through :func:`judged_frames` first.
    """
    frames = judged_frames(frames, fps, set_name)
    n_total = len(frames)
    idx = frame_indices(n_total, N_FRAMES)
    content: list[dict] = [
        {"type": "text", "text": question(rollout, scene, set_name, len(idx), n_total)}
    ]
    for i in idx:
        cap = f"frame {i + 1}/{n_total}  t={i / fps:.1f}s"
        content.append({"type": "text", "text": f"t={i / fps:.1f}s"})
        content.append(_image(_grid(frames[i], set_name, cap, FRAME_SCALE)))
    cap = (
        f"LAST FRAME (end state) frame {n_total}/{n_total}  "
        f"t={(n_total - 1) / fps:.1f}s"
    )
    content.append(
        {"type": "text", "text": "LAST FRAME (end state), higher resolution:"}
    )
    content.append(_image(_grid(frames[-1], set_name, cap, LAST_SCALE)))
    return {
        "model": model,
        "max_completion_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt(set_name)},
            {"role": "user", "content": content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "robot_video_judgement",
                "strict": True,
                "schema": schema(set_name),
            },
        },
    }


def tts_request(
    rollout: dict,
    scene: dict,
    set_name: str,
    frames: list[np.ndarray],
    fps: float,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Return the chat-completions arguments of the time-to-success judge.

    ``frames`` go through :func:`judged_frames` first.
    """
    frames = judged_frames(frames, fps, set_name)
    n_total = len(frames)
    idx = frame_indices(n_total, TTS_FRAMES)
    q = (
        f"Scene: {scene.get('description', '')}\n{_cameras(set_name)}\n"
        f"Task given to the robot: {rollout['prompt']}\n"
        f"{adherence_text(rollout, scene, set_name)}\n\n"
        "IGNORE any instruction above about count_in_destination / relation_correct "
        "/ moved_instance; use the instance list only to tell identical instances "
        "apart.\n"
        f"You get {len(idx)} evenly spaced frames of the {n_total}-frame clip "
        f"({(n_total - 1) / fps:.1f} s). This clip was judged a success. At which "
        "frame time is the target first grasped, and at which frame time is the "
        "task first visibly done?"
    )
    content: list[dict] = [{"type": "text", "text": q}]
    for i in idx:
        content.append({"type": "text", "text": f"t={i / fps:.2f}s"})
        content.append(
            _image(_grid(frames[i], set_name, f"t={i / fps:.2f}s", FRAME_SCALE))
        )
    return {
        "model": model,
        "max_completion_tokens": TTS_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": tts_system_prompt()},
            {"role": "user", "content": content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "time_to_success",
                "strict": True,
                "schema": tts_schema(),
            },
        },
    }


def request_sha256(req: dict) -> str:
    """SHA-256 of the canonical JSON of a request (for parity checks)."""
    return hashlib.sha256(
        json.dumps(req, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


def client():
    """Return an OpenAI client from ``ASTRA_BASE_URL`` and ``ASTRA_API_KEY``."""
    from openai import OpenAI

    return OpenAI(
        base_url=os.environ["ASTRA_BASE_URL"],
        api_key=os.environ["ASTRA_API_KEY"],
        timeout=600,
        max_retries=0,
    )


def parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def call(
    req: dict,
    api=None,
    retries: int = RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Send a request and return the parsed answer, with retries and backoff.

    If the endpoint refuses the json_schema format (HTTP 400 or 422, or no
    status, with ``response_format`` or ``json_schema`` in the error), the
    call falls back to ``{"type": "json_object"}`` once. An error with the
    status of :data:`FATAL_STATUS` (a bad key, no access, an unknown model)
    or another 400 raises ``RuntimeError`` at once, without retries. The
    answer gets ``_judge`` (model, usage, attempts, the SHA-256 of the
    request).
    """
    api = api or client()
    req = dict(req)
    sha = request_sha256(req)
    err: Exception | None = None
    t0 = time.monotonic()
    for attempt in range(retries):
        try:
            r = api.chat.completions.create(**req)
            out = parse_json(r.choices[0].message.content or "")
            u = r.usage
            out["_judge"] = {
                "model": req["model"],
                "request_sha256": sha,
                "tokens_in": getattr(u, "prompt_tokens", None),
                "tokens_out": getattr(u, "completion_tokens", None),
                "attempts": attempt + 1,
                "seconds": round(time.monotonic() - t0, 1),
            }
            return out
        except Exception as e:
            err = e
            status = getattr(e, "status_code", None)
            schema_refused = (
                status in (None, 400, 422)
                and req.get("response_format", {}).get("type") == "json_schema"
                and ("response_format" in str(e) or "json_schema" in str(e))
            )
            if schema_refused:
                req["response_format"] = {"type": "json_object"}
            elif status in FATAL_STATUS or status == 400:
                raise RuntimeError(
                    f"judge failed without a retry (HTTP {status}, check "
                    f"ASTRA_API_KEY, ASTRA_BASE_URL and ASTRA_MODEL={req['model']}): "
                    f"{type(e).__name__}: {str(e)[:300]}"
                ) from e
            sleep(min(60, 4 * 2**attempt))
    raise RuntimeError(f"judge failed: {type(err).__name__}: {str(err)[:300]}")
