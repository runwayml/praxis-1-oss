"""Profiling-only NVTX ranges (``PRAXIS_NVTX=1``; off by default, then every helper is a no-op).

Every op range is pushed with its full context, outermost first::

    praxis::<module>:<pass>  >  praxis::block:<i> | praxis::pre | praxis::post  >  praxis::<sub-module>  >  praxis::<op>

``module`` is enc, dec, ae or flow; ``pass`` is g (guided), u (unguided) or p (one launch serves both passes).
The context lives in a Python state that :func:`tag_gen` saves and restores around each resume of a
``forward_gen`` generator, so the lockstep pair driver (``layers.run_pair``) never leaves a range open across a
yield: each op range opens and closes inside one generator segment. The step and Euler-step ranges nest normally
around everything. The ranges only push and pop host-side markers; kernels, launch order, streams and outputs
are unchanged.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext

import torch

ON = os.environ.get("PRAXIS_NVTX", "0") == "1"
_NULL = nullcontext()
STATE = {"mod": None, "pass": None, "block": None}


def _push(name: str) -> None:
    torch.cuda.nvtx.range_push("praxis::" + name)


def _pop(n: int) -> None:
    for _ in range(n):
        torch.cuda.nvtx.range_pop()


@contextmanager
def _op(sub: str | None, op: str | None, block, mod, pas):
    n = 0
    if mod is not None:
        _push(f"{mod}:{pas}" if pas else mod)
        n += 1
    if block is not None:
        _push(block if isinstance(block, str) else f"block:{block}")
        n += 1
    if sub is not None:
        _push(sub)
        n += 1
    if op is not None:
        _push(op)
        n += 1
    try:
        yield
    finally:
        _pop(n)


def op(sub: str | None, op_name: str | None = None, *, block=..., pas=..., mod=...):
    """Context manager: one op range with its module/pass/block/sub-module context (overrides optional)."""
    if not ON:
        return _NULL
    s = STATE
    return _op(
        sub,
        op_name,
        s["block"] if block is ... else block,
        s["mod"] if mod is ... else mod,
        s["pass"] if pas is ... else pas,
    )


def rng(name: str):
    """A plain nested range (step, Euler step)."""
    if not ON:
        return _NULL
    return _plain(name)


@contextmanager
def _plain(name: str):
    _push(name)
    try:
        yield
    finally:
        _pop(1)


@contextmanager
def _ctx(**kw):
    old = dict(STATE)
    STATE.update(kw)
    try:
        yield
    finally:
        STATE.clear()
        STATE.update(old)


def ctx(**kw):
    """Set module / pass / block for the enclosed host code (no range of its own)."""
    if not ON:
        return _NULL
    return _ctx(**kw)


def set_block(b) -> None:
    if ON:
        STATE["block"] = b


def tag_gen(gen, mod: str, pas: str):
    """Wrap a ``forward_gen`` generator so its segments run with their own module/pass/block state."""
    if not ON:
        return gen
    return _TaggedGen(gen, {"mod": mod, "pass": pas, "block": "pre"})


class _TaggedGen:
    def __init__(self, gen, state):
        self.gen, self.state = gen, state

    def _run(self, fn, *a):
        outer = dict(STATE)
        STATE.clear()
        STATE.update(self.state)
        try:
            return fn(*a)
        finally:
            self.state = dict(STATE)
            STATE.clear()
            STATE.update(outer)

    def __iter__(self):
        return self

    def __next__(self):
        return self._run(self.gen.__next__)

    def send(self, v):
        return self._run(self.gen.send, v)

    def close(self):
        return self._run(self.gen.close)


def like(gen, **kw):
    """The state of a tagged generator's last segment (for the driver's feed-forward call), with overrides."""
    if not ON or not isinstance(gen, _TaggedGen):
        return _NULL
    st = dict(gen.state)
    st.update(kw)
    return _ctx(**st)


def pass_labels(n: int) -> list[dict]:
    """Per-input context overrides inside a paired call: guided then unguided."""
    if ON and n == 2 and STATE.get("pass") == "p":
        return [{"pass": "g"}, {"pass": "u"}]
    return [{} for _ in range(n)]


def pas_kw(label: dict) -> dict:
    return {"pas": label["pass"]} if "pass" in label else {}


def tag_current(gen):
    """Tag a generator with the current module/pass (run_solo of an untagged stack)."""
    if not ON or isinstance(gen, _TaggedGen):
        return gen
    return _TaggedGen(
        gen, {"mod": STATE["mod"], "pass": STATE["pass"], "block": STATE["block"]}
    )


def begin_euler(step: int) -> None:
    """Open the Euler-step range (closed by :func:`end_euler` at the end of the step's body)."""
    if ON:
        _push(f"euler:{step}")


def end_euler() -> None:
    if ON:
        _pop(1)
