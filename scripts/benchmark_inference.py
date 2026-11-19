"""Time the policy calls of praxis on a local GPU, per stage. No server is needed.

Usage:
    python scripts/benchmark_inference.py --config configs/serve_droid.yaml
    python scripts/benchmark_inference.py --config configs/serve_droid.yaml \\
        --set sampler.cfg_video_scale=1.0 --calls 20 --out outputs/bench --trace

The script loads the pipeline with the model, sampler and session settings of
the config (the serve config by default: the recommended settings), opens
sessions and runs policy calls on synthetic observations at the shapes of a
robot client: a first call with one frame, then calls with a stack of 25
frames (24 executed frames and the current one), two views of 416x234 (the
fit size of the antialiased checkpoint) or 512x288. Each session runs
``--calls`` calls, so the context depth goes 0, 1, ..., up to
``max_context_chunks``.

As on the policy server, each call skips the video decoder of the last Euler
step (``skip_video_decoder``, default on): the commands keep their bits and
the call returns no predicted video. ``--set session.skip_video_decoder=off``
times the full last step.

The ``--sessions`` staged sessions time each stage (:mod:`praxis.profiling`)
with a device synchronize around it. The sync stops the CPU from queuing the
next stage while the GPU runs this one, so one more plain session times each
call with a single sync around it: its ``total`` is the call time a server
sees. ``total_staged`` is the total of the staged calls. The records go to
``calls.jsonl``; the summary (p50, p90, max, first call of a session and
later calls apart) goes to ``summary.json`` with the GPU, the dtype and the
package versions. The first ``--warmup-calls`` calls of the run are not
counted: they compile the attention kernels of each context depth.

``--trace`` runs one more session under ``torch.profiler`` (its calls are
slower and not counted) and writes ``trace.json`` (Chrome trace format, opens in
Perfetto or ``chrome://tracing``) and a kernel table ``kernels.txt``. Each
stage is a named range in the trace. The stages are NVTX ranges too: for
Nsight Systems, run the script under ``nsys profile -t cuda,nvtx`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def process_age() -> float | None:
    """Return the seconds since the start of this process (Linux), or None."""
    try:
        stat = Path("/proc/self/stat").read_text()
        # Field 22, the start time in clock ticks after boot. The command name
        # (field 2) is in parentheses and can hold spaces.
        ticks = int(stat.rsplit(")", 1)[1].split()[19])
        boot = time.clock_gettime(time.CLOCK_BOOTTIME)
        return boot - ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


# The process age at this line, before torch is imported. Without /proc, the
# times of summary.json["startup"] count from this line.
_AGE0, _T0 = process_age(), time.monotonic()


def since_start() -> float:
    """Return the seconds since the start of the process."""
    return (_AGE0 or 0.0) + time.monotonic() - _T0


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.compile_cache import (  # noqa: E402
    CACHE_DIR_ENV,
    cache_env,
    configure_compile_cache,
)

# Before torch: the compile caches read their variables at import.
configure_compile_cache()

import torch  # noqa: E402

from praxis import single_arm  # noqa: E402
from praxis.checkpoint import load_config as checkpoint_config  # noqa: E402
from praxis.layers import Attention  # noqa: E402
from praxis.pipeline import PraxisPipeline  # noqa: E402
from praxis.profiling import (  # noqa: E402
    STAGES,
    StageTimer,
    instrument,
    summarize,
    summarize_calls,
    synthetic_obs,
)
from scripts.run_inference import build_pipeline, load_config  # noqa: E402

PROMPT = "Put the red block in the bowl."


def check_single_arm(checkpoint: str | Path) -> None:
    """Refuse a checkpoint that is not single-arm, from its ``config.json``.

    The check reads only the config, so it runs before the weight load.
    """
    embodiment = checkpoint_config(checkpoint)["embodiment"]
    if embodiment != "single_arm":
        raise SystemExit(
            f"The benchmark makes single-arm observations; the checkpoint is for "
            f"the {embodiment!r} rig."
        )


def configure_optimizations(
    pipe: PraxisPipeline, enabled: bool = True
) -> dict[str, bool]:
    """Set the inference optimizations for every session of this benchmark.

    The CUDA graphs are set up when the pipeline loads; the opt-out builds it
    without them, and the result reports whether they are installed.
    """
    from praxis import layers, vae

    layers.FUSED_ROPE = enabled
    layers.CACHE_DEVICE_TABLES = enabled
    # Off with --no-optimizations; else the module default
    # (PRAXIS_VAE_ENCODE_GRAPH=0 turns it off for an A/B run).
    vae.VAE_ENCODE_GRAPH = enabled and vae.VAE_ENCODE_GRAPH
    original = pipe.session

    def session(*args, **kwargs):
        result = original(*args, **kwargs)
        result.ctx_cache = enabled
        result.overlap_expert = enabled
        result.state.dedup_encodes = enabled
        # Session.__init__ set the default (CUDA, PRAXIS_NOISE_PREFETCH).
        result.noise_prefetch = enabled and result.noise_prefetch
        return result

    pipe.session = session
    return dict(
        FUSED_ROPE=enabled,
        CACHE_DEVICE_TABLES=enabled,
        ctx_cache=enabled,
        overlap_expert=enabled,
        dedup_encodes=enabled,
        VAE_ENCODE_GRAPH=vae.VAE_ENCODE_GRAPH,
        action_cuda_graphs=hasattr(pipe, "action_cuda_graphs"),
        nvfp4_moe_cuda_graphs=hasattr(pipe, "nvfp4_moe_cuda_graphs"),
    )


def view_size(pipe: PraxisPipeline) -> tuple[int, int]:
    """Return the ``(height, width)`` of the views that a client sends."""
    variant = pipe.variant
    if variant.antialias:
        return single_arm.fit_resize_size(variant.tile_size[1])
    # The client scales the native view down to fit 1024x288: 512x288.
    height, width = single_arm.NATIVE_VIEW
    scale = min(1.0, single_arm.CANVAS_H / height, single_arm.CANVAS_W / width)
    return round(height * scale), round(width * scale)


def environment(pipe: PraxisPipeline) -> dict:
    """Return the hardware, the dtype and the versions of the run."""
    device = pipe.device
    cuda = device.type == "cuda"
    return {
        "gpu": torch.cuda.get_device_name(device) if cuda else platform.processor(),
        "dtype": str(pipe.policy.dtype).removeprefix("torch."),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "attention_backend": pipe.variant.attention_backend,
        "attention_backend_effective": next(
            (
                m.effective_backend(device)
                for m in pipe.policy.modules()
                if isinstance(m, Attention)
            ),
            None,
        ),
        "routed_precision": getattr(pipe.policy, "routed_quantization", {}).get(
            "format", "bf16"
        ),
        "python": platform.python_version(),
    }


def skip_video_decoder(session: dict) -> bool:
    """Return the ``skip_video_decoder`` of the session settings, as the server.

    Absent or ``auto`` skips the video decoder of the last Euler step. The
    script reads the value itself: it must not import the server package.
    """
    value = session.get("skip_video_decoder")
    if value is None:
        return True
    if isinstance(value, bool | int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("auto", "default", "on", "true", "1", "yes"):
        return True
    if text in ("off", "false", "0", "no", "none", ""):
        return False
    raise SystemExit(f"skip_video_decoder: expected on|off|auto, got {value!r}")


def noise_prefetch_setting(session: dict) -> bool | None:
    """Return the ``noise_prefetch`` of the session settings, or None if absent.

    Absent keeps the session default. Any value other than on or off is
    refused, as for ``skip_video_decoder``.
    """
    value = session.get("noise_prefetch")
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("on", "true", "1", "yes"):
        return True
    if text in ("off", "false", "0", "no"):
        return False
    raise SystemExit(f"noise_prefetch: expected on|off, got {value!r}")


def settings(pipe: PraxisPipeline, session: dict) -> dict:
    """Return the sampler and session settings of the run."""
    s = pipe.sampler
    guided = s.guided
    skip = skip_video_decoder(session)
    # Without its video the last step needs the unguided pass only for the
    # action guidance.
    last = 2 if guided and not (skip and float(s.cfg_scale) == 1.0) else 1
    return {
        "num_steps": s.num_steps,
        "t_shift": s.t_shift,
        "cfg_scale": s.cfg_scale,
        "cfg_video_scale": s.cfg_video_scale,
        # window_joint runs the guided and the unguided pass one after the other.
        "cfg_mode": "sequential" if guided else "off",
        "passes_per_step": 2 if guided else 1,
        "last_step_passes": last,
        "context_mode": session.get("context_mode", "observed"),
        "max_context_chunks": int(session["max_context_chunks"]),
        "skip_video_decoder": skip,
        "prompt": PROMPT,
        "exec_horizon": session.get("exec_horizon"),
        "control_mode": session.get("control_mode"),
    }


def run(
    pipe: PraxisPipeline,
    session: dict,
    sessions: int,
    calls: int,
    warmup_calls: int,
    trace_dir: Path | None = None,
    stage_timing: bool = True,
    save_outputs: Path | None = None,
    noise_counts: dict[str, int] | None = None,
) -> list[dict]:
    """Run the sessions and return one record per call.

    ``noise_counts`` (``hits``, ``misses``) gets the noise prefetch counters
    of all the sessions of the run, warm-up and trace sessions included.
    """
    height, width = view_size(pipe)
    cuda = pipe.device.type == "cuda"
    timer = StageTimer(
        sync=lambda: torch.cuda.synchronize(pipe.device) if cuda else None
    )
    # A stateless session starts each call cold, with one frame (as the server).
    stateless = session.get("context_mode", "observed") == "stateless"
    depth = 0 if stateless else int(session["max_context_chunks"])
    skip = skip_video_decoder(session)
    noise_prefetch = noise_prefetch_setting(session)
    records: list[dict] = []
    # After the staged sessions come one plain session (one sync around each
    # call, no sync between the stages: the call time a server sees) and the
    # trace session. The profiler slows the calls down, so its calls are not
    # counted.
    plain_session = sessions if stage_timing else -1
    trace_session = sessions + int(stage_timing)
    total = trace_session + int(trace_dir is not None)
    if save_outputs is not None:
        save_outputs.mkdir(parents=True, exist_ok=True)
    with instrument(pipe, timer) if stage_timing else nullcontext(timer):
        for s in range(total):
            sess = pipe.session(
                prompt=PROMPT,
                seed=int(session.get("seed", 0)) + 1000 * s,
                max_context_chunks=depth,
                control_mode=str(session.get("control_mode", "droid_impedance")),
                exec_horizon=session.get("exec_horizon"),
                skip_last_video=skip,
            )
            if noise_prefetch is not None:
                # --set session.noise_prefetch=off|on overrides the default.
                sess.noise_prefetch = noise_prefetch
            plain = s == plain_session or not stage_timing
            prof = None
            if trace_dir is not None and s == trace_session:
                prof = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                )
                prof.start()
            for k in range(calls):
                counted = prof is None and len(records) >= warmup_calls
                stack = 1 if k == 0 or sess.state.stateless else 25
                obs = synthetic_obs(len(records), stack, height, width)
                try:
                    if plain:
                        # The wrappers time nothing outside timer.call().
                        timer.sync()
                        t0 = time.monotonic()
                        res = sess.step(obs)
                        timer.sync()
                        rec = {"total": (time.monotonic() - t0) * 1e3}
                    else:
                        with (
                            timer.call() as rec,
                            torch.profiler.record_function("call"),
                            timer.nvtx("call"),
                        ):
                            res = sess.step(obs)
                except BaseException:
                    if prof is not None:
                        prof.stop()
                    raise
                rec.update(
                    session=s,
                    call=k,
                    depth=res.depth,
                    rows=len(res.commands),
                    warmup=not counted and prof is None,
                    traced=prof is not None,
                    plain=plain,
                    stage_timing=stage_timing,
                )
                if save_outputs is not None:
                    # Copies and disk I/O are outside the call timer. Use
                    # this option for parity checks; it can affect pacing.
                    torch.save(
                        dict(
                            actions=res.pred_action_norm.detach().cpu().contiguous(),
                            video=(
                                None
                                if res.pred_video_latent is None
                                else res.pred_video_latent.detach().cpu().contiguous()
                            ),
                            commands=res.commands,
                        ),
                        save_outputs / f"session_{s:03d}_call_{k:03d}.pt",
                    )
                # The process age when the call returned (outside the timer).
                rec["since_start_s"] = round(since_start(), 3)
                records.append(dict(rec))
                print(
                    f"session {s} call {k}: depth {res.depth}, {rec['total']:.0f} ms"
                    + (" (trace)" if prof is not None else "")
                    + (" (plain)" if plain else "")
                    + (" (warm-up)" if not counted and prof is None else ""),
                    flush=True,
                )
            if prof is not None and trace_dir is not None:
                prof.stop()
                prof.export_chrome_trace(str(trace_dir / "trace.json"))
                table = prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=60
                )
                (trace_dir / "kernels.txt").write_text(table)
            cache = getattr(sess, "_noise", None)
            if noise_counts is not None and cache is not None:
                noise_counts["hits"] += cache.hits
                noise_counts["misses"] += cache.misses
    return records


def summary(records: list[dict]) -> dict:
    """Return the summary of the counted calls: first calls and later calls.

    ``total`` comes from the plain session: one sync around each call, the
    time a server sees. The stages and ``total_staged`` (the same calls with
    a sync around each stage) come from the staged sessions.
    """
    counted = [r for r in records if not (r["warmup"] or r["traced"])]
    out = {}
    for group, first in (("first", True), ("later", False)):
        rows = [r for r in counted if (r["call"] == 0) == first]
        staged = [{**r, "total_staged": r["total"]} for r in rows if not r["plain"]]
        out[group] = {
            "total": summarize(r["total"] for r in rows if r["plain"]),
            **(summarize_calls(staged, ("total_staged", *STAGES)) if staged else {}),
        }
    return out


def startup(
    records: list[dict], load_start: float, load_end: float, depths: int
) -> dict:
    """Return the startup times of the process and the compile cache in use.

    The times count from the start of the process (``process_age``), in
    seconds: ``imports_s`` until the pipeline load starts (torch, praxis),
    ``load_s`` the pipeline load (weights, text encoder, VAE, CUDA graphs),
    ``first_call_s`` the first policy call (it compiles the kernels of depth
    0), ``to_first_call_s`` from the start of the process to the end of the
    first call, and ``to_all_depths_s`` to the end of the first call of each
    context depth. With a warm compile cache the first calls load their
    kernels instead of compiling them.
    """
    keys = (CACHE_DIR_ENV, *cache_env("."))
    return {
        "imports_s": round(load_start, 3),
        "load_s": round(load_end - load_start, 3),
        "first_call_s": round(records[0]["total"] / 1e3, 3),
        "to_first_call_s": records[0]["since_start_s"],
        "to_all_depths_s": records[depths - 1]["since_start_s"],
        "process_age_available": _AGE0 is not None,
        "compile_cache": {k: os.environ.get(k) for k in keys},
    }


def format_summary(out: dict) -> str:
    """Return the summary as a text table, in milliseconds."""
    lines = [f"{'calls':<7}{'stage':<13}{'n':>5}{'p50':>9}{'p90':>9}{'max':>9}"]
    for group, stages in out.items():
        for stage, s in stages.items():
            if s.get("n"):
                lines.append(
                    f"{group:<7}{stage:<13}{s['n']:>5}"
                    f"{s['p50']:>9.1f}{s['p90']:>9.1f}{s['max']:>9.1f}"
                )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--config", default="configs/serve_droid.yaml")
    ap.add_argument(
        "--set", action="append", default=[], help="override, e.g. sampler.num_steps=4"
    )
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--calls", type=int, default=10, help="calls per session")
    ap.add_argument(
        "--warmup-calls",
        type=int,
        default=None,
        help="first calls that are not counted (default: one session)",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/benchmark"))
    ap.add_argument("--trace", action="store_true", help="write a torch.profiler trace")
    ap.add_argument(
        "--optimizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the default inference optimizations (default: enabled; "
        "--no-optimizations disables the fused rope, the device tables, the context "
        "cache, the expert overlap, the encode dedup and the CUDA graphs, "
        "including the VAE encode graphs)",
    )
    ap.add_argument(
        "--timing",
        choices=("stages", "total"),
        default="stages",
        help="stages synchronizes each stage; total synchronizes only call boundaries and preserves overlap",
    )
    ap.add_argument(
        "--save-outputs",
        action="store_true",
        help="save actions, video latents and commands outside the timer for bitwise comparisons",
    )
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    session = dict(cfg["session"])
    # Refuse a bad --set session.noise_prefetch before the weights load.
    noise_prefetch = noise_prefetch_setting(session)
    warmup = args.calls if args.warmup_calls is None else args.warmup_calls
    if args.sessions < 1 or args.calls < 1:
        ap.error("--sessions and --calls must be at least 1")
    # Each context depth compiles its kernels on its first call.
    stateless = session.get("context_mode", "observed") == "stateless"
    max_depth = 0 if stateless else int(session["max_context_chunks"])
    depths = min(args.calls, max_depth + 1)
    if warmup < depths:
        ap.error(f"--warmup-calls must be at least {depths}: one call per depth")
    if warmup >= args.sessions * args.calls:
        ap.error("--warmup-calls leaves no counted call")
    check_single_arm(cfg["model"]["checkpoint"])
    if not args.optimizations:
        # The graphs are set up at load. Without CACHE_DEVICE_TABLES the action
        # expert copies host tables, which a CUDA graph capture refuses.
        cfg["model"]["action_cuda_graphs"] = False
        cfg["model"]["nvfp4_moe_cuda_graphs"] = False
    load_start = since_start()
    pipe = build_pipeline(cfg)
    load_end = since_start()
    if pipe.embodiment != "single_arm":
        raise SystemExit(
            f"The benchmark makes single-arm observations; the checkpoint is for "
            f"the {pipe.embodiment!r} rig."
        )
    optimization_options = configure_optimizations(pipe, args.optimizations)
    from praxis.pipeline import noise_prefetch_default

    # The effective noise prefetch: --set session.noise_prefetch, else the
    # session default (CUDA, PRAXIS_NOISE_PREFETCH) unless optimizations are off.
    optimization_options["noise_prefetch"] = (
        noise_prefetch
        if noise_prefetch is not None
        else args.optimizations and noise_prefetch_default(pipe.device)
    )
    noise_counts = {"hits": 0, "misses": 0}
    args.out.mkdir(parents=True, exist_ok=True)
    records = run(
        pipe,
        session,
        args.sessions,
        args.calls,
        warmup,
        args.out if args.trace else None,
        stage_timing=args.timing == "stages",
        save_outputs=args.out / "outputs" if args.save_outputs else None,
        noise_counts=noise_counts,
    )
    out = {
        "noise_prefetch": (
            noise_counts if optimization_options["noise_prefetch"] else None
        ),
        "environment": environment(pipe),
        "settings": settings(pipe, session),
        "checkpoint": str(cfg["model"]["checkpoint"]),
        "view_size": list(view_size(pipe)),
        "warmup_calls": warmup,
        "optimization_options": optimization_options,
        "action_cuda_graphs": (
            pipe.action_cuda_graphs.stats()
            if hasattr(pipe, "action_cuda_graphs")
            else None
        ),
        "vae_encode_graphs": pipe.vae.encode_graph_stats(),
        "timing_mode": args.timing,
        "saved_outputs": args.save_outputs,
        "summary_ms": summary(records),
        "startup": startup(records, load_start, load_end, depths),
    }
    (args.out / "calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("environment", "settings")}, indent=1))
    print(format_summary(out["summary_ms"]))
    print(json.dumps({"startup": out["startup"]}, indent=1))


if __name__ == "__main__":
    main()
