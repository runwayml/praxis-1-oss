"""Tune ``layers.CUTE_DENSE_FFN`` for one tcgen05 GPU (SM100/SM103/SM110).

The fused dense SwiGLU gate|up GEMM (``praxis.kernels.cute_dense``) dispatches by row count through a
per-architecture table. This script builds that table for the current GPU:

  gu    gate|up + SiLU*up: candidate Quack configs at each row count, each checked bitwise (torch.equal) against
        the eager ``F.silu(x A^T) * (x B^T)`` and timed against it (cold L2, CUDA events, median of reps).
  down  h @ W^T: candidate Quack configs bitwise against F.linear and timed against cuBLAS.
  ffn   the whole module flag off vs on with this run's winners, on random weights and (with --ckpt) the
        checkpoint's dense blocks, with stress rows 2^-12..2^5; bitwise and timed.

It prints ``GATE_UP_TABLES[(major, minor)] = (...)`` (and a down table when Quack wins) to add to cute_dense.py.
Rows where no bitwise config beats the eager ops are left out (they run the ordinary ops). After adding a table,
check the full policy bitwise (actions, commands, predicted latents) with the flag off vs on before enabling it.

    python scripts/tune_cute_dense.py OUT_DIR [--parts gu,down,ffn] [--reps 7] [--rows 256,512,...] [--ckpt DIR]

The default row counts are the DROID policy's dense-FFN row counts (per pass and the first call) plus a grid.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

C, HD = 3072, 11520
ROWS = [
    256,
    512,
    768,
    1024,
    1279,
    1286,
    1536,
    2048,
    2560,
    3463,
    3470,
    4372,
    5120,
    6933,
    8740,
    17480,
]
REAL = [1279, 1286, 2560, 3463, 3470, 4372, 8740]
GU = [
    (256, 256, 2, 1, True, 8, False),
    (256, 256, 2, 1, False, 8, False),
    (256, 256, 2, 2, True, 8, False),
    (256, 256, 2, 2, False, 8, False),
    (256, 256, 4, 1, True, 8, False),
    (256, 256, 4, 1, False, 4, False),
    (256, 256, 2, 1, True, 1, False),
    (256, 256, 2, 1, True, 4, False),
    (128, 256, 4, 1, False, 4, False),
    (128, 256, 2, 1, True, 8, False),
    (128, 256, 1, 1, True, 8, False),
    (256, 224, 2, 1, True, 8, False),
    (256, 192, 2, 1, True, 8, False),
    (256, 128, 2, 1, True, 8, False),
    (256, 128, 2, 2, True, 8, False),
    (256, 160, 2, 1, True, 8, False),
]
DOWN = [
    (128, 256, 1, 1, True, 8, False),
    (256, 256, 4, 1, False, 8, False),
    (256, 256, 2, 1, False, 8, False),
    (256, 192, 2, 1, False, 8, True),
    (256, 256, 2, 1, False, 8, True),
    (256, 224, 4, 1, False, 8, True),
]
DENSE_BLOCKS = {
    "enc0": "backbone.encoder.core.blocks.0",
    "dec0": "backbone.decoder.core.blocks.0",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--parts", default="gu,down,ffn")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="checkpoint dir (model.safetensors.index.json)",
    )
    args = ap.parse_args()

    from praxis import layers as L
    from praxis.kernels import cute_dense as CD
    from praxis.kernels.cute_experts import _cfg

    dev = torch.device("cuda")
    cap = tuple(torch.cuda.get_device_capability(dev))
    if not CD.supported():
        raise SystemExit(f"capability {cap}: no tcgen05 Quack path")
    rows = sorted({int(v) for v in args.rows.split(",")})
    flush = torch.empty(256 * 2**20 // 4, dtype=torch.float32, device=dev)
    res: dict = {
        "time": time.strftime("%FT%TZ", time.gmtime()),
        "capability": cap,
        "device": torch.cuda.get_device_name(),
    }
    args.out.mkdir(parents=True, exist_ok=True)

    def dump() -> None:
        (args.out / "tune_cute_dense.json").write_text(
            json.dumps(res, indent=1, default=str)
        )

    def timed(fn) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.reps):
            flush.zero_()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return round(st.median(ts), 4)

    def ndiff(a, b):
        if a.shape != b.shape:
            return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
        return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())

    def best(rec: dict, ref, fns) -> list:
        for key, fn in fns:
            try:
                nd = ndiff(fn(), ref)
                rec[key] = {
                    "eq": nd == 0,
                    "ms": timed(fn),
                    **({"ndiff": nd} if nd else {}),
                }
            except Exception as e:  # a config can fail to compile on a part
                rec[key] = {"err": repr(e)[:300]}
                torch.cuda.synchronize()
        return sorted((v["ms"], k) for k, v in rec.items() if v.get("eq"))[:6]

    def module(real: str | None = None, seed: int = 7):
        m = L.SwiGLUFeedForward(C, HD).to(dev).bfloat16()
        if real is None:
            g = torch.Generator(device=dev).manual_seed(seed)
            for p in m.parameters():
                p.data.copy_(torch.randn(p.shape, device=dev, generator=g) * 0.02)
            return m
        from safetensors import safe_open

        index = json.loads((args.ckpt / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        params = {
            "0.linear.a_weight": m[0].linear.a_weight,
            "0.linear.b_weight": m[0].linear.b_weight,
            "1.weight": m[1].weight,
        }
        for suffix, p in params.items():
            key = f"{real}.ffn.{suffix}"
            with safe_open(
                str(args.ckpt / index[key]), framework="pt", device="cpu"
            ) as f:
                p.data.copy_(f.get_tensor(key).to(dev, torch.bfloat16))
        return m

    def key(c) -> str:
        tm, tn, cm, cn, dyn, sw, swap = c
        return f"m{tm}_n{tn}_c{cm}x{cn}_d{int(dyn)}_s{sw}" + ("_swap" if swap else "")

    def table(part: str, configs, eager: str) -> tuple:
        out = []
        for m, rec in sorted(res.get(part, {}).items(), key=lambda kv: int(kv[0])):
            top = rec.get("top") or []
            if top and top[0][0] < rec[eager]:
                out.append(
                    (
                        int(m),
                        int(m) + 1,
                        next(c for c in configs if key(c) == top[0][1]),
                    )
                )
        return tuple(out)

    def part_gu() -> None:
        mod = module()
        a, b = mod[0].linear.a_weight, mod[0].linear.b_weight
        wt = CD.packed_weight(a, b)
        g = torch.Generator(device=dev).manual_seed(11)
        for m in rows:
            rec = res.setdefault("gu", {}).setdefault(str(m), {})
            x = torch.randn(m, C, device=dev, generator=g).bfloat16()
            ref = F.silu(F.linear(x, a)) * F.linear(x, b)
            rec["eager_ms"] = timed(lambda x=x: F.silu(F.linear(x, a)) * F.linear(x, b))
            fns = [
                (key(c), lambda c=c, x=x: CD.gate_up_silu(x, wt, _cfg(*c))) for c in GU
            ]
            rec["top"] = best(rec.setdefault("v", {}), ref, fns)
            print("gu", m, rec["eager_ms"], rec["top"][:2], flush=True)
            dump()

    def part_down() -> None:
        w = module()[1].weight
        g = torch.Generator(device=dev).manual_seed(13)
        for m in rows:
            rec = res.setdefault("down", {}).setdefault(str(m), {})
            h = (torch.randn(m, HD, device=dev, generator=g) * 0.5).bfloat16()
            ref = F.linear(h, w)
            rec["cublas_ms"] = timed(lambda h=h: F.linear(h, w))
            fns = [(key(c), lambda c=c, h=h: CD.down(h, w, _cfg(*c))) for c in DOWN]
            rec["top"] = best(rec.setdefault("v", {}), ref, fns)
            print("down", m, rec["cublas_ms"], rec["top"][:2], flush=True)
            dump()

    def part_ffn() -> None:
        if "gu" in res:
            CD.GATE_UP_TABLES[cap] = table("gu", GU, "eager_ms")
        g = torch.Generator(device=dev).manual_seed(17)
        sources = {"rand": None, **(DENSE_BLOCKS if args.ckpt else {})}
        for label, real in sources.items():
            mod = module(real)
            for m in REAL:
                x = torch.randn(1, m, C, device=dev, generator=g).bfloat16()
                n = min(m, 18)
                x[0, :n] *= torch.tensor(
                    [2.0**e for e in range(-12, 6)][:n], device=dev
                )[:, None]
                with torch.inference_mode():
                    L.CUTE_DENSE_FFN = False
                    ref = mod(x)
                    t_off = timed(lambda x=x, mod=mod: mod(x))
                    L.CUTE_DENSE_FFN = True
                    y = mod(x)
                    t_on = timed(lambda x=x, mod=mod: mod(x))
                    L.CUTE_DENSE_FFN = False
                    y2 = mod(x)
                rec = {
                    "off_ms": t_off,
                    "on_ms": t_on,
                    "ndiff": ndiff(y, ref),
                    "off_after_pack_ndiff": ndiff(y2, ref),
                }
                res.setdefault("ffn", {}).setdefault(label, {})[str(m)] = rec
                print("ffn", label, m, rec, flush=True)
                dump()

    parts = {"gu": part_gu, "down": part_down, "ffn": part_ffn}
    for name in args.parts.split(","):
        try:
            parts[name]()
        except Exception:
            res.setdefault("errors", {})[name] = traceback.format_exc()[-2000:]
            print(res["errors"][name], flush=True)
        dump()
    print(f"GATE_UP_TABLES[{cap}] = {table('gu', GU, 'eager_ms')!r}")
    if "down" in res:
        print(f"DOWN_TABLES[{cap}] = {table('down', DOWN, 'cublas_ms')!r}")


if __name__ == "__main__":
    main()
