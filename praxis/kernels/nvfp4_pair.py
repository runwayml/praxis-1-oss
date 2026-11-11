"""Paired CFG dispatch for the SM 11.0 NVFP4 routed experts (``nvfp4.PAIR_NVFP4``).

``Session.pair_cfg`` runs the guided and unguided encoder passes in lockstep and, for BF16 banks, sends the
routed rows of both passes through one grouped GEMM so each expert's weights are read once. The NVFP4 bank
cannot simply concatenate the two passes: the activation global scale of an expert is the amax of *that pass's*
rows of the expert, so a joint dispatch would change the globals and the codes.

This module keeps one GEMM group per expert (rows of pass 0, then rows of pass 1, then the expert's padding) but
quantizes and rescales every row with the global of its own (expert, pass) *segment*:

* segment ``2 e + s`` = the rows of pass ``s`` routed to expert ``e`` (``seg_csr``, 2E + 1 offsets);
* group ``e`` = both segments (``csr``, E + 1 offsets), padded to multiples of 4 (``pcsr``);
* the FP4 GEMM never sees a global (unit alpha): each output row depends only on its own codes and block scales
  and the expert's weights, so it has the bits of the unpaired GEMM row;
* globals per segment are computed exactly as ``expert_global_scales`` computes them per expert and pass
  (padding rows are zeros and do not move an amax).

The quantize body is ``praxis.kernels.nvfp4_sm110._quantize_q``; only the row -> (segment, group, local row)
mapping differs. Rescale + SwiGLU and the combine are those of ``praxis.kernels.nvfp4_fused``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from praxis.hardware import require_thor
from praxis.kernels.nvfp4 import _expert, _pad_csr
from praxis.kernels.nvfp4_fused import globals_from_row_amax, row_amax
from praxis.kernels.nvfp4_sm110 import _e2m1_pack


@triton.jit
def _row_map(
    SEG,
    CSR,
    PCSR,
    rows,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    SEARCH2: tl.constexpr,
    PADDED: tl.constexpr,
):
    """(segment, group, local row in the group) of unpadded combined rows (PADDED False) or padded rows."""
    if PADDED:
        group = _expert(PCSR, rows, E, SEARCH)
        local = rows - tl.load(PCSR + group)
        n0 = tl.load(SEG + 2 * group + 1) - tl.load(SEG + 2 * group)
        seg = 2 * group + (local >= n0).to(tl.int32)
    else:
        seg = _expert(SEG, rows, 2 * E, SEARCH2)
        group = seg // 2
        local = rows - tl.load(CSR + group)
    return seg, group, local


@triton.jit
def _quantize_seg(
    X,
    Q,
    SF,
    G,
    SEG,
    CSR,
    PCSR,
    MAP,
    M: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    SEARCH2: tl.constexpr,
    PADDED: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    NATIVE: tl.constexpr = False,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * C + tl.arange(0, C)
    seg, expert, local = _row_map(SEG, CSR, PCSR, rows, E, SEARCH, SEARCH2, PADDED)
    pstart = tl.load(PCSR + expert)
    source = rows
    if not PADDED:
        source = tl.load(MAP + rows, rows < M, other=0)
    values = tl.load(
        X + source[:, None].to(tl.int64) * K + cols[None, :],
        (rows[:, None] < M) & (cols[None, :] < K),
        other=0,
    ).to(tl.float32)
    blocks = values.reshape(R, C // 16, 16)
    block_scale = tl.max(tl.abs(blocks), axis=2) * (1.0 / 6.0)
    g = tl.load(G + seg)
    sf = tl.minimum(
        tl.maximum(libdevice.div_rn(block_scale, g[:, None]), 0.015625), 448.0
    ).to(tl.float8e4nv)
    reciprocal = libdevice.div_rn(libdevice.div_rn(1.0, g[:, None]), sf.to(tl.float32))
    norm = tl.minimum(tl.maximum(blocks * reciprocal[:, :, None], -6.0), 6.0).reshape(
        R, C
    )
    packed = _e2m1_pack(norm, R, C, NATIVE)
    pcols = tl.program_id(1) * (C // 2) + tl.arange(0, C // 2)
    tl.store(
        Q + (pstart + local)[:, None].to(tl.int64) * (K // 2) + pcols[None, :],
        packed,
        (rows[:, None] < M) & (pcols[None, :] < K // 2),
    )
    scol = tl.program_id(1) * (C // 16) + tl.arange(0, C // 16)
    sfbase = (pstart // 128 + expert) * 128 * (K // 16)
    offset = sfbase[:, None] + (
        (local[:, None] // 128 * (K // 64) + scol[None, :] // 4) * 512
        + (local[:, None] % 32) * 16
        + ((local[:, None] % 128) // 32) * 4
        + scol[None, :] % 4
    )
    tl.store(
        SF + offset,
        sf.to(tl.uint8, bitcast=True),
        (rows[:, None] < M) & (scol[None, :] < K // 16),
    )


def _quantize(x, g, seg_csr, csr, pcsr, padded_rows, m, row_map=None):
    from praxis.kernels.nvfp4_sm110 import alloc_qsf, native_e2m1_launch, quant_launch

    e = csr.numel() - 1
    k = x.shape[1]
    q, sf = alloc_qsf(
        padded_rows, k, e, csr, pcsr, x.device, all_rows_written=row_map is None
    )
    rp, nw = quant_launch()
    if m:

        def launch(qb, sfb, native):
            _quantize_seg[(triton.cdiv(m, rp), triton.cdiv(k, 256))](
                x,
                qb,
                sfb,
                g,
                seg_csr,
                csr,
                pcsr,
                csr if row_map is None else row_map,
                m,
                k,
                e,
                (e + 1).bit_length(),
                (2 * e + 1).bit_length(),
                row_map is None,
                rp,
                256,
                native,
                num_warps=nw,
                enable_fp_fusion=False,
                enable_reflect_ftz=False,
            )

        native_e2m1_launch(
            launch, q, sf, f"seg.K{k}.{'pad' if row_map is None else 'gather'}"
        )
    return q, sf


@triton.jit
def _rescale_swiglu_amax_seg(
    GU,
    OUT,
    AMAX,
    SEG,
    PCSR,
    G,
    W,
    H: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    rows = tl.full((1,), 0, tl.int32) + row
    seg, expert, local = _row_map(SEG, PCSR, PCSR, rows, E, SEARCH, SEARCH, True)
    multiplier = (tl.load(G + seg) * tl.load(W + expert)).to(tl.bfloat16).to(tl.float32)
    r64 = row.to(tl.int64)
    gate_raw = tl.load(GU + r64 * (2 * H) + col, mask=mask, other=0).to(tl.float32)
    up_raw = tl.load(GU + r64 * (2 * H) + H + col, mask=mask, other=0).to(tl.float32)
    gate_s = (gate_raw * multiplier).to(tl.bfloat16)
    up_s = (up_raw * multiplier).to(tl.bfloat16)
    gate = gate_s.to(tl.float32)
    out = (gate * tl.sigmoid(gate)).to(up_s.dtype) * up_s
    tl.store(OUT + r64 * H + col, out, mask=mask)
    m = tl.max(tl.where(mask, tl.abs(out.to(tl.float32)), 0.0), axis=0)
    tl.atomic_max(AMAX + row, m)


@triton.jit
def _combine_rescale_seg(
    RAW,
    INVERSE,
    SCALE,
    SEG,
    CSR,
    PCSR,
    G,
    W,
    OUT,
    T: tl.constexpr,
    C: tl.constexpr,
    TOP_K: tl.constexpr,
    E: tl.constexpr,
    SEARCH2: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    a0 = tl.full((R, BC), 0.0, tl.float32)
    a1 = tl.full((R, BC), 0.0, tl.float32)
    a2 = tl.full((R, BC), 0.0, tl.float32)
    a3 = tl.full((R, BC), 0.0, tl.float32)
    for slot in tl.static_range(TOP_K):
        index = tl.load(INVERSE + rows * TOP_K + slot, rows < T, other=0)
        seg, expert, local = _row_map(
            SEG, CSR, PCSR, index.to(tl.int32), E, SEARCH2, SEARCH2, False
        )
        source = tl.load(PCSR + expert) + local
        multiplier = (
            (tl.load(G + seg) * tl.load(W + expert)).to(tl.bfloat16).to(tl.float32)
        )
        raw = tl.load(
            RAW + source[:, None].to(tl.int64) * C + cols[None, :],
            (rows[:, None] < T) & (cols[None, :] < C),
            other=0,
        ).to(tl.float32)
        value = (raw * multiplier[:, None]).to(tl.bfloat16).to(tl.float32)
        scale = tl.load(SCALE + index, rows < T, other=0)
        product = value * scale[:, None]
        if slot % 4 == 0:
            a0 = a0 + product
        elif slot % 4 == 1:
            a1 = a1 + product
        elif slot % 4 == 2:
            a2 = a2 + product
        else:
            a3 = a3 + product
    result = ((a0 + a1) + a2) + a3
    tl.store(
        OUT + rows[:, None].to(tl.int64) * C + cols[None, :],
        result.to(tl.bfloat16),
        (rows[:, None] < T) & (cols[None, :] < C),
    )


def _padded_segments(seg_csr, pcsr, padded_rows):
    """Segment of every padded row (padding rows belong to the group's last segment)."""
    e = pcsr.numel() - 1
    p = torch.arange(padded_rows, device=pcsr.device, dtype=pcsr.dtype)
    group = torch.bucketize(p, pcsr[1:], right=True).clamp_max_(e - 1)
    local = p - pcsr[group.long()]
    n0 = (seg_csr[1::2] - seg_csr[0:-1:2])[group.long()]
    return 2 * group + (local >= n0).to(group.dtype)


def combine_pass(raw, inverse, scale, seg_csr, csr, pcsr, hg, wg, t):
    """The per-pass combine of ``pair_expert_outputs`` (``inverse`` starts at the pass's first slot)."""
    e = csr.numel() - 1
    c = raw.shape[1]
    out = torch.empty((t, c), device=raw.device, dtype=torch.bfloat16)
    if t:
        _combine_rescale_seg[(triton.cdiv(t, 4), triton.cdiv(c, 256))](
            raw,
            inverse.contiguous(),
            scale,
            seg_csr,
            csr,
            pcsr,
            hg,
            wg,
            out,
            t,
            c,
            8,
            e,
            (2 * e + 1).bit_length(),
            4,
            256,
            num_warps=4,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return out


def pair_expert_outputs(
    bank, xs, top_idxs, gates, scaling, tail_shared=None, dispatch=None
):
    """Routed sums ``[bank.routed_forward(x, ...)]`` of the inputs ``xs`` (one dispatch chunk each), with one
    gate|up and one down FP4 GEMM for all inputs. ``bank``: NVFP4RoutedExperts on SM 11.0.
    ``tail_shared`` (top-8 only, ``praxis.layers.MOE_TAIL_NORM``): the shared-expert outputs of the inputs; returns
    the per-input ``praxis.layers.MoETail`` (mode 3) of ``routed + shared`` instead of combining. ``dispatch``: a
    ``praxis.layers.ROUTER_FUSED`` router_fused.Routing (P = 2) of the two inputs; its order / tokens / sorted
    gates / seg_csr / inverse are the values of the sort below."""
    for x in xs:
        require_thor(x.device)
    from praxis.kernels import nvfp4_sm110 as NS
    from praxis.kernels.nvfp4_epi import gate_up_swiglu_amax, sync_gate_up_layout
    from praxis.kernels.nvfp4_sm110 import grouped_gemm_nvfp4

    assert len(xs) == 2
    epi = NS.EPI_SWIGLU
    sync_gate_up_layout(bank, epi)
    e = bank.num_experts
    top_k = top_idxs[0].shape[1]
    t0, t1 = xs[0].shape[0], xs[1].shape[0]
    dev = xs[0].device
    # Dispatch: stable sort by segment 2 * expert + pass over the slots of pass 0 then pass 1, so the rows of a
    # segment keep the order of the pass's own _dispatch_rows (a stable sort by expert).
    if dispatch is not None:
        assert (
            tuple(dispatch.rows) == (t0, t1) and dispatch.seg_csr.numel() == 2 * e + 1
        )
        order, tokens, sorted_gate, seg_csr = (
            dispatch.order,
            dispatch.tokens,
            dispatch.sorted_gate,
            dispatch.seg_csr,
        )
    else:
        keys = torch.cat([top_idxs[0].reshape(-1) * 2, top_idxs[1].reshape(-1) * 2 + 1])
        _, order = keys.sort(stable=True)
        tokens = torch.cat(
            [
                torch.arange(t0, device=dev).repeat_interleave(top_k),
                torch.arange(t1, device=dev).repeat_interleave(top_k) + t0,
            ]
        )[order]
        sorted_gate = torch.cat([gates[0].reshape(-1), gates[1].reshape(-1)])[order]
        seg_counts = torch.zeros(2 * e, device=dev, dtype=torch.int64)
        seg_counts.scatter_add_(
            0, keys.long(), torch.ones_like(keys, dtype=torch.int64)
        )
        seg_csr = torch.nn.functional.pad(seg_counts.cumsum(0).to(torch.int32), (1, 0))
    # group starts = segment 2e starts; csr[e] = seg_csr[2e]
    csr = seg_csr[0::2].contiguous()
    m = order.numel()
    padded_rows = ((m + 3 * e) // 4) * 4
    pcsr = torch.empty_like(csr)
    _pad_csr[(1,)](csr, pcsr, e, triton.next_power_of_2(e + 1), 4)
    x = torch.cat(xs)
    # gate|up
    ag = globals_from_row_amax(row_amax(x), seg_csr, tokens)
    aq, asf = _quantize(x, ag, seg_csr, csr, pcsr, padded_rows, m, row_map=tokens)
    if epi:
        # Rescale + SwiGLU + row amax in the GEMM epilogue (same bits).
        hidden, hamax = gate_up_swiglu_amax(bank, aq, asf, pcsr, ag, seg_csr=seg_csr)
        del aq, asf
    else:
        gu = grouped_gemm_nvfp4(aq, asf, bank.gate_up_q, bank.gate_up_sf, pcsr)
        del aq, asf
        h = gu.shape[1] // 2
        hidden = torch.empty((padded_rows, h), dtype=torch.bfloat16, device=dev)
        hamax = torch.zeros(padded_rows, dtype=torch.float32, device=dev)
        if padded_rows:
            _rescale_swiglu_amax_seg[(padded_rows, triton.cdiv(h, 1024))](
                gu,
                hidden,
                hamax,
                seg_csr,
                pcsr,
                ag,
                bank.gate_up_global,
                h,
                e,
                (e + 1).bit_length(),
                1024,
                num_warps=4,
                enable_fp_fusion=False,
                enable_reflect_ftz=False,
            )
        del gu
    # Hidden globals per segment from the padded rows (padding rows are zeros).
    seg_of_row = _padded_segments(seg_csr, pcsr, padded_rows)
    group_amax = torch.zeros(2 * e, dtype=torch.bfloat16, device=dev)
    group_amax.scatter_reduce_(0, seg_of_row.long(), hamax.to(torch.bfloat16), "amax")
    hg = group_amax.clamp_min(1e-12).float() * (1.0 / 2688.0)
    hq, hsf = _quantize(hidden, hg, seg_csr, csr, pcsr, padded_rows, padded_rows)
    del hidden
    raw = grouped_gemm_nvfp4(hq, hsf, bank.down_q, bank.down_sf, pcsr)
    del hq, hsf
    # Combine per pass.
    if dispatch is not None:
        inverse = dispatch.inverse
    else:
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=dev)
    scale = sorted_gate.to(torch.float32) * scaling
    if tail_shared is not None:
        from praxis.layers import MoETail

        return [
            MoETail(
                3,
                raw,
                inverse[start:],
                scale,
                sh,
                float(scaling),
                None,
                seg=seg_csr,
                csr=csr,
                pcsr=pcsr,
                g=hg,
                w=bank.down_global,
            )
            for sh, start in ((tail_shared[0], 0), (tail_shared[1], t0 * top_k))
        ]
    outs = []
    for t, start in ((t0, 0), (t1, t0 * top_k)):
        c = raw.shape[1]
        out = torch.empty((t, c), device=dev, dtype=torch.bfloat16)
        if t:
            _combine_rescale_seg[(triton.cdiv(t, 4), triton.cdiv(c, 256))](
                raw,
                inverse[start:].contiguous(),
                scale,
                seg_csr,
                csr,
                pcsr,
                hg,
                bank.down_global,
                out,
                t,
                c,
                top_k,
                e,
                (2 * e + 1).bit_length(),
                4,
                256,
                num_warps=4,
                enable_fp_fusion=False,
                enable_reflect_ftz=False,
            )
        outs.append(out)
    return outs
