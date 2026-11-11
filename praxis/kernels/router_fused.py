"""MoE routing and dispatch after the stock FP32 router GEMM in one CuTe DSL kernel pair (``layers.ROUTER_FUSED``).

Scope: Blackwell tcgen05 and Thor (SM100/SM103/SM110), compiled per CUDA device.
Validated on B200 and Thor (bitwise replay and the routing audit). The kernel reproduces the torch.topk
slot order and the gate-sum order of the pinned torch (2.13). After a torch upgrade rerun
the routing audit (``ROUTER_FUSED_AUDIT``) on a
full replay before relying on the flag.

The logits stay those of the stock router (``layers._router_logits``: the FP32 GEMM, incl. ``ROUTER_F32`` rows and
``CTX_FULL_ROWS``' full-shape GEMM on context-cache reuse). Everything after it runs here, with the bits of the stock
ops (``MoEFeedForward._route`` + the dispatch of each expert path):

    scores = torch.sigmoid(logits)                          ATen: 1 / (1 + expf(-x)), precise expf, IEEE division
    biased = scores + bias                                  fp32 add
    _, top_idx = torch.topk(biased, 8, dim=-1)              set + slot order of torch.topk, incl. ties (see below)
    picked = scores.gather(-1, top_idx)
    gate = (picked / picked.sum(-1, keepdim=True)).to(bf16) sum order ((p0+p4)+(p2+p6))+((p1+p5)+(p3+p7)), div.rn

plus the dispatch metadata (order / inverse / sorted gates / tokens / seg_csr / offsets) of the expert paths:
BF16 FUSED_MOE (argsort(stable) + moe_dispatch_metadata), the NVFP4 pair (sort by 2 e + pass), the NVFP4 solo bank
(_dispatch_rows). For P = 2 (the CFG pair, inputs back to back) the rows of pass 0 come first inside each expert.

torch.topk slot order (CUDA, k = 8 of 128, sorted=True): radix select of the k-th value in the order-preserving
integer space (-0.0 < +0.0), gatherTopK writes the elements above it in index order, then the ones equal to it in
index order (the lowest indices win), then sortKeyValueInplace(stable=False) -> SmallBitonicSort: a 32-wide bitonic
network (16 threads x 2) with GTOp (float >) and invalid padding entries, which is NOT stable on ties. Which slots
the valid entries reach does not depend on the data, so ``_torch_topk_net`` folds it into a fixed compare-exchange
list on 8 registers. ``emulate_topk`` is the torch twin (probe: bitwise vs torch.topk on tie-heavy data).

Kernels (two launches of G <= #SM CTAs of 8 warps with the same token partition; a single launch with a spin grid barrier
hangs in a long replay when not all CTAs were resident, see below):
  phase 1  4 lanes per token, 8 tokens per warp step: 32 logits per lane, 64-bit keys (order-preserving bits << 32 |
           127 - expert), sorting networks + 2 butterfly merges -> the top-8 set on all 4 lanes; then the gather
           order (index order, k-th-value ties last) by an 8-key network and the folded torch bitonic network; gates
           in the stock sum order and div.rn; per-warp (expert, pass) histograms -> global rows (G * 8, P * 128).
  launch 2 (phase 2) every CTA sums all histogram rows (segment starts by a block scan; its base from the rows
           before it) -> per-warp cursors; CTA 0 writes seg_csr / offsets; (phase 3) stable expert-major scatter
           (tokens in order, one token at a time on the cursors).
The kernel boundary is the only grid-wide synchronization: no CTA ever waits on another, so the kernels make progress
with any number of resident CTAs (e.g. beside the action-expert side stream).
"""

from typing import NamedTuple

import torch
from torch import Tensor

from praxis.hardware import is_blackwell_tcgen05

E_ROUTER = 128
TOP_K = 8
_WARPS = 8

# Division of the gates: "ptx" (inline div.rn.f32) or "ir" (the DSL's float division). The probe checks both.
DIV_MODE = "ptx"


class Routing(NamedTuple):
    top_idx: Tensor
    gate: Tensor
    order: Tensor
    inverse: Tensor
    sorted_gate: Tensor
    tokens: Tensor
    tokens32: Tensor
    seg_csr: Tensor
    offsets: Tensor
    rows: tuple


# ---------------------------------------------------------------------------------------------------------------
# torch.topk's slot order on ties
# ---------------------------------------------------------------------------------------------------------------
def _torch_topk_net(
    n: int = TOP_K, size: int = 32, flag_from: str = "tid"
) -> tuple[list, list]:
    """Fold ATen's bitonicSort<size> (SortUtils.cuh; size/2 threads, invalid entries always sort to the end) on ``n``
    valid entries into compare-exchanges on n registers: ``ops`` = [(ra, rb, dir)]: exchange the contents of
    registers ra (the A side) and rb when ``(v[ra] > v[rb]) == dir``. ``out[i]`` = the register that ends in slot i."""
    pos2reg = {i: i for i in range(n)}
    ops: list = []
    half = size // 2

    def swap(a, b, d):
        va, vb = a in pos2reg, b in pos2reg
        if va and vb:
            ops.append((pos2reg[a], pos2reg[b], d))
        elif va:  # B invalid: swap = True
            if d:
                pos2reg[b] = pos2reg.pop(a)
        elif vb:  # A invalid: swap = False
            if not d:
                pos2reg[a] = pos2reg.pop(b)

    s = 2
    while s < size:
        st = s // 2
        while st > 0:
            for tid in range(half):
                pos = 2 * tid - (tid & (st - 1))
                flag = (
                    ((tid & (s // 2)) != 0) if flag_from == "tid" else ((pos & s) != 0)
                )
                swap(pos, pos + st, flag)
            st //= 2
        s *= 2
    st = size // 2
    while st > 0:
        for tid in range(half):
            pos = 2 * tid - (tid & (st - 1))
            swap(pos, pos + st, False)
        st //= 2
    return ops, [pos2reg[i] for i in range(n)]


_NET_OPS, _NET_OUT = _torch_topk_net()


def _ordered_bits(v: Tensor) -> Tensor:
    """float32 -> int64 of the order-preserving int32 (ATen TopKTypeConfig order; -0.0 < +0.0)."""
    b = v.view(torch.int32).to(torch.int64)
    return torch.where(b >= 0, b, b ^ 0x7FFFFFFF)


def emulate_topk(biased: Tensor, k: int = TOP_K, variant: str = "bitonic32") -> Tensor:
    """torch.topk(biased, k, -1).indices emulated (biased (T, E) float32, no NaN). Variants (probe): 'bitonic32'
    (the model above), 'bitonic32pos' (direction flag from the position bits), 'stable' (gather order kept on ties),
    'lowidx' (ties by lower index)."""
    T, E = biased.shape
    idx = torch.arange(E, device=biased.device, dtype=torch.int64)
    keys = (_ordered_bits(biased) << 32) | (E - 1 - idx)
    acc = keys.topk(k, dim=-1).values  # distinct keys: value desc, lower index first
    sel = (E - 1) - (acc & (E - 1))
    if variant == "lowidx":
        return sel
    hi = acc >> 32
    eq = hi == hi[:, k - 1 : k]
    g = eq.to(torch.int64) * E + sel  # gather order: index order, k-th-value ties last
    gi = sel.gather(1, g.argsort(dim=-1))
    vals = biased.gather(1, gi)
    if variant == "stable":
        o = torch.sort(vals, dim=-1, descending=True, stable=True).indices
        return gi.gather(1, o)
    ops, out = _torch_topk_net(
        k, flag_from="pos" if variant == "bitonic32pos" else "tid"
    )
    v = [vals[:, i].clone() for i in range(k)]
    r = [gi[:, i].clone() for i in range(k)]
    for a, b, d in ops:
        c = v[a] > v[b]
        sw = c if d else ~c
        va, vb, ra, rb = v[a], v[b], r[a], r[b]
        v[a], v[b] = torch.where(sw, vb, va), torch.where(sw, va, vb)
        r[a], r[b] = torch.where(sw, rb, ra), torch.where(sw, ra, rb)
    return torch.stack([r[out[i]] for i in range(k)], dim=1)


# The add tree of ``picked.sum(-1)`` (8 wide, float32) as ATen's reduction runs it on Thor (torch 2.13): leaves are
# slot numbers, a pair is one fp32 add. Reduce.cuh with 4 threads per output, each thread two elements 4 apart
# (thread_reduce), then a shuffle-down tree with offsets 2, 1. Found by the probe among _sum_candidates (the only one
# bitwise equal to torch: 0 mismatches over ~720k rows).
SUM_TREE: object = (((0, 4), (2, 6)), ((1, 5), (3, 7)))


def _sum_candidates(n: int = TOP_K) -> dict:
    """Symbolic add trees of ATen Reduce.cuh for an n-wide inner reduction: ``bw`` threads per output, each
    thread_reduce (vt0 = 4 accumulators, element stride bw; or 'vec' loads of 4 / 2 adjacent elements), the
    accumulators combined in order, then a warp shuffle-down tree over the bw threads (offsets ascending or
    descending). Plus the plain sequential and the halves trees."""

    def add(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return (a, b)

    out = {}
    for mode in ("strided", "vec4", "vec2"):
        for bw in (1, 2, 4, 8):
            per = []
            for t in range(bw):
                if mode == "strided":
                    acc = [None] * 4
                    idx = t
                    while idx + 3 * bw < n:
                        for i in range(4):
                            acc[i] = add(acc[i], idx + i * bw)
                        idx += 4 * bw
                    for i in range(4):
                        if idx >= n:
                            break
                        acc[i] = add(acc[i], idx)
                        idx += bw
                else:
                    v = 4 if mode == "vec4" else 2
                    acc = [None] * v
                    base = t * v
                    while base < n:
                        for i in range(v):
                            if base + i < n:
                                acc[i] = add(acc[i], base + i)
                        base += bw * v
                tot = None
                for a in acc:
                    tot = add(tot, a)
                per.append(tot)
            for order in ("asc", "desc"):
                offs = []
                o = 1
                while o < bw:
                    offs.append(o)
                    o *= 2
                if order == "desc":
                    offs = offs[::-1]
                lanes = list(per)
                for o in offs:
                    lanes = [
                        add(lanes[j], lanes[j + o]) if j + o < bw else lanes[j]
                        for j in range(bw)
                    ]
                out[f"{mode}/bw{bw}/{order}"] = lanes[0]
    seq = 0
    for i in range(1, n):
        seq = (seq, i)
    out["sequential"] = seq
    out["pairs_tree"] = (((0, 1), (2, 3)), ((4, 5), (6, 7)))
    dedup, seen = {}, set()
    for k, tr in out.items():
        if repr(tr) not in seen:
            seen.add(repr(tr))
            dedup[k] = tr
    return dedup


def emulate_gate_sum(picked: Tensor, tree=None) -> Tensor:
    """picked.sum(-1) along the add tree ``tree`` (default SUM_TREE)."""
    tree = SUM_TREE if tree is None else tree

    def ev(t):
        if isinstance(t, int):
            return picked[:, t]
        return ev(t[0]) + ev(t[1])

    return ev(tree)


# ---------------------------------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------------------------------
# Sorting network for 8 keys (19 comparators, optimal).
_NET8 = (
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (2, 4),
    (3, 5),
    (1, 4),
    (3, 6),
    (1, 2),
    (3, 4),
    (5, 6),
)


def _build_kernel_class(div_mode: str, tree):
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64, const_expr

    from praxis.kernels.cute_act import _div_rn

    if div_mode == "ptx":
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T, dsl_user_op

        @dsl_user_op
        def _div(a, b, *, loc=None, ip=None):
            return Float32(
                llvm.inline_asm(
                    T.f32(),
                    [
                        Float32(a).ir_value(loc=loc, ip=ip),
                        Float32(b).ir_value(loc=loc, ip=ip),
                    ],
                    "div.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )
    else:

        def _div(a, b):
            return a / b

    def _ptr(t, coord):
        return t.iterator + cute.crd2idx(coord, t.layout)

    def _sel(c, a, b, ty):
        return ty(cutlass.select_(c, a, b))

    def _sigmoid(v):
        # ATen sigmoid (float): 1 / (1 + expf(-x)), precise expf, IEEE division (d >= 1 or +inf).
        return _div_rn(Float32(1.0), Float32(1.0) + cute.math.exp(-v, fastmath=False))

    def _cas(a, i, j):  # descending
        hi = cutlass.max(a[i], a[j])
        lo = cutlass.min(a[i], a[j])
        a[i] = hi
        a[j] = lo

    def _cas_up(a, i, j):  # ascending
        lo = cutlass.min(a[i], a[j])
        hi = cutlass.max(a[i], a[j])
        a[i] = lo
        a[j] = hi

    def _sort8_up(a):
        for i, j in _NET8:
            _cas_up(a, i, j)

    def _torch_net(rV, rI):
        for a, b, d in _NET_OPS:
            va = rV[a]
            vb = rV[b]
            ia = rI[a]
            ib = rI[b]
            c = va > vb
            if d:  # exchange when v[a] > v[b]
                rV[a] = _sel(c, vb, va, Float32)
                rV[b] = _sel(c, va, vb, Float32)
                rI[a] = _sel(c, ib, ia, Int32)
                rI[b] = _sel(c, ia, ib, Int32)
            else:  # exchange unless v[a] > v[b] (ties exchange)
                rV[a] = _sel(c, va, vb, Float32)
                rV[b] = _sel(c, vb, va, Float32)
                rI[a] = _sel(c, ia, ib, Int32)
                rI[b] = _sel(c, ib, ia, Int32)

    def _sort8(a, off):
        for i, j in _NET8:
            _cas(a, off + i, off + j)

    def _merge_top8(acc, other, off):
        """acc[0:8], other[off:off+8] sorted descending -> acc = top-8 of both, descending."""
        for i in range(8):
            acc[i] = cutlass.max(acc[i], other[off + 7 - i])
        for step in (4, 2, 1):
            for b in range(0, 8, 2 * step):
                for i in range(b, b + step):
                    _cas(acc, i, i + step)

    sum_tree = tree

    def _tsum(pv, tr):
        if isinstance(tr, int):
            return pv[tr]
        return _tsum(pv, tr[0]) + _tsum(pv, tr[1])

    class RouteDispatch:
        def __init__(self, P: int, debug: bool, phase: int):
            self.P = P
            self.debug = debug
            self.phase = phase
            self.E = E_ROUTER
            self.W = _WARPS
            self.S = P * E_ROUTER
            assert self.S % 32 == 0 and self.S <= self.W * 32

        @cute.jit
        def __call__(
            self,
            mLogits,
            mBias,
            mTopIdx,
            mGate,
            mOrder,
            mInverse,
            mSortedGate,
            mTok64,
            mTok32,
            mSegCsr,
            mOffsets,
            mHist,
            mDbgScore,
            mDbgGate,
            T: Int32,
            t0: Int32,
            G: Int32,
            stream,
        ):
            self.kernel(
                mLogits,
                mBias,
                mTopIdx,
                mGate,
                mOrder,
                mInverse,
                mSortedGate,
                mTok64,
                mTok32,
                mSegCsr,
                mOffsets,
                mHist,
                mDbgScore,
                mDbgGate,
                T,
                t0,
                G,
            ).launch(grid=[G, 1, 1], block=[self.W * 32, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            mLogits,
            mBias,
            mTopIdx,
            mGate,
            mOrder,
            mInverse,
            mSortedGate,
            mTok64,
            mTok32,
            mSegCsr,
            mOffsets,
            mHist,
            mDbgScore,
            mDbgGate,
            T: Int32,
            t0: Int32,
            G: Int32,
        ):
            P, W, S = self.P, self.W, self.S
            NT = W * 32
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp = tidx // 32
            lane = tidx % 32

            smem = cutlass.utils.SmemAllocator()
            sHist = smem.allocate_tensor(
                Int32, cute.make_layout(W * S), byte_alignment=16
            )
            sWarp = smem.allocate_tensor(Int32, cute.make_layout(W), byte_alignment=16)
            for i in cutlass.range_constexpr(W * S // NT):
                sHist[tidx + i * NT] = Int32(0)
            cute.arch.sync_threads()

            U = G * W
            u = bidx * W + warp
            lo = (u * T) // U
            hi = ((u + 1) * T) // U

            if const_expr(self.phase == 1):
                # ---- phase 1: top-8 (torch order) + gates + per-warp segment counts ----------------------------
                q = lane % 4
                sub = lane // 4
                bias_v = cute.tiled_divide(mBias, (4,))
                for base in cutlass.range(lo, hi, 8):
                    t = base + sub
                    valid = t < hi
                    tl = cutlass.min(t, hi - 1)
                    rX = cute.make_rmem_tensor((4, 8), Float32)
                    row_v = cute.tiled_divide(mLogits[tl, None], (4,))
                    for i in cutlass.range_constexpr(8):
                        cute.autovec_copy(row_v[None, 4 * i + q], rX[None, i])
                    rB = cute.make_rmem_tensor((4, 8), Float32)
                    for i in cutlass.range_constexpr(8):
                        cute.autovec_copy(bias_v[None, 4 * i + q], rB[None, i])
                    for i in cutlass.range_constexpr(8):
                        for v in cutlass.range_constexpr(4):
                            s_ = _sigmoid(rX[v, i])
                            if const_expr(self.debug):
                                if valid:
                                    mDbgScore[t, Int32(16 * i + v) + 4 * q] = s_
                            rX[v, i] = s_ + rB[v, i]
                    rBits = cute.recast_tensor(rX, Int32)
                    keys = cute.make_rmem_tensor(32, Int64)
                    for i in cutlass.range_constexpr(8):
                        for v in cutlass.range_constexpr(4):
                            bits = rBits[v, i]
                            ordv = _sel(
                                bits >= 0, bits, bits ^ Int32(0x7FFFFFFF), Int32
                            )
                            col = Int32(16 * i + v) + 4 * q
                            keys[i * 4 + v] = (Int64(ordv) << 32) | Int64(
                                Int32(E_ROUTER - 1) - col
                            )
                    acc = cute.make_rmem_tensor(8, Int64)
                    _sort8(keys, 0)
                    for i in cutlass.range_constexpr(8):
                        acc[i] = keys[i]
                    for c in cutlass.range_constexpr(1, 4):
                        _sort8(keys, 8 * c)
                        _merge_top8(acc, keys, 8 * c)
                    for off in cutlass.range_constexpr(2):
                        oth = cute.make_rmem_tensor(8, Int64)
                        for i in cutlass.range_constexpr(8):
                            oth[i] = cute.arch.shuffle_sync_bfly(
                                acc[i], offset=1 << off
                            )
                        _merge_top8(acc, oth, 0)
                    # gather order of torch.topk: index order, the k-th-value ties last. key = (eq << 7 | idx) << 32 | bits
                    kth = Int32(acc[7] >> 32)
                    g = cute.make_rmem_tensor(8, Int64)
                    for i in cutlass.range_constexpr(8):
                        hv = Int32(acc[i] >> 32)
                        ix = Int32(E_ROUTER - 1) - Int32(acc[i] & Int64(E_ROUTER - 1))
                        gk = ix + _sel(hv == kth, Int32(E_ROUTER), Int32(0), Int32)
                        fb = _sel(hv >= 0, hv, hv ^ Int32(0x7FFFFFFF), Int32)
                        g[i] = (Int64(gk) << 32) | (Int64(fb) & Int64(0xFFFFFFFF))
                    _sort8_up(g)
                    rVb = cute.make_rmem_tensor(8, Int32)
                    rI = cute.make_rmem_tensor(8, Int32)
                    for i in cutlass.range_constexpr(8):
                        rVb[i] = Int32(g[i] & Int64(0xFFFFFFFF))
                        rI[i] = Int32(g[i] >> 32) & Int32(E_ROUTER - 1)
                    rV = cute.recast_tensor(rVb, Float32)
                    # SmallBitonicSort, folded (see _torch_topk_net)
                    _torch_net(rV, rI)
                    # this lane's two slots: 2q, 2q + 1
                    c0 = rI[_NET_OUT[0]]
                    c1 = rI[_NET_OUT[1]]
                    for j in cutlass.range_constexpr(1, 4):
                        c0 = _sel(q == j, rI[_NET_OUT[2 * j]], c0, Int32)
                        c1 = _sel(q == j, rI[_NET_OUT[2 * j + 1]], c1, Int32)
                    p0 = _sigmoid(mLogits[tl, c0])
                    p1 = _sigmoid(mLogits[tl, c1])
                    # all 8 picked scores on every lane of the token's quad, then ATen's add tree
                    pv = cute.make_rmem_tensor(8, Float32)
                    qbase = lane - q
                    for j in cutlass.range_constexpr(4):
                        pv[2 * j] = cute.arch.shuffle_sync(p0, offset=qbase + j)
                        pv[2 * j + 1] = cute.arch.shuffle_sync(p1, offset=qbase + j)
                    tot = _tsum(pv, sum_tree)
                    g0 = _div(p0, tot)
                    g1 = _div(p1, tot)
                    if valid:
                        j0 = 2 * q
                        mTopIdx[t, j0] = Int64(c0)
                        mTopIdx[t, j0 + 1] = Int64(c1)
                        mGate[t, j0] = g0.to(BFloat16)
                        mGate[t, j0 + 1] = g1.to(BFloat16)
                        if const_expr(self.debug):
                            mDbgGate[t, j0] = g0
                            mDbgGate[t, j0 + 1] = g1
                        pas = Int32(0)
                        if const_expr(P == 2):
                            pas = _sel(t >= t0, Int32(1), Int32(0), Int32)
                        cute.arch.atomic_add(
                            _ptr(sHist, warp * S + c0 * P + pas), Int32(1)
                        )
                        cute.arch.atomic_add(
                            _ptr(sHist, warp * S + c1 * P + pas), Int32(1)
                        )
                cute.arch.sync_threads()

                # per-warp (expert, pass) counts -> global rows bidx * W + warp (the kernel boundary is the
                # grid-wide barrier: no co-residency assumption)
                if tidx < S:
                    for w in cutlass.range_constexpr(W):
                        mHist[bidx * W + w, tidx] = sHist[w * S + tidx]

            if const_expr(self.phase == 2):
                # ---- phase 2: segment starts, this CTA's base, per-warp cursors ---------------------------------
                tot_s = Int32(0)
                pre_s = Int32(0)
                if tidx < S:
                    for c in cutlass.range(U):
                        hv = mHist[c, tidx]
                        tot_s += hv
                        pre_s += _sel(c < bidx * W, hv, Int32(0), Int32)
                incl = tot_s
                for i in cutlass.range_constexpr(5):
                    o = 1 << i
                    up = cute.arch.shuffle_sync_up(incl, offset=o, mask_and_clamp=0)
                    incl += _sel(lane >= o, up, Int32(0), Int32)
                if lane == 31:
                    sWarp[warp] = incl
                cute.arch.sync_threads()
                excl_w = Int32(0)
                for w in cutlass.range_constexpr(W):
                    excl_w += _sel(warp > w, sWarp[w], Int32(0), Int32)
                start = excl_w + incl - tot_s
                if tidx < S:
                    run = start + pre_s
                    for w in cutlass.range_constexpr(W):
                        cnt = mHist[bidx * W + w, tidx]
                        sHist[w * S + tidx] = run
                        run += cnt
                    if bidx == 0:
                        mSegCsr[tidx] = start
                        if tidx == S - 1:
                            mSegCsr[S] = start + tot_s
                        if tidx % P == P - 1:
                            mOffsets[tidx // P] = start + tot_s
                cute.arch.sync_threads()

                # ---- phase 3: stable scatter, 4 tokens x 8 slots per step ---------------------------------------
                slot = lane % 8
                tk = lane // 8
                for base in cutlass.range(lo, hi, 4):
                    t = base + tk
                    valid = t < hi
                    tl = cutlass.min(t, hi - 1)
                    e = Int32(mTopIdx[tl, slot])
                    gv = mGate[tl, slot]
                    pas = Int32(0)
                    if const_expr(P == 2):
                        pas = _sel(tl >= t0, Int32(1), Int32(0), Int32)
                    sidx = warp * S + e * P + pas
                    for k in cutlass.range_constexpr(4):
                        if valid & (tk == k):
                            pos = sHist[sidx]
                            sHist[sidx] = pos + 1
                            f = t * TOP_K + slot
                            mOrder[pos] = Int64(f)
                            mInverse[f] = Int64(pos)
                            mSortedGate[pos] = gv
                            mTok64[pos] = Int64(t)
                            mTok32[pos] = t
                        cute.arch.sync_warp()

    return RouteDispatch


_COMPILED: dict = {}
_WS: dict = {}


def _compiled(
    P: int, debug: bool = False, div_mode: str | None = None, tree=None, phase: int = 1
):
    div_mode = div_mode or DIV_MODE
    tree = SUM_TREE if tree is None else tree
    if tree is None:
        raise RuntimeError("router_fused.SUM_TREE is not set")
    key = (torch.cuda.current_device(), P, debug, div_mode, repr(tree), phase)
    fn = _COMPILED.get(key)
    if fn is not None:
        return fn
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64

    def fake(dtype, shape, divisibility=1):
        stride = tuple(
            cute.sym_int64(divisibility=divisibility) if i != len(shape) - 1 else 1
            for i in range(len(shape))
        )
        return cute.runtime.make_fake_tensor(
            dtype,
            shape,
            stride=stride,
            assumed_align=max(divisibility * dtype.width // 8, 1),
        )

    cls = _build_kernel_class(div_mode, tree)
    t_sym = cute.sym_int()
    r_sym = cute.sym_int()
    g_sym = cute.sym_int()
    d_sym = cute.sym_int()
    S = P * E_ROUTER
    fn = cute.compile(
        cls(P, debug, phase),
        fake(Float32, (t_sym, E_ROUTER), 4),  # logits
        fake(Float32, (E_ROUTER,), 4),  # bias
        fake(Int64, (t_sym, TOP_K), 2),  # top_idx
        fake(BFloat16, (t_sym, TOP_K), 8),  # gate
        fake(Int64, (r_sym,), 1),  # order
        fake(Int64, (r_sym,), 1),  # inverse
        fake(BFloat16, (r_sym,), 1),  # sorted_gate
        fake(Int64, (r_sym,), 1),  # tokens
        fake(Int32, (r_sym,), 1),  # tokens32
        fake(Int32, (S + 1,), 1),  # seg_csr
        fake(Int32, (E_ROUTER,), 1),  # offsets
        fake(Int32, (g_sym, S), 1),  # per-warp histograms (G * warps, S)
        fake(Float32, (d_sym, E_ROUTER), 1),  # debug: sigmoid scores
        fake(Float32, (d_sym, TOP_K), 1),  # debug: fp32 gates
        Int32(0),
        Int32(0),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _COMPILED[key] = fn
    return fn


def _workspace(device: torch.device, P: int):
    key = (device.index, P)
    ws = _WS.get(key)
    if ws is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ws = (
            sms,
            torch.empty(1, E_ROUTER, dtype=torch.float32, device=device),
            torch.empty(1, TOP_K, dtype=torch.float32, device=device),
        )
        _WS[key] = ws
    return ws


def num_ctas(T: int, sms: int) -> int:
    # >= 32 tokens per warp keeps phase 1's 8-token steps busy; never more CTAs than SMs (co-residency).
    return max(1, min(sms, -(-T // (_WARPS * 32))))


def supported(logits_: Tensor, bias: Tensor, rows) -> bool:
    return (
        logits_.is_cuda
        and logits_.dtype == torch.float32
        and logits_.dim() == 2
        and logits_.shape[1] == E_ROUTER
        and logits_.is_contiguous()
        and bias.dtype == torch.float32
        and bias.is_contiguous()
        and bias.numel() == E_ROUTER
        and bias.device == logits_.device
        and len(rows) in (1, 2)
        and sum(rows) == logits_.shape[0]
        and min(rows) > 0
        and logits_.shape[0] * TOP_K < 2**31
        and is_blackwell_tcgen05(logits_.device)
    )


def route_dispatch(
    logits_: Tensor,
    bias: Tensor,
    rows,
    debug: bool = False,
    div_mode: str | None = None,
    tree=None,
):
    """The fused routing + dispatch of ``logits_`` (T, 128) fp32 (see the module docstring). ``debug``: also
    returns the kernel's sigmoid scores (T, 128) and fp32 gates (T, 8)."""
    rows = tuple(int(r) for r in rows)
    P = len(rows)
    T = logits_.shape[0]
    dev = logits_.device
    n = T * TOP_K
    top_idx = torch.empty(T, TOP_K, dtype=torch.int64, device=dev)
    gate = torch.empty(T, TOP_K, dtype=torch.bfloat16, device=dev)
    order = torch.empty(n, dtype=torch.int64, device=dev)
    inverse = torch.empty(n, dtype=torch.int64, device=dev)
    sorted_gate = torch.empty(n, dtype=torch.bfloat16, device=dev)
    tokens = torch.empty(n, dtype=torch.int64, device=dev)
    tokens32 = torch.empty(n, dtype=torch.int32, device=dev)
    seg_csr = torch.empty(P * E_ROUTER + 1, dtype=torch.int32, device=dev)
    offsets = torch.empty(E_ROUTER, dtype=torch.int32, device=dev)
    sms, dsc, dg = _workspace(dev, P)
    if debug:
        dsc = torch.empty(T, E_ROUTER, dtype=torch.float32, device=dev)
        dg = torch.empty(T, TOP_K, dtype=torch.float32, device=dev)
    G = num_ctas(T, sms)
    t0 = rows[0] if P == 2 else T
    # Per-call histogram rows from the caching allocator (stream-ordered; no state shared between launches).
    hist = torch.empty(G * _WARPS, P * E_ROUTER, dtype=torch.int32, device=dev)
    args = (
        logits_,
        bias,
        top_idx,
        gate,
        order,
        inverse,
        sorted_gate,
        tokens,
        tokens32,
        seg_csr,
        offsets,
        hist,
        dsc,
        dg,
        T,
        t0,
        G,
    )
    with torch.cuda.device(dev):
        _compiled(P, debug, div_mode, tree, phase=1)(*args)
        _compiled(P, debug, div_mode, tree, phase=2)(*args)
    r = Routing(
        top_idx,
        gate,
        order,
        inverse,
        sorted_gate,
        tokens,
        tokens32,
        seg_csr,
        offsets,
        rows,
    )
    return (r, dsc, dg) if debug else r


# ---------------------------------------------------------------------------------------------------------------
# Torch reference (stock ops) for probes and the routing audit
# ---------------------------------------------------------------------------------------------------------------
def route_ref(logits_: Tensor, bias: Tensor, top_k: int = TOP_K, dtype=torch.bfloat16):
    """MoEFeedForward._route after the logits (same ops, same order)."""
    scores = torch.sigmoid(logits_)
    biased = scores + bias.to(scores.dtype)
    _, top_idx = torch.topk(biased, top_k, dim=-1)
    picked = scores.gather(dim=-1, index=top_idx)
    gate = picked / picked.sum(dim=-1, keepdim=True)
    return top_idx, gate.to(dtype)


def dispatch_ref(
    top_idx: Tensor, gate: Tensor, rows, num_experts: int = E_ROUTER
) -> Routing:
    """The stock dispatch metadata of ``top_idx`` / ``gate`` (P = 1: _dispatch_rows / moe_dispatch_metadata;
    P = 2: nvfp4_pair's sort by 2 e + pass = the BF16 pair's argsort of the concatenation)."""
    rows = tuple(int(r) for r in rows)
    P = len(rows)
    T, k = top_idx.shape
    dev = top_idx.device
    pas = torch.zeros(T, dtype=torch.int64, device=dev)
    if P == 2:
        pas[rows[0] :] = 1
    keys = (top_idx * P + pas[:, None]).reshape(-1)
    _, order = keys.sort(stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=dev)
    counts = torch.zeros(P * num_experts, dtype=torch.int64, device=dev)
    counts.scatter_add_(0, keys, torch.ones_like(keys))
    seg_csr = torch.nn.functional.pad(counts.cumsum(0).to(torch.int32), (1, 0))
    offsets = seg_csr[P::P].contiguous()
    tokens = order // k
    return Routing(
        top_idx,
        gate,
        order,
        inverse,
        gate.reshape(-1)[order],
        tokens,
        tokens.to(torch.int32),
        seg_csr,
        offsets,
        rows,
    )
