"""Optional inference attention backends, loaded only when selected.

Plans and converted masks retain their source masks and use bounded caches.
Warm each shape before CUDA Graph capture. Masks must remain immutable.
"""

from collections import OrderedDict
from functools import partial

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from praxis.hardware import is_thor

# Flash compiles each attention shape separately. Raise Dynamo's process-wide
# limits when FlashFlex is created, before warmup. None keeps the caller's limits.
FLASH_RECOMPILE_LIMIT: int | None = 64

# Thor (SM110) only. True: Flash masks use 256-row blocks on Thor as on SM100,
# so FA4 block sparsity runs there. False keeps the checkpoint's 128-row mask
# blocks on Thor; Attention.effective_backend then selects Triton, whose outputs
# are bitwise equal to the Triton (non-Flash) reference. Other GPUs ignore it.
SM110_FLASH_MASK_256 = True


def flash_mask_block_size(device: torch.device | str, block_size: int = 128) -> int:
    """Fit sparse query blocks to Blackwell's two 128-row query stages."""
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] in (
        10,
        11,
    ):
        if not SM110_FLASH_MASK_256 and is_thor(device):
            return block_size
        return ((block_size + 255) // 256) * 256
    return block_size


def _interval_endpoints(block_mask, sq, sk, device):
    """Encode visible key intervals as ``[0,F0) U [F1,F2) U ...``."""
    if block_mask is not None and tuple(block_mask.kv_num_blocks.shape[:2]) != (1, 1):
        raise ValueError("cuDNN attention requires a batch-one, shared-head mask")
    endpoints = []
    for start in range(0, sq, 128):
        qi = torch.arange(start, min(start + 128, sq), device=device)[:, None]
        ki = torch.arange(sk, device=device)[None, :]
        visible = (
            block_mask.mask_mod(0, 0, qi, ki)
            if block_mask is not None
            else torch.ones((qi.numel(), sk), device=device, dtype=torch.bool)
        )
        visible = visible.expand(qi.numel(), sk)
        # Start visible at the left sentinel and hidden at the right. Every
        # row then has an odd number of transitions, including an endpoint
        # at zero when its first visible interval starts after key zero.
        padded = torch.cat(
            [
                torch.ones_like(visible[:, :1]),
                visible,
                torch.zeros_like(visible[:, :1]),
            ],
            dim=1,
        )
        rows, columns = torch.where(padded[:, :-1] != padded[:, 1:])
        counts = torch.bincount(rows, minlength=visible.shape[0])
        offsets = counts.cumsum(0) - counts
        slots = torch.arange(columns.numel(), device=device) - offsets[rows]
        end = torch.full(
            (visible.shape[0], int(counts.max().item())),
            sk,
            dtype=torch.int32,
            device=device,
        )
        end[rows, slots] = columns.to(torch.int32)
        endpoints.append(end)
    # Repeated endpoints at sk add empty intervals. They preserve each row
    # while allowing rows with different interval counts in one plan.
    width = max(end.shape[1] for end in endpoints)
    padded = [
        torch.nn.functional.pad(end, (0, width - end.shape[1]), value=sk)
        for end in endpoints
    ]
    return torch.cat(padded).T.unsqueeze(0).contiguous()


class CudnnFlex:
    """cuDNN Flex adapter for batch-one serve interval masks."""

    def __init__(self):
        try:
            from cudnn import create_mask_plan, flex_attn_func
        except ImportError as exc:
            raise ImportError(
                "cuDNN attention requires pip install 'praxis[cudnn]'"
            ) from exc
        self.create_mask_plan = create_mask_plan
        self.flex_attn_func = flex_attn_func
        self.plans = OrderedDict()

    def __call__(self, q, k, v, *, block_mask=None, scale=None, enable_gqa=False):
        if q.shape[0] != 1:
            raise ValueError("cuDNN attention supports batch-one inference")
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
            raise ValueError(
                "cuDNN attention is inference-only; use torch.inference_mode()"
            )
        qb, kb, vb = (t.transpose(1, 2) for t in (q, k, v))
        key = (
            id(block_mask),
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            q.stride(),
            k.stride(),
            v.stride(),
            q.dtype,
            q.device,
        )
        if key not in self.plans:
            endpoints = _interval_endpoints(
                block_mask, q.shape[2], k.shape[2], q.device
            )
            plan = self.create_mask_plan(endpoints, qb, kb, vb, build_backward=False)
            self.plans[key] = (block_mask, plan)
            if len(self.plans) > 64:
                self.plans.popitem(last=False)
        self.plans.move_to_end(key)
        return self.flex_attn_func(
            qb, kb, vb, mask_plan=self.plans[key][1], softmax_scale=scale
        ).transpose(1, 2)


class FlashFlex:
    """Flash attention with equivalent full-block metadata for existing serve masks."""

    def __init__(self):
        if FLASH_RECOMPILE_LIMIT is not None:
            import torch._dynamo.config as dynamo_config

            limit = FLASH_RECOMPILE_LIMIT
            dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, limit)
            dynamo_config.accumulated_recompile_limit = max(
                dynamo_config.accumulated_recompile_limit, 16 * limit
            )
        self.compiled = torch.compile(
            partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        self.masks = OrderedDict()

    def __call__(self, q, k, v, *, block_mask=None, scale=None, enable_gqa=False):
        converted = block_mask
        if block_mask is not None and block_mask.full_kv_num_blocks is None:
            sq, sk = q.shape[2], k.shape[2]
            key = (id(block_mask), sq, sk, q.device)
            if key not in self.masks:
                original = block_mask.mask_mod

                def bounded(b, h, qi, ki):
                    return (
                        (qi < sq)
                        & (ki < sk)
                        & original(b, h, qi.clamp_max(sq - 1), ki.clamp_max(sk - 1))
                    )

                converted = create_block_mask(
                    bounded,
                    *block_mask.kv_num_blocks.shape[:2],
                    sq,
                    sk,
                    device=q.device,
                    BLOCK_SIZE=block_mask.BLOCK_SIZE,
                    separate_full_blocks=True,
                )
                self.masks[key] = (block_mask, converted)
                if len(self.masks) > 64:
                    self.masks.popitem(last=False)
            self.masks.move_to_end(key)
            converted = self.masks[key][1]
        return self.compiled(
            q, k, v, block_mask=converted, scale=scale, enable_gqa=enable_gqa
        )


_backends = {}


def optional_attention(backend, q, k, v, **kwargs):
    """Share plans across transformer layers without modifying global dispatch."""
    if backend not in ("flash", "cudnn"):
        raise ValueError(f"unknown optional attention backend {backend!r}")
    if backend not in _backends:
        _backends[backend] = CudnnFlex() if backend == "cudnn" else FlashFlex()
    return _backends[backend](q, k, v, **kwargs)
