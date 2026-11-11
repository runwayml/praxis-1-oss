"""Video autoencoder (Wan2.2 VAE) for praxis.

The autoencoder maps RGB video to latents and back. Shapes:

* Video: ``(B, 3, T, H, W)`` with ``T = 1 + 4k``. ``H`` and ``W`` are
  multiples of 32.
* Latent: ``(B, 48, 1 + k, H / 16, W / 16)``.
* Policy latent: the latent with 2x2 spatial patches moved into channels,
  ``(B, 192, 1 + k, H / 32, W / 32)``. The video backbone reads this form.

Numerics follow the original implementation:

* The weights are rounded twice: float32 to float16, then to bfloat16.
* The convolutions run under float16 autocast.
* The latent normalisation ``(mu - mean) / std`` uses float32 constants.

The encoder has two paths:

* ``encode`` (default) is the fused path. It runs channels-last and uses the
  custom Triton kernels in ``praxis.kernels.vae_encoder_triton``. Its output
  is bitwise equal to the encoder of the original serve code on H100 GPUs.
* ``encode_eager`` is plain PyTorch. It needs no Triton. Its output is not
  bitwise equal to the fused path. The rounding differs.

On a CUDA device, ``Autoencoder.from_pretrained`` (and the pipeline)
default to the fused decoder: fused RMSNorm/SiLU, implicit spatial padding and
streamed RGB output with FP16 convolutions. Other devices, and an
``Autoencoder`` built directly, use the original plain PyTorch decoder
(``decoder_impl="reference"``); fused decoding requires numerical validation
rather than bitwise reference parity.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from praxis.kernels import vae_encoder_triton as K

Z_DIM = 48
CACHE_T = 2

# True: the fused encoder keeps its causal caches as views of the
# halo and output buffers they come from instead of clones. Those buffers are
# never written after the cache is taken, so the values are the same (no
# arithmetic changes); this removes one device copy per cache per chunk.
# Default on.
ENCODER_CACHE_VIEWS = True

# True: the fused encoder hands its convolutions float16 copies
# of the bf16 weights and biases, made once, instead of letting float16 autocast
# cast them on every call; the input chunk buffer is built in float16 directly.
# The casts are the same ``.to(float16)`` autocast applies, so values are equal.
# Default on.
VAE_HOIST_CASTS = True

# True: in a down block, the residual sum ``x + h`` of a residual block that feeds
# another residual block is not computed by itself; the next block's first norm
# computes it (K.add_rmsnorm_silu: the same fp16 add, stored once, then the norm
# of ``_rmsnorm_silu_kernel``). Default on.
VAE_FUSED_ADD_NORM = True


def _keep(t: Tensor) -> Tensor:
    """A fused-encoder cache entry: ``t`` itself (ENCODER_CACHE_VIEWS) or a clone."""
    return t if ENCODER_CACHE_VIEWS else t.clone()


CL3D = torch.channels_last_3d

# Latent normalisation constants of the Wan2.2 VAE.
LATENT_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)  # fmt: skip
LATENT_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)  # fmt: skip


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class CausalConv3d(nn.Conv3d):
    """3-D convolution that pads the time axis on the past side only."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # ty: ignore[invalid-argument-type]
        pt, ph, pw = self.padding  # ty: ignore[not-iterable]
        self._pad = (pw, pw, ph, ph, 2 * pt, 0)
        self.padding = (0, 0, 0)
        self._decoder_adapter = None

    def forward(self, x: Tensor, cache: Tensor | None = None) -> Tensor:
        if self._decoder_adapter is not None:
            return self._decoder_adapter(self, x, cache)
        pad = list(self._pad)
        if cache is not None and pad[4] > 0:
            x = torch.cat([cache.to(x.device), x], dim=2)
            pad[4] -= cache.shape[2]
        return super().forward(F.pad(x, pad))


class RMSNorm(nn.Module):
    """Channel RMS norm: ``normalize(x) * sqrt(C) * gamma``."""

    def __init__(self, dim: int, images: bool = True) -> None:
        super().__init__()
        shape = (dim, 1, 1) if images else (dim, 1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))

    def forward(self, x: Tensor) -> Tensor:
        # The original implementation adds a zero bias. The add turns -0.0 into +0.0.
        return F.normalize(x, dim=1) * self.scale * self.gamma + 0.0


def _time_cache(x: Tensor, cache: Tensor | None) -> Tensor:
    """Return the last ``CACHE_T`` frames of ``x`` as the next causal context.

    When ``x`` has one frame, the last frame of the old context comes first.
    """
    new = x[:, :, -CACHE_T:].clone()
    if new.shape[2] < CACHE_T and cache is not None:
        new = torch.cat([cache[:, :, -1:].to(new.device), new], dim=2)
    return new


class Caches:
    """Per-convolution causal contexts for one streamed encode or decode."""

    def __init__(self, n: int) -> None:
        self.items: list[Tensor | str | None] = [None] * n
        self.index = 0

    def next_index(self) -> int:
        i = self.index
        self.index += 1
        return i


def _conv_with_cache(conv: CausalConv3d, x: Tensor, caches: Caches) -> Tensor:
    i = caches.next_index()
    old = caches.items[i]
    old_t = old if isinstance(old, Tensor) else None
    adapter = conv._decoder_adapter
    if getattr(adapter, "fuse_cache_input", False) and adapter.supports(x, old_t):
        out, new = adapter.with_cache(conv, x, old_t)
        caches.items[i] = new
        return out
    new = _time_cache(x, old_t)
    out = conv(x, old_t)
    caches.items[i] = new
    return out


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(0.0),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut: nn.Module = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: Tensor, caches: Caches) -> Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d):
                x = _conv_with_cache(layer, x, caches)
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head self-attention over the pixels of each frame."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class Upsample(nn.Upsample):
    """Nearest upsampling in float32."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """2x spatial down- or upsampling, with 2x time resampling in 3-D mode."""

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
                )
        else:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

    def _upsample_time(self, x: Tensor, caches: Caches) -> Tensor:
        b, c, t, h, w = x.shape
        i = caches.next_index()
        old = caches.items[i]
        if old is None:
            # The first chunk holds the anchor frame. It is not doubled.
            caches.items[i] = "first"
            return x
        new = x[:, :, -CACHE_T:].clone()
        if new.shape[2] < CACHE_T:
            prev = (
                old[:, :, -1:].to(new.device)
                if isinstance(old, Tensor)
                else torch.zeros_like(new)
            )
            new = torch.cat([prev, new], dim=2)
        x = self.time_conv(x, old) if isinstance(old, Tensor) else self.time_conv(x)
        caches.items[i] = new
        x = x.reshape(b, 2, c, t, h, w)
        x = torch.stack((x[:, 0], x[:, 1]), 3)
        return x.reshape(b, c, t * 2, h, w)

    def forward(self, x: Tensor, caches: Caches) -> Tensor:
        if self.mode == "upsample3d":
            x = self._upsample_time(x, caches)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, *x.shape[1:]).permute(0, 2, 1, 3, 4)
        if self.mode == "downsample3d":
            i = caches.next_index()
            old = caches.items[i]
            if old is None:
                caches.items[i] = x.clone()
            else:
                assert isinstance(old, Tensor)
                new = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([old[:, :, -1:], x], 2))
                caches.items[i] = new
        return x


class AvgDown3D(nn.Module):
    """Average space-time patches and fold them into channels."""

    def __init__(self, in_dim: int, out_dim: int, factor_t: int, factor_s: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        self.group_size = in_dim * self.factor // out_dim

    def forward(self, x: Tensor) -> Tensor:
        ft, fs = self.factor_t, self.factor_s
        pad_t = (ft - x.shape[2] % ft) % ft
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t // ft, h // fs, w // fs)
        x = x.view(b, self.out_dim, self.group_size, t // ft, h // fs, w // fs)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    """Repeat channels into space-time patches (the inverse of AvgDown3D)."""

    def __init__(self, in_dim: int, out_dim: int, factor_t: int, factor_s: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.repeats = out_dim * factor_t * factor_s * factor_s // in_dim

    def forward(self, x: Tensor, first_chunk: bool) -> Tensor:
        ft, fs = self.factor_t, self.factor_s
        x = x.repeat_interleave(self.repeats, dim=1)
        b, _, t, h, w = x.shape
        x = x.view(b, self.out_dim, ft, fs, fs, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, self.out_dim, t * ft, h * fs, w * fs)
        if first_chunk:
            x = x[:, :, ft - 1 :]
        return x


class DownBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, mult: int, down_t: bool, down_s: bool
    ) -> None:
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim, 2 if down_t else 1, 2 if down_s else 1
        )
        layers: list[nn.Module] = []
        for _ in range(mult):
            layers.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if down_s:
            layers.append(
                Resample(out_dim, "downsample3d" if down_t else "downsample2d")
            )
        self.downsamples = nn.Sequential(*layers)

    def forward(self, x: Tensor, caches: Caches) -> Tensor:
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, caches)
        return x + self.avg_shortcut(x_copy)


class UpBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, mult: int, up_t: bool, up_s: bool
    ) -> None:
        super().__init__()
        self.avg_shortcut = (
            DupUp3D(in_dim, out_dim, 2 if up_t else 1, 2) if up_s else None
        )
        layers: list[nn.Module] = []
        for _ in range(mult):
            layers.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if up_s:
            layers.append(Resample(out_dim, "upsample3d" if up_t else "upsample2d"))
        self.upsamples = nn.Sequential(*layers)

    def forward(self, x: Tensor, caches: Caches, first_chunk: bool) -> Tensor:
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, caches)
        if self.avg_shortcut is None:
            return x_main
        return x_main + self.avg_shortcut(x, first_chunk)


class Encoder(nn.Module):
    def __init__(self, dim: int = 160, z_dim: int = 2 * Z_DIM) -> None:
        super().__init__()
        mult = [1, 2, 4, 4]
        down_t = [False, True, True]
        dims = [dim * u for u in [1, *mult]]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        self.downsamples = nn.Sequential(
            *[
                DownBlock(
                    dims[i],
                    dims[i + 1],
                    2,
                    down_t[i] if i < len(down_t) else False,
                    i != len(mult) - 1,
                )
                for i in range(len(mult))
            ]
        )
        out_dim = dims[-1]
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim),
        )
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x: Tensor, caches: Caches) -> Tensor:
        x = _conv_with_cache(self.conv1, x, caches)
        for layer in self.downsamples:
            x = layer(x, caches)
        for layer in self.middle:
            x = layer(x, caches) if isinstance(layer, ResidualBlock) else layer(x)
        x = self.head[1](self.head[0](x))
        return _conv_with_cache(self.head[2], x, caches)


class Decoder(nn.Module):
    def __init__(self, dim: int = 256, z_dim: int = Z_DIM) -> None:
        super().__init__()
        mult = [1, 2, 4, 4]
        up_t = [True, True, False]
        dims = [dim * u for u in [mult[-1], *mult[::-1]]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0]),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0]),
        )
        self.upsamples = nn.Sequential(
            *[
                UpBlock(
                    dims[i],
                    dims[i + 1],
                    3,
                    up_t[i] if i < len(up_t) else False,
                    i != len(mult) - 1,
                )
                for i in range(len(mult))
            ]
        )
        self.head = nn.Sequential(
            RMSNorm(dims[-1], images=False),
            nn.SiLU(),
            CausalConv3d(dims[-1], 12, 3, padding=1),
        )

    def forward(self, x: Tensor, caches: Caches, first_chunk: bool) -> Tensor:
        x = _conv_with_cache(self.conv1, x, caches)
        for layer in self.middle:
            x = layer(x, caches) if isinstance(layer, ResidualBlock) else layer(x)
        for layer in self.upsamples:
            x = layer(x, caches, first_chunk)
        x = self.head[1](self.head[0](x))
        return _conv_with_cache(self.head[2], x, caches)


def count_causal_convs(module: nn.Module) -> int:
    return sum(isinstance(m, CausalConv3d) for m in module.modules())


def patchify(x: Tensor, p: int = 2) -> Tensor:
    """``(B, C, T, H, W)`` to ``(B, C*p*p, T, H/p, W/p)``. Channel order (c, w, h)."""
    b, c, t, h, w = x.shape
    x = x.view(b, c, t, h // p, p, w // p, p)
    return x.permute(0, 1, 6, 4, 2, 3, 5).reshape(b, c * p * p, t, h // p, w // p)


def unpatchify(x: Tensor, p: int = 2) -> Tensor:
    """The inverse of ``patchify``."""
    b, cpp, t, h, w = x.shape
    c = cpp // (p * p)
    x = x.view(b, c, p, p, t, h, w)
    return x.permute(0, 1, 4, 5, 3, 6, 2).reshape(b, c, t, h * p, w * p)


def to_policy_latent(z: Tensor) -> Tensor:
    """``(B, 48, T, h, w)`` to the policy form ``(B, 192, T, h/2, w/2)``."""
    b, c, t, h, w = z.shape
    z = z.view(b, c, t, h // 2, 2, w // 2, 2)
    return z.permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * 4, t, h // 2, w // 2)


def from_policy_latent(z: Tensor) -> Tensor:
    """The inverse of ``to_policy_latent``."""
    b, c4, t, h, w = z.shape
    z = z.view(b, c4 // 4, 2, 2, t, h, w)
    return z.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, c4 // 4, t, h * 2, w * 2)


# ---------------------------------------------------------------------------
# Fused channels-last encoder
# ---------------------------------------------------------------------------


class _FusedNormSiLU:
    """``silu(rmsnorm(x))`` with a custom kernel, optionally into a halo buffer."""

    def __init__(self, norm: RMSNorm, block_c: int) -> None:
        self.sqrt_c = float(norm.scale)
        self.gamma = norm.gamma.detach().reshape(-1).float().contiguous()
        self.block_c = block_c

    def __call__(self, x: Tensor, cache: Tensor | None, halo: bool) -> Tensor:
        x = (
            x
            if x.is_contiguous(memory_format=CL3D)
            else x.contiguous(memory_format=CL3D)
        )
        b, c, t, h, w = x.shape
        if not halo:
            y = torch.empty_like(x, memory_format=CL3D)
            K.rmsnorm_silu(x, y, self.gamma, self.sqrt_c, self.block_c)
            return y
        y = torch.empty(
            (b, c, t + CACHE_T, h, w),
            dtype=x.dtype,
            device=x.device,
            memory_format=CL3D,
        )
        if cache is None:
            y[:, :, :CACHE_T].zero_()
        else:
            y[:, :, :CACHE_T].copy_(cache)
        # Pass the complete halo allocation to the mutating Triton kernel.
        # Functionalizing a write through its body view can otherwise shrink
        # the allocation while a generated clone still reads the full storage.
        K.rmsnorm_silu(
            x, y, self.gamma, self.sqrt_c, self.block_c, output_t_offset=CACHE_T
        )
        return y

    def add_halo(
        self, a: Tensor, b: Tensor, cache: Tensor | None
    ) -> tuple[Tensor, Tensor] | None:
        """``(a + b, self(a + b, cache, halo=True))`` from one kernel, or None if
        ``a`` and ``b`` are not channels-last tensors of one shape and dtype
        (VAE_FUSED_ADD_NORM)."""
        if not (
            a.shape == b.shape
            and a.dtype == b.dtype
            and a.is_contiguous(memory_format=CL3D)
            and b.is_contiguous(memory_format=CL3D)
            and a.stride() == b.stride()
        ):
            return None
        bsz, c, t, h, w = a.shape
        s = torch.empty_like(a)
        if s.stride() != a.stride():
            return None
        y = torch.empty(
            (bsz, c, t + CACHE_T, h, w),
            dtype=a.dtype,
            device=a.device,
            memory_format=CL3D,
        )
        if cache is None:
            y[:, :, :CACHE_T].zero_()
        else:
            y[:, :, :CACHE_T].copy_(cache)
        K.add_rmsnorm_silu(
            a, b, s, y, self.gamma, self.sqrt_c, self.block_c, output_t_offset=CACHE_T
        )
        return s, y


def _conv_cl(
    conv: nn.Conv3d, x: Tensor, t_pad: int, spatial: tuple[int, int]
) -> Tensor:
    """Channels-last causal conv: explicit past padding, spatial padding in cuDNN."""
    if t_pad > 0:
        x = F.pad(x, (0, 0, 0, 0, t_pad, 0))
    return F.conv3d(
        x,
        conv.weight,
        conv.bias,
        conv.stride,
        (0, *spatial),
        conv.dilation,
        conv.groups,
    )


class FusedEncoder:
    """The fused encoder path, built once from a loaded ``Encoder``.

    It works channels-last, folds the latent head and the normalisation into
    the last convolution, and runs the norm and pooling steps as custom
    kernels. The math is the math of ``Encoder`` plus ``Autoencoder.encode``.
    Only the rounding differs from the eager path.
    """

    def __init__(
        self,
        enc: Encoder,
        conv_mu: CausalConv3d,
        scale: tuple[Tensor, Tensor],
        norm_block_c: dict[int, int],
    ) -> None:
        self.enc = enc
        self.n_caches = count_causal_convs(enc)
        self._prepadded: dict[int, Tensor] = {}
        self.norms: dict[int, _FusedNormSiLU] = {}

        def fused(norm: nn.Module) -> _FusedNormSiLU:
            assert isinstance(norm, RMSNorm)
            return _FusedNormSiLU(norm, norm_block_c[norm.gamma.shape[0]])

        for m in enc.modules():
            if isinstance(m, ResidualBlock):
                self.norms[id(m.residual[0])] = fused(m.residual[0])
                self.norms[id(m.residual[3])] = fused(m.residual[3])
        self.norms[id(enc.head[0])] = fused(enc.head[0])
        # Head fold: the mu half of the 1x1x1 conv and the latent affine.
        # The original implementation folds inside its encode call, under bfloat16
        # autocast. The einsum and the matmul then run in bfloat16. The fold
        # here does the same to give the same folded weights.
        head = enc.head[2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = conv_mu.weight.detach().float()[:Z_DIM, :, 0, 0, 0]
            b1 = conv_mu.bias.detach().float()[:Z_DIM]
            w = torch.einsum("om,mikjl->oikjl", a, head.weight.detach().float())
            b = a @ head.bias.detach().float() + b1
            mean, inv_std = (s.detach().float().view(Z_DIM).to(w.device) for s in scale)
            w = w * inv_std.view(Z_DIM, 1, 1, 1, 1)
            b = (b - mean) * inv_std
        self.head_w = w.to(head.weight.dtype).contiguous(memory_format=CL3D)
        self.head_b = b.to(head.weight.dtype)
        self.weights: dict[int, Tensor] = {}
        for m in enc.modules():
            if isinstance(m, nn.Conv3d):
                self.weights[id(m)] = m.weight.detach().contiguous(memory_format=CL3D)
        self.resample_w: dict[int, Tensor] = {}
        for m in enc.modules():
            if isinstance(m, Resample):
                conv2d = m.resample[1]
                self.resample_w[id(m)] = (
                    conv2d.weight.detach().unsqueeze(2).contiguous(memory_format=CL3D)
                )

    def _h(self, t: Tensor | None) -> Tensor | None:
        """``t`` as float16 under float16 autocast (cached per tensor), else
        ``t`` (VAE_HOIST_CASTS)."""
        if (
            t is None
            or not VAE_HOIST_CASTS
            or t.dtype == torch.float16
            or not t.is_floating_point()
            or not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.float16
        ):
            return t
        cache = self.__dict__.setdefault("_half", {})
        hit = cache.get(id(t))
        if hit is None or hit[0] is not t:
            hit = (t, t.to(torch.float16))
            cache[id(t)] = hit
        return hit[1]

    def _conv(
        self, conv: CausalConv3d, x: Tensor, cache: Tensor | None = None
    ) -> Tensor:
        t_pad = conv._pad[4]
        if cache is not None and t_pad > 0:
            t_pad -= cache.shape[2]
            x = torch.cat([cache, x], dim=2)
        if t_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, t_pad, 0))
        return F.conv3d(
            x,
            self._h(self.weights[id(conv)]),
            self._h(conv.bias),
            conv.stride,
            (0, conv._pad[2], conv._pad[0]),
            conv.dilation,
            conv.groups,
        )

    def _conv_halo(self, conv: CausalConv3d, buf: Tensor) -> Tensor:
        """Convolve a halo buffer whose first frames hold the causal context."""
        return F.conv3d(
            buf,
            self._h(self.weights[id(conv)]),
            self._h(conv.bias),
            conv.stride,
            (0, conv._pad[2], conv._pad[0]),
            conv.dilation,
            conv.groups,
        )

    def _residual(
        self,
        rb: ResidualBlock,
        x: Tensor | tuple[Tensor, Tensor],
        caches: Caches,
        prepad: bool,
        defer: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """``rb(x)``. ``x`` may be a pair ``(a, b)`` whose sum is the input; with
        ``defer`` (and no ``prepad``) the block returns its pair ``(x, h)``
        (VAE_FUSED_ADD_NORM). Without either, the ordinary operations."""
        seq = rb.residual
        if isinstance(x, tuple):
            i = caches.next_index()
            r = self.norms[id(seq[0])].add_halo(
                x[0],
                x[1],
                caches.items[i],  # ty: ignore[invalid-argument-type]
            )
            if r is None:
                x = x[0] + x[1]
                y = None
            else:
                x, y = r
            h = (
                rb.shortcut(x)
                if isinstance(rb.shortcut, nn.Identity)
                else self._conv(rb.shortcut, x)
            )
            if y is None:
                y = self.norms[id(seq[0])](
                    x,
                    caches.items[i],
                    halo=True,  # ty: ignore[invalid-argument-type]
                )
        else:
            h = (
                rb.shortcut(x)
                if isinstance(rb.shortcut, nn.Identity)
                else self._conv(rb.shortcut, x)
            )
            i = caches.next_index()
            y = self.norms[id(seq[0])](
                x,
                caches.items[i],
                halo=True,  # ty: ignore[invalid-argument-type]
            )
        caches.items[i] = _keep(y[:, :, -CACHE_T:])
        x = self._conv_halo(seq[2], y)
        i = caches.next_index()
        y = self.norms[id(seq[3])](
            x,
            caches.items[i],
            halo=True,  # ty: ignore[invalid-argument-type]
        )
        caches.items[i] = _keep(y[:, :, -CACHE_T:])
        x = self._conv_halo(seq[6], y)
        if defer and not prepad and not torch.compiler.is_compiling():
            return x, h
        if not prepad or torch.compiler.is_compiling():
            return x + h
        # Write the sum into a buffer with a zero right and bottom ring. The
        # next downsampling convolution reads the buffer without F.pad.
        b, c, t, hh, ww = x.shape
        buf = torch.empty(
            (b, c, t, hh + 1, ww + 1),
            dtype=x.dtype,
            device=x.device,
            memory_format=CL3D,
        )
        buf[:, :, :, hh:, :].zero_()
        buf[:, :, :, :hh, ww:].zero_()
        out = buf[:, :, :, :hh, :ww]
        torch.add(x, h, out=out)
        self._prepadded[id(out)] = buf
        return out

    def _resample(self, rs: Resample, x: Tensor, caches: Caches) -> Tensor:
        # Dynamo cannot trace add(out=noncontiguous_view). In compiled execution
        # add and padding are expressed functionally and can fuse without the
        # eager path's identity-keyed prepadding side channel.
        buf = (
            None if torch.compiler.is_compiling() else self._prepadded.pop(id(x), None)
        )
        x = buf if buf is not None else F.pad(x, (0, 1, 0, 1))
        x = F.conv3d(
            x,
            self._h(self.resample_w[id(rs)]),
            self._h(rs.resample[1].bias),
            stride=(1, 2, 2),
        )
        if rs.mode == "downsample3d":
            i = caches.next_index()
            old = caches.items[i]
            if old is None:
                caches.items[i] = _keep(x[:, :, -1:])
            else:
                assert isinstance(old, Tensor)
                new = _keep(x[:, :, -1:])
                x = self._conv(rs.time_conv, torch.cat([old[:, :, -1:], x], 2))
                caches.items[i] = new
        return x

    def _avg_down(self, avg: AvgDown3D, x: Tensor) -> Tensor:
        ft, fs = avg.factor_t, avg.factor_s
        if avg.factor == 1 and avg.group_size == 1:
            return x
        supported = (
            fs == 2
            and avg.group_size == 4
            and (
                (ft == 1 and avg.out_dim == avg.in_dim)
                or (ft == 2 and avg.out_dim == 2 * avg.in_dim)
            )
        )
        if not supported or (ft == 2 and x.shape[2] % 2):
            return avg(x)
        x = (
            x
            if x.is_contiguous(memory_format=CL3D)
            else x.contiguous(memory_format=CL3D)
        )
        return K.avg_down(x, ft)

    def _down_block(self, blk: DownBlock, x: Tensor, caches: Caches) -> Tensor:
        shortcut = self._avg_down(blk.avg_shortcut, x)
        mods = list(blk.downsamples)
        for j, m in enumerate(mods):
            if isinstance(m, ResidualBlock):
                prepad = j + 1 < len(mods) and isinstance(mods[j + 1], Resample)
                defer = (
                    VAE_FUSED_ADD_NORM
                    and j + 1 < len(mods)
                    and isinstance(mods[j + 1], ResidualBlock)
                )
                x = self._residual(m, x, caches, prepad, defer)
            else:
                assert isinstance(m, Resample)
                x = self._resample(m, x, caches)
        return x + shortcut

    def _forward(self, buf: Tensor, caches: Caches) -> Tensor:
        enc = self.enc
        i = caches.next_index()
        caches.items[i] = _keep(buf[:, :, -CACHE_T:])
        x = self._conv_halo(enc.conv1, buf)
        for blk in enc.downsamples:
            x = self._down_block(blk, x, caches)
        for layer in enc.middle:
            if isinstance(layer, ResidualBlock):
                x = self._residual(layer, x, caches, prepad=False)
            else:
                x = layer(x)
        i = caches.next_index()
        y = self.norms[id(enc.head[0])](
            x,
            caches.items[i],
            halo=True,  # ty: ignore[invalid-argument-type]
        )
        caches.items[i] = _keep(y[:, :, -CACHE_T:])
        head = enc.head[2]
        return F.conv3d(
            y,
            self._h(self.head_w),
            self._h(self.head_b),
            head.stride,
            (0, head._pad[2], head._pad[0]),
        )

    @torch.no_grad()
    def encode(self, video: Tensor, chunk_frames: int = 4) -> Tensor:
        """Encode one video ``(B, 3, T, H, W)`` in ``[-1, 1]``. Returns bf16 latents."""
        self._prepadded: dict[int, Tensor] = {}
        t_eff = 1 + 4 * ((video.shape[2] - 1) // 4)
        cuts = {1, t_eff}
        cut = 1 + chunk_frames
        while cut < t_eff:
            cuts.add(cut)
            cut += chunk_frames
        boundaries = sorted(cuts)
        h2, w2 = video.shape[-2] // 2, video.shape[-1] // 2
        rows = video.shape[0]
        caches = Caches(self.n_caches)
        chunks: list[Tensor] = []
        prev = 0
        with torch.autocast("cuda", dtype=torch.float16):
            for cut in boundaries:
                buf = torch.empty(
                    (rows, 12, cut - prev + CACHE_T, h2, w2),
                    dtype=torch.float16 if VAE_HOIST_CASTS else video.dtype,
                    device=video.device,
                    memory_format=CL3D,
                )
                first = caches.items[0]
                if first is None:
                    buf[:, :, :CACHE_T].zero_()
                else:
                    buf[:, :, :CACHE_T].copy_(
                        first  # ty: ignore[invalid-argument-type]
                    )
                body = buf[:, :, CACHE_T:]
                t_hi = min(cut, video.shape[2])
                tlen = t_hi - prev
                # Patchify, chunk copy and layout change in four strided copies.
                for q in (0, 1):
                    for r in (0, 1):
                        dst = body[:, r * 2 + q :: 4]
                        dst[:, :, :tlen] = video[:, :, prev:t_hi, q::2, r::2]
                        if tlen < cut - prev:
                            dst[:, :, tlen:] = video[:, :, t_hi - 1 : t_hi, q::2, r::2]
                caches.index = 0
                chunks.append(self._forward(buf, caches))
                prev = cut
        z = torch.cat(chunks, dim=2)
        return z[:, :, : 1 + (video.shape[2] - 1) // 4].to(torch.bfloat16)


class StreamEncode:
    """``Autoencoder.encode`` of a clip whose frames arrive one at a time.

    ``FusedEncoder.encode`` runs the causal encoder chunk by chunk: the first
    frame alone, then 4-frame chunks, each reading only the caches of the
    chunks before it. This class runs the same chunks with the same buffers,
    the same caches and the same autocast in the same order, each chunk as
    soon as its frames are in. So :meth:`latent` has the bits of one
    ``encode`` of the whole clip. It shares the fused encoder of the
    autoencoder (:meth:`Autoencoder.fused_encoder`): the encoder keeps no
    state between two chunks. The design is the ``StreamEncoder`` of a
    WebRTC receiver.
    """

    def __init__(self, fused: FusedEncoder, n_frames: int) -> None:
        if n_frames < 1:
            raise ValueError("n_frames must be positive")
        self.fused = fused
        self.n = int(n_frames)
        t_eff = 1 + 4 * ((self.n - 1) // 4)
        cuts = {1, t_eff}
        cut = 5
        while cut < t_eff:
            cuts.add(cut)
            cut += 4
        self.boundaries = sorted(cuts)
        self.caches = Caches(fused.n_caches)
        self.added = 0
        # The frames of the chunk that is not complete yet, in [-1, 1].
        self._pending: list[Tensor] = []
        self.chunks: list[Tensor] = []
        self._prev = 0
        self._next = 0

    @property
    def done(self) -> bool:
        return self._next == len(self.boundaries)

    def add(self, frame: Tensor) -> None:
        """Append one ``(1, 3, 1, H, W)`` frame in ``[0, 255]``; run each chunk it completes."""
        if self.added >= self.n:
            raise ValueError("the clip is complete")
        self.added += 1
        if self.done:
            return  # past the last chunk, as FusedEncoder.encode skips it
        self._pending.append(Autoencoder._to_unit(frame))
        while not self.done:
            cut = self.boundaries[self._next]
            t_hi = min(cut, self.n)
            if self.added < t_hi:
                return
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16),
                torch.autocast("cuda", dtype=torch.float16),
            ):
                self.chunks.append(self._chunk(cut, t_hi))
            self._prev = cut
            self._next += 1

    def _chunk(self, cut: int, t_hi: int) -> Tensor:
        # The loop body of FusedEncoder.encode on the frames [prev, t_hi).
        prev = self._prev
        seg = torch.cat(self._pending[: t_hi - prev], dim=2)
        del self._pending[: t_hi - prev]
        h2, w2 = seg.shape[-2] // 2, seg.shape[-1] // 2
        buf = torch.empty(
            (1, 12, cut - prev + CACHE_T, h2, w2),
            dtype=torch.float16 if VAE_HOIST_CASTS else seg.dtype,
            device=seg.device,
            memory_format=CL3D,
        )
        first = self.caches.items[0]
        if first is None:
            buf[:, :, :CACHE_T].zero_()
        else:
            buf[:, :, :CACHE_T].copy_(first)  # ty: ignore[invalid-argument-type]
        body = buf[:, :, CACHE_T:]
        tlen = t_hi - prev
        for q in (0, 1):
            for r in (0, 1):
                dst = body[:, r * 2 + q :: 4]
                dst[:, :, :tlen] = seg[:, :, :, q::2, r::2]
                if tlen < cut - prev:
                    dst[:, :, tlen:] = seg[:, :, -1:, q::2, r::2]
        self.caches.index = 0
        # The prepadded buffers are keyed by tensor id and live within one
        # _forward. FusedEncoder.encode starts each encode with an empty map.
        self.fused._prepadded = {}
        return self.fused._forward(buf, self.caches)

    def latent(self) -> Tensor:
        """Return the policy latent of the whole clip, as ``Autoencoder.encode`` does."""
        if not self.done:
            raise RuntimeError(f"{self.added} of {self.n} frames encoded")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = torch.cat(self.chunks, dim=2)
            z = z[:, :, : 1 + (self.n - 1) // 4].to(torch.bfloat16)
            return to_policy_latent(z)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class Autoencoder(nn.Module):
    """The Wan2.2 video autoencoder with the policy pre- and post-processing."""

    def __init__(
        self,
        norm_block_c: dict[int, int] | None = None,
        *,
        decoder_impl: str = "reference",
    ) -> None:
        super().__init__()
        if decoder_impl not in ("reference", "fused"):
            raise ValueError(f"Unknown VAE decoder implementation: {decoder_impl}")
        self.norm_block_c = dict(norm_block_c or K.DEFAULT_NORM_BLOCK_C)
        self.encoder = Encoder()
        self.conv1 = CausalConv3d(2 * Z_DIM, 2 * Z_DIM, 1)
        self.conv2 = CausalConv3d(Z_DIM, Z_DIM, 1)
        self.decoder = Decoder()
        self.register_buffer("latent_mean", torch.tensor(LATENT_MEAN), persistent=False)
        self.register_buffer(
            "latent_inv_std", 1.0 / torch.tensor(LATENT_STD), persistent=False
        )
        self._fused: FusedEncoder | None = None
        self._decoder_controller = None
        if decoder_impl == "fused":
            self.enable_fused_decoder()

    def _apply(self, fn, recurse: bool = True) -> nn.Module:
        mean, inv_std = self.latent_mean, self.latent_inv_std
        super()._apply(fn, recurse)
        # The normalisation constants stay float32. Move them only.
        self.latent_mean = mean.to(device=self.latent_mean.device)
        self.latent_inv_std = inv_std.to(device=self.latent_inv_std.device)
        self._fused = None
        if self._decoder_controller is not None:
            self._decoder_controller.reset()
        return self

    def enable_fused_decoder(
        self,
        *,
        stream_output: bool = True,
        compile_decoder: bool = False,
        fuse_norm_silu: bool = True,
        norm_output_dtype: str = "float32",
        fuse_cache_input: bool = True,
    ):
        """Enable FP16 convolutions and fused video-decoder kernels.

        Decoder-step compilation is opt-in through compile_decoder=True.
        Causal convolution inputs and history use one Triton pass by default.
        Set fuse_cache_input=False to disable it.
        norm_output_dtype="bfloat16" stores normalized activations and history
        in BF16 while retaining FP32 arithmetic. The default is "float32".
        Configure before warmup. Original checkpoint parameters stay unchanged.
        """
        from praxis.vae_decoder import install

        if self._decoder_controller is not None:
            self.disable_decoder_optimizations()
        return install(
            self,
            stream_output=stream_output,
            compile_decoder=compile_decoder,
            fuse_norm_silu=fuse_norm_silu,
            norm_output_dtype=norm_output_dtype,
            fuse_cache_input=fuse_cache_input,
        )

    def disable_decoder_optimizations(self) -> None:
        """Restore the original reference decoder, including unfused norms."""
        if self._decoder_controller is not None:
            self.decoder.forward = self._decoder_controller.original_forward
            self._decoder_controller.restore_norm_silu()
        for module in self.modules():
            if isinstance(module, CausalConv3d):
                module._decoder_adapter = None
        self._decoder_controller = None

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str | torch.device = "cuda",
        norm_block_c: dict[int, int] | None = None,
        *,
        decoder_impl: str = "fused",
    ) -> Autoencoder:
        """Load ``Wan2.2_VAE.pth`` (float32) with the rounding of the checkpoint.

        The weights go float32 to float16 to bfloat16. ``norm_block_c`` maps a
        channel count to the ``BLOCK_C`` of the fused norm kernel. None uses
        ``DEFAULT_NORM_BLOCK_C``.
        """
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        with torch.device("meta"):
            model = cls(norm_block_c, decoder_impl="reference")
        state = {k: v.to(torch.float16).to(torch.bfloat16) for k, v in state.items()}
        model.load_state_dict(state, strict=True, assign=True)
        model.latent_mean = torch.tensor(LATENT_MEAN)
        model.latent_inv_std = 1.0 / torch.tensor(LATENT_STD)
        model = model.to(device).eval().requires_grad_(False)
        if decoder_impl not in ("reference", "fused"):
            raise ValueError(f"Unknown VAE decoder implementation: {decoder_impl}")
        # The fused decoder runs CUDA kernels; other devices keep the reference.
        if decoder_impl == "fused" and torch.device(device).type == "cuda":
            model.enable_fused_decoder()
        return model

    @property
    def scale(self) -> tuple[Tensor, Tensor]:
        return self.latent_mean, self.latent_inv_std

    @staticmethod
    def _to_unit(frames: Tensor) -> Tensor:
        """Map ``[0, 255]`` frames to ``[-1, 1]`` float32."""
        return (frames.float() / 255) * 2 - 1

    def fused_encoder(self) -> FusedEncoder:
        """Return the fused encoder of :meth:`encode`, built on the first use."""
        if self._fused is None:
            self._fused = FusedEncoder(
                self.encoder, self.conv1, self.scale, self.norm_block_c
            )
        return self._fused

    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """Encode ``(1, 3, T, H, W)`` frames in ``[0, 255]`` to the policy latent.

        This is the fused path. It needs CUDA and Triton.
        """
        fused = self.fused_encoder()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = fused.encode(self._to_unit(frames))
            return to_policy_latent(z)

    @torch.no_grad()
    def encode_eager(self, frames: Tensor) -> Tensor:
        """Encode with plain PyTorch ops. Not bitwise equal to ``encode``."""
        x = patchify(self._to_unit(frames))
        caches = Caches(count_causal_convs(self.encoder))
        n = 1 + (x.shape[2] - 1) // 4
        dev_type = x.device.type
        outs = []
        with (
            torch.autocast(dev_type, dtype=torch.bfloat16),
            torch.autocast(dev_type, dtype=torch.float16),
        ):
            mean, inv_std = self.scale
            for i in range(n):
                caches.index = 0
                sl = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
                mu = self.conv1(self.encoder(sl, caches)).chunk(2, dim=1)[0]
                mu = (mu - mean.view(1, Z_DIM, 1, 1, 1)) * inv_std.view(
                    1, Z_DIM, 1, 1, 1
                )
                outs.append(mu)
            return to_policy_latent(torch.cat(outs, dim=2).to(torch.bfloat16))

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """Decode a policy latent ``(1, 192, T, h, w)`` to uint8 frames ``(T', 3, H, W)``.

        The frames are :meth:`decode_unit` mapped to ``[0, 1]`` and scaled by
        255, with a truncation to uint8. A streaming fused decoder returns its
        own frames.
        """
        if (
            self._decoder_controller is not None
            and self._decoder_controller.stream_output
        ):
            return self._decoder_controller.decode(latent)
        video = self.decode_unit(latent)
        v = video[0].permute(1, 0, 2, 3).float().mul(0.5).add(0.5).clamp(0, 1)
        return (v * 255).to(torch.uint8)

    @torch.no_grad()
    def decode_unit(self, latent: Tensor) -> Tensor:
        """Decode a policy latent ``(1, 192, T, h, w)`` to ``(1, 3, T', H, W)`` in [-1, 1].

        The values are float32. The autoregressive eval rounds them to uint8
        with ``round((x + 1) * 127.5)``, the rule of its reference rollouts.
        """
        z = from_policy_latent(latent.to(torch.bfloat16))
        dev_type = z.device.type
        with (
            torch.autocast(dev_type, dtype=torch.bfloat16),
            torch.autocast(dev_type, dtype=torch.float16),
        ):
            mean, inv_std = self.scale
            z = z / inv_std.view(1, Z_DIM, 1, 1, 1) + mean.view(1, Z_DIM, 1, 1, 1)
            x = self.conv2(z)
            caches = Caches(count_causal_convs(self.decoder))
            outs = []
            for i in range(z.shape[2]):
                caches.index = 0
                outs.append(
                    self.decoder(x[:, :, i : i + 1], caches, first_chunk=i == 0)
                )
            out = unpatchify(torch.cat(outs, dim=2))
            return out.float().clamp_(-1, 1)
