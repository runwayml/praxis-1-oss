"""Fused FP16 video-decoder execution.

Convolutions use implicit spatial padding and explicit causal time padding.
Normalization and SiLU use a fused Triton kernel with float32 arithmetic.
Decoded chunks write directly into the final RGB buffer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SpatialConv:
    """cuDNN convolution with implicit spatial and explicit causal padding."""

    def __call__(self, conv, x, cache):
        pad = list(conv._pad)
        if cache is not None and pad[4] > 0:
            x = torch.cat([cache.to(x.device), x], dim=2)
            pad[4] -= cache.shape[2]
        # cuDNN handles H/W padding without writing
        # a large spatial halo. Retain explicit causal time padding/cropping.
        if pad[4]:
            x = F.pad(x, (0, 0, 0, 0, pad[4], 0))
        return F.conv3d(
            x,
            conv.weight,
            conv.bias,
            conv.stride,
            (0, pad[2], pad[0]),
            conv.dilation,
            conv.groups,
        )


class CacheInputConv(SpatialConv):
    """Assemble temporal padding and next history once, in convolution layout."""

    fuse_cache_input = True

    def __init__(self, temporal_padding, compute_dtype):
        self.temporal_padding = temporal_padding
        self.compute_dtype = compute_dtype

    def supports(self, x, cache):
        return (
            x.is_cuda
            and x.ndim == 5
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and all(size > 0 for size in x.shape)
            and not (torch.is_grad_enabled() and x.requires_grad)
            and (
                cache is None
                or (
                    cache.ndim == 5
                    and cache.device == x.device
                    and cache.dtype == x.dtype
                    and cache.shape[:2] == x.shape[:2]
                    and cache.shape[3:] == x.shape[3:]
                    and cache.shape[2] <= self.temporal_padding
                    and not (torch.is_grad_enabled() and cache.requires_grad)
                )
            )
        )

    def with_cache(self, conv, x, cache):
        from praxis.kernels.vae_decoder_cache_triton import assemble_causal_input

        packed, history = assemble_causal_input(
            x, cache, self.temporal_padding, self.compute_dtype
        )
        out = F.conv3d(
            packed,
            conv.weight,
            conv.bias,
            conv.stride,
            (0, conv._pad[2], conv._pad[0]),
            conv.dilation,
            conv.groups,
        )
        return out, history


class DecoderController:
    """Install reversible decoder adapters and optional normalization fusion.

    Configure before warmup. The controller owns prepared
    execution state; it never replaces checkpoint parameter tensors.
    """

    def __init__(
        self,
        vae,
        stream_output=True,
        compile_decoder=False,
        fuse_norm_silu=True,
        norm_output_dtype="float32",
        fuse_cache_input=True,
    ):
        from praxis.vae import CausalConv3d, ResidualBlock

        if norm_output_dtype not in ("float32", "bfloat16"):
            raise ValueError("norm_output_dtype must be float32 or bfloat16")
        if not fuse_norm_silu and norm_output_dtype != "float32":
            raise ValueError("BF16 normalization output requires fuse_norm_silu=True")
        self.compute_dtype = torch.float16
        self.vae = vae
        self.stream_output = stream_output
        self.compile_decoder = compile_decoder
        self.compiled = False
        self.original_forward = vae.decoder.forward
        self.fuse_norm_silu = fuse_norm_silu
        self.norm_output_dtype = norm_output_dtype
        self.fuse_cache_input = bool(fuse_cache_input)
        self.norm_silu_originals = []
        if fuse_norm_silu:
            from praxis.kernels.vae_decoder_norm_triton import decoder_norm_silu

            pairs = []
            for module in vae.decoder.modules():
                if isinstance(module, ResidualBlock):
                    pairs.extend(
                        [
                            (module.residual[0], module.residual[1]),
                            (module.residual[3], module.residual[4]),
                        ]
                    )
            pairs.append((vae.decoder.head[0], vae.decoder.head[1]))

            def fused_forward(norm):
                def call(value):
                    return decoder_norm_silu(
                        value, norm.gamma, norm.scale, norm_output_dtype == "bfloat16"
                    )

                return call

            def identity(value):
                return value

            for norm, activation in pairs:
                self.norm_silu_originals.extend(
                    [(norm, norm.forward), (activation, activation.forward)]
                )
                norm.forward = fused_forward(norm)
                activation.forward = identity
        self.adapters = {}
        for name, conv in vae.named_modules():
            if not (name.startswith("decoder.") or name == "conv2") or not isinstance(
                conv, CausalConv3d
            ):
                continue
            adapter = (
                CacheInputConv(conv._pad[4], self.compute_dtype)
                if self.fuse_cache_input and name.startswith("decoder.")
                else SpatialConv()
            )
            conv._decoder_adapter = adapter
            self.adapters[name] = adapter

    def reset(self):
        self.vae.decoder.forward = self.original_forward
        self.compiled = False

    def restore_norm_silu(self):
        for module, forward in self.norm_silu_originals:
            module.forward = forward
        self.norm_silu_originals.clear()

    def decode(self, latent):
        if self.compile_decoder and not self.compiled:
            # Warm the decoder before compiling. Return the compiled result
            # on first use too, so graph validation uses the same rounding.
            self._decode(latent)
            self.vae.decoder.forward = torch.compile(
                self.original_forward,
                dynamic=False,
                fullgraph=False,
                options={"triton.cudagraphs": False, "emulate_precision_casts": True},
            )
            self.compiled = True
        return self._decode(latent)

    def _decode(self, latent):
        from praxis.kernels.vae_decoder_triton import store_rgb
        from praxis.vae import Z_DIM, Caches, count_causal_convs, from_policy_latent

        vae = self.vae
        z = from_policy_latent(latent.to(torch.bfloat16))
        with torch.autocast("cuda", dtype=self.compute_dtype):
            mean, inv_std = vae.scale
            z = z / inv_std.view(1, Z_DIM, 1, 1, 1) + mean.view(1, Z_DIM, 1, 1, 1)
            x = vae.conv2(z)
            caches = Caches(count_causal_convs(vae.decoder))
            # Preallocate the final RGB video and commit
            # streamed chunks immediately; no temporal all-frame tensor list.
            frames = 1 + 4 * (z.shape[2] - 1)
            rgb = torch.empty(
                (frames, 3, z.shape[-2] * 16, z.shape[-1] * 16),
                device=z.device,
                dtype=torch.uint8,
            )
            offset = 0
            for i in range(z.shape[2]):
                caches.index = 0
                # Canonicalize the sliced latent's strides. Otherwise the
                # parent context length creates redundant compiled variants and
                # can exhaust Dynamo's cache, silently falling back to eager.
                chunk = vae.decoder(
                    x[:, :, i : i + 1].contiguous(), caches, first_chunk=i == 0
                )
                store_rgb(chunk, rgb, offset)
                offset += chunk.shape[2]
            assert offset == frames
            return rgb

    def report(self):
        return {
            "precision": "float16",
            "compute_dtype": str(self.compute_dtype),
            "encoder": "unchanged",
            "spatial_padding": "cuDNN implicit H/W",
            "stream_output": self.stream_output,
            "decoder_compiled": self.compiled,
            "custom_norm_silu": self.fuse_norm_silu,
            "norm_output_dtype": self.norm_output_dtype,
            "norm_accumulation_dtype": "float32",
            "custom_norm_silu_pairs": len(self.norm_silu_originals) // 2,
            "fused_cache_input": self.fuse_cache_input,
        }


def install(vae, **kwargs):
    controller = DecoderController(vae, **kwargs)
    vae._decoder_controller = controller
    return controller
