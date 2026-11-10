"""``StreamEncode`` against one ``FusedEncoder.encode`` of the whole clip (CPU).

The fused encoder runs two Triton kernels (the norm and the pooling). Here
they are replaced by torch functions with the same contract, so the fused
encoder runs on the CPU with a small random encoder. The stream must give
the bits of one encode for every clip length: ``1 + 4k`` frames and the
lengths in between, whose last chunk is padded or dropped.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from praxis import vae
from praxis.kernels import vae_encoder_triton as kernels


def _rmsnorm_silu(
    x: Tensor, out: Tensor, gamma: Tensor, sqrt_c: float, block_c: int
) -> None:
    y = F.normalize(x.float(), dim=1) * sqrt_c * gamma.view(1, -1, 1, 1, 1)
    out.copy_(F.silu(y).to(out.dtype))


def _avg_down(x: Tensor, factor_t: int) -> Tensor:
    b, c, t, h, w = x.shape
    y = x.float().reshape(b, c, t // factor_t, factor_t, h // 2, 2, w // 2, 2)
    y = y.mean(dim=(5, 7)).permute(0, 1, 3, 2, 4, 5)
    y = y.reshape(b, c * factor_t, t // factor_t, h // 2, w // 2)
    return y.to(x.dtype).contiguous(memory_format=torch.channels_last_3d)


@pytest.fixture
def fused(monkeypatch: pytest.MonkeyPatch) -> vae.FusedEncoder:
    monkeypatch.setattr(kernels, "rmsnorm_silu", _rmsnorm_silu)
    monkeypatch.setattr(kernels, "avg_down", _avg_down)
    torch.manual_seed(0)
    enc = vae.Encoder(dim=8).eval().requires_grad_(False)
    conv_mu = vae.CausalConv3d(2 * vae.Z_DIM, 2 * vae.Z_DIM, 1).requires_grad_(False)
    for module in (enc, conv_mu):
        for p in module.parameters():
            p.data.normal_(0, 0.1)
    scale = (torch.randn(vae.Z_DIM), torch.rand(vae.Z_DIM) + 0.5)
    return vae.FusedEncoder(enc, conv_mu, scale, {8: 64, 16: 64, 32: 64})


def _encode(fused: vae.FusedEncoder, frames: Tensor) -> Tensor:
    return vae.to_policy_latent(fused.encode(vae.Autoencoder._to_unit(frames)))


def _stream(fused: vae.FusedEncoder, frames: Tensor, order: list[int]) -> Tensor:
    stream = vae.StreamEncode(fused, frames.shape[2])
    for i in order:
        stream.add(frames[:, :, i : i + 1])
    return stream.latent()


@pytest.mark.parametrize("n", [1, 2, 4, 5, 7, 9, 13, 21, 22, 45])
def test_the_stream_gives_the_bits_of_one_encode(
    fused: vae.FusedEncoder, n: int
) -> None:
    frames = torch.randint(0, 256, (1, 3, n, 64, 64)).float()
    want = _encode(fused, frames)
    got = _stream(fused, frames, list(range(n)))
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.equal(got, want)


def test_the_stream_reads_the_frames_in_order(fused: vae.FusedEncoder) -> None:
    """Two frames swapped inside a chunk change the bits: the check above
    is not blind to the frame order."""
    frames = torch.randint(0, 256, (1, 3, 21, 64, 64)).float()
    order = list(range(21))
    order[6], order[7] = order[7], order[6]
    assert not torch.equal(_stream(fused, frames, order), _encode(fused, frames))


def test_the_stream_refuses_a_latent_before_the_last_frame(
    fused: vae.FusedEncoder,
) -> None:
    frames = torch.randint(0, 256, (1, 3, 9, 64, 64)).float()
    stream = vae.StreamEncode(fused, 9)
    for i in range(8):
        stream.add(frames[:, :, i : i + 1])
    with pytest.raises(RuntimeError, match="8 of 9"):
        stream.latent()
    stream.add(frames[:, :, 8:9])
    with pytest.raises(ValueError, match="complete"):
        stream.add(frames[:, :, 8:9])
