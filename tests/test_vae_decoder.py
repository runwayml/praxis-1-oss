"""Check decoder padding, normalization and streamed pixel assembly."""

import pytest
import torch
import torch.nn.functional as F

from praxis.vae import CausalConv3d, unpatchify
from praxis.vae_decoder import SpatialConv


@pytest.mark.parametrize("cached_frames", [0, 1, 2])
def test_implicit_spatial_padding_preserves_causal_convolution(cached_frames):
    torch.manual_seed(3)
    conv = CausalConv3d(4, 6, 3, padding=1)
    x = torch.randn(1, 4, 2, 5, 7)
    cache = torch.randn(1, 4, cached_frames, 5, 7) if cached_frames else None
    expected = conv(x, cache)
    actual = SpatialConv()(conv, x, cache)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder kernels")
@pytest.mark.parametrize("channels", [256, 512, 1024])
@pytest.mark.parametrize("bf16_output", [False, True])
def test_decoder_norm_matches_fp32_reference(channels, bf16_output):
    from praxis.kernels.vae_decoder_norm_triton import decoder_norm_silu

    torch.manual_seed(9)
    x = torch.randn(2, channels, 2, 3, 5, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    expected = F.silu(
        F.normalize(x.float(), dim=1)
        * channels**0.5
        * gamma.float().view(1, -1, 1, 1, 1)
    )
    if bf16_output:
        expected = expected.bfloat16()
    actual = decoder_norm_silu(x, gamma, channels**0.5, bf16_output)
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.016 if bf16_output else 1e-5,
        rtol=0.016 if bf16_output else 1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder kernels")
@pytest.mark.parametrize("channels_last", [False, True])
def test_streamed_rgb_matches_reference_and_preserves_other_frames(channels_last):
    from praxis.kernels.vae_decoder_triton import store_rgb

    torch.manual_seed(11)
    x = torch.randn(1, 12, 2, 3, 5, device="cuda", dtype=torch.float16)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    video = unpatchify(x).float().clamp(-1, 1)[0].permute(1, 0, 2, 3)
    expected = ((video * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    output = torch.full((4, 3, 6, 10), 17, device="cuda", dtype=torch.uint8)
    store_rgb(x, output, 1)
    assert torch.equal(output[1:3], expected)
    assert torch.all(output[0] == 17) and torch.all(output[3] == 17)


def test_cache_assembly_controller_defaults_on_is_reversible_and_decoder_only():
    from praxis.vae import Autoencoder
    from praxis.vae_decoder import CacheInputConv

    with torch.device("meta"):
        vae = Autoencoder(decoder_impl="reference")
    keys = set(vae.state_dict())
    vae.enable_fused_decoder(compile_decoder=True, norm_output_dtype="bfloat16")
    assert vae._decoder_controller.fuse_cache_input
    assert not any(
        isinstance(m._decoder_adapter, CacheInputConv)
        for m in vae.encoder.modules()
        if isinstance(m, CausalConv3d)
    )
    assert any(
        isinstance(m._decoder_adapter, CacheInputConv)
        for m in vae.decoder.modules()
        if isinstance(m, CausalConv3d)
    )
    vae.disable_decoder_optimizations()
    assert all(
        m._decoder_adapter is None for m in vae.modules() if isinstance(m, CausalConv3d)
    )
    assert set(vae.state_dict()) == keys
    vae.enable_fused_decoder()
    assert vae._decoder_controller.fuse_cache_input
    assert not vae._decoder_controller.compile_decoder
    assert vae._decoder_controller.norm_output_dtype == "float32"
    vae.enable_fused_decoder(fuse_cache_input=False)
    assert not vae._decoder_controller.fuse_cache_input


def test_cache_assembly_falls_back_for_cpu():
    from praxis.vae_decoder import CacheInputConv

    adapter = CacheInputConv(2, torch.float16)
    assert not adapter.supports(torch.randn(1, 4, 1, 3, 5), None)


@pytest.mark.parametrize("enabled", [False, True])
def test_decoder_cache_fusion_preserves_override(enabled):
    from praxis.vae import Autoencoder

    with torch.device("meta"):
        vae = Autoencoder(decoder_impl="reference")
    controller = vae.enable_fused_decoder(fuse_cache_input=enabled)
    assert controller.fuse_cache_input == enabled
    assert controller.report()["precision"] == "float16"
    vae.disable_decoder_optimizations()


def test_fp8_decoder_mode_is_rejected():
    from praxis.vae import Autoencoder

    with pytest.raises(ValueError, match="Unknown VAE decoder implementation: fp8"):
        Autoencoder(decoder_impl="fp8")
