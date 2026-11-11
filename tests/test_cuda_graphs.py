"""Graph boundaries, device eligibility and independent replay outputs."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import praxis.cuda_graphs as graphs


@pytest.mark.parametrize(
    ("device", "options", "capability", "expected", "moe_expected"),
    [
        ("cuda", {}, (10, 0), True, False),
        ("cuda", {"action_cuda_graphs": False}, (10, 0), False, False),
        ("cpu", {}, None, False, False),
        ("cuda", {"routed_precision": "nvfp4"}, (10, 0), True, True),
        ("cuda", {"routed_precision": "nvfp4"}, (9, 0), True, False),
        ("cuda", {"routed_precision": "nvfp4"}, (11, 0), True, False),
        ("cpu", {"routed_precision": "nvfp4"}, None, False, False),
        (
            "cuda",
            {"routed_precision": "nvfp4", "nvfp4_moe_cuda_graphs": False},
            (10, 0),
            True,
            False,
        ),
    ],
)
def test_pipeline_graph_defaults(
    monkeypatch, device, options, capability, expected, moe_expected
):
    from tiny_model import tiny_config

    import praxis.pipeline as module

    config = tiny_config()
    config["normalization"] = {
        "action_q01": [0.0] * 10,
        "action_q99": [1.0] * 10,
        "proprio_mean": [0.0] * 10,
        "proprio_std": [1.0] * 10,
    }
    monkeypatch.setattr(
        module.PraxisPolicy,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        module.Autoencoder,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(enable_fused_decoder=lambda: None),
    )
    monkeypatch.setattr(module, "TextEncoder", lambda *args, **kwargs: None)
    enabled = []
    moe_enabled = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(
        module.PraxisPipeline,
        "enable_nvfp4_moe_cuda_graphs",
        lambda self: moe_enabled.append(self),
    )
    monkeypatch.setattr(
        module.PraxisPipeline,
        "enable_action_cuda_graphs",
        lambda self: enabled.append(self),
    )
    pipe = module.PraxisPipeline("checkpoint", "vae", "text", device=device, **options)
    assert enabled == ([pipe] if expected else [])
    assert moe_enabled == ([pipe] if moe_expected else [])


def test_nvfp4_moe_graph_boundaries_and_guards(monkeypatch):
    from praxis.layers import MoEFeedForward

    class Packed(nn.Module):
        format = "nvfp4"

    class Pool:
        def __init__(self, device, **kwargs):
            self.options = kwargs
            self.targets = []

        def wrap(self, owner, attr, name):
            self.targets.append((owner, attr, name))

    monkeypatch.setattr(graphs, "CudaGraphPool", Pool)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    moe = MoEFeedForward(dim=8, hidden=4, num_experts=4, top_k=2)
    moe.routed = Packed()
    backbone = nn.Sequential(moe, nn.Linear(8, 8)).eval().requires_grad_(False)
    pipe = SimpleNamespace(
        device=torch.device("cuda"),
        sampler=SimpleNamespace(layout="window_joint"),
        variant=SimpleNamespace(backbone_layout="chunk_causal"),
        policy=SimpleNamespace(
            backbone=backbone, routed_quantization={"format": "nvfp4"}
        ),
        action_cuda_graphs=object(),
    )
    pipe.policy.routed_quantization = {}
    with pytest.raises(ValueError, match="window_joint NVFP4"):
        graphs.enable_nvfp4_moe_cuda_graphs(pipe)
    pipe.policy.routed_quantization = {"format": "nvfp4"}
    moe.train()
    with pytest.raises(ValueError, match="frozen eval"):
        graphs.enable_nvfp4_moe_cuda_graphs(pipe)
    moe.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen eval"):
        graphs.enable_nvfp4_moe_cuda_graphs(pipe)
    moe.requires_grad_(False)
    pool = graphs.enable_nvfp4_moe_cuda_graphs(pipe)
    assert pool.targets == [(moe, "forward", "0")]
    assert not pool.options.get("allow_streams", False)
    assert pipe.nvfp4_moe_cuda_graphs is pool
    with pytest.raises(ValueError, match="already enabled"):
        graphs.enable_nvfp4_moe_cuda_graphs(pipe)


def pipeline(device="cpu"):
    language = nn.Module()
    language.embed_tokens = nn.Embedding(16, 8)
    language.rotary_emb = nn.Identity()
    language.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
    language.norm = nn.LayerNorm(8)
    text = nn.Module()
    text.language_model = language
    text.to(device).requires_grad_(False)
    action = nn.Linear(8, 8).to(device).requires_grad_(False)
    return SimpleNamespace(
        device=torch.device(device),
        embodiment="bimanual",
        sampler=SimpleNamespace(layout="chunk", guided=False),
        variant=SimpleNamespace(backbone_layout="serve"),
        text=SimpleNamespace(model=text),
        # Uncaptured modules need not be frozen for text/action graph use.
        vae=SimpleNamespace(encode=nn.Identity()),
        policy=SimpleNamespace(
            action_expert=action,
            backbone=SimpleNamespace(encoder=nn.Linear(8, 8), decoder=nn.Linear(8, 8)),
        ),
    )


def test_only_text_and_action_expert_are_registered(monkeypatch):
    class Pool:
        def __init__(self, *args, **kwargs):
            self.names = []

        def wrap(self, owner, attr, name):
            self.names.append(name)

    monkeypatch.setattr(graphs, "CudaGraphPool", Pool)
    pipe = pipeline()
    owners = [pipe.vae, pipe.policy.backbone.encoder, pipe.policy.backbone.decoder]
    attrs = ["encode", "forward", "forward"]
    before = [getattr(owner, attr) for owner, attr in zip(owners, attrs, strict=True)]
    pool = graphs.enable_cuda_graphs(pipe)
    assert pool.names == [
        "text_embedding",
        "text_rotary",
        "text_layer_0",
        "text_layer_1",
        "text_norm",
        "action_expert",
    ]
    assert before == [
        getattr(owner, attr) for owner, attr in zip(owners, attrs, strict=True)
    ]
    for owner in owners:
        assert not hasattr(owner, "_cuda_graph_positions")


def test_window_joint_graphs_capture_only_action_expert(monkeypatch):
    class Pool:
        def __init__(self, device, **kwargs):
            self.options = kwargs
            self.targets = []

        def wrap(self, owner, attr, name):
            self.targets.append((owner, attr, name))

    monkeypatch.setattr(graphs, "CudaGraphPool", Pool)
    pipe = pipeline()
    pipe.variant.backbone_layout = "chunk_causal"
    pipe.sampler = SimpleNamespace(layout="window_joint", guided=True)
    pipe.policy.action_expert.eval()
    pool = graphs.enable_action_cuda_graphs(pipe)
    assert pool.targets == [
        (pipe.policy.action_expert, "forward_trunk_rope", "action_expert")
    ]
    assert pool.options["allow_streams"] and pool.options["validate"]
    assert pipe.action_cuda_graphs is pool
    with pytest.raises(ValueError, match="already enabled"):
        graphs.enable_action_cuda_graphs(pipe)
    with pytest.raises(ValueError, match="already enabled"):
        graphs.enable_cuda_graphs(pipe)


def test_window_joint_graphs_reject_training_and_unsupported_layouts():
    pipe = pipeline()
    with pytest.raises(ValueError, match="window_joint"):
        graphs.enable_action_cuda_graphs(pipe)
    pipe.variant.backbone_layout = "chunk_causal"
    pipe.sampler.layout = "window_joint"
    with pytest.raises(ValueError, match="frozen eval"):
        graphs.enable_action_cuda_graphs(pipe)
    pipe.policy.action_expert.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen eval"):
        graphs.enable_action_cuda_graphs(pipe)
    pipe.policy.action_expert.requires_grad_(False)
    with pytest.raises(ValueError, match="CUDA device"):
        graphs.enable_action_cuda_graphs(pipe)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph smoke test")
def test_text_and_action_graphs_replay_changed_inputs():
    torch.manual_seed(17)
    pipe = pipeline("cuda")
    language = pipe.text.model.language_model
    modules = [
        language.embed_tokens,
        language.rotary_emb,
        *language.layers,
        language.norm,
        pipe.policy.action_expert,
    ]
    original = [module.forward for module in modules]
    pool = graphs.enable_cuda_graphs(pipe, warmup=1)
    with torch.inference_mode():
        for _ in range(2):
            ids = torch.randint(0, 16, (1, 3), device="cuda")
            expected = ids
            for fn in original:
                expected = fn(expected)
            actual = ids
            for module in modules:
                actual = module(actual)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            pool.freeze()
    assert all(
        s["captures"] == 1 and s["validated"] == 1 and s["replays"] == 2
        for s in pool.stats().values()
    )
