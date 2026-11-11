"""CPU regressions for cached query rows and compact attention masks."""

import pytest
import torch

from praxis import layers
from praxis.backbone import _assemble_tf_block_mask, _slice_q_blocks
from praxis.layers import Attention, KVSplice, MoEFeedForward, _pad_context_rows


@pytest.mark.parametrize("full_blocks", [False, True])
def test_slice_compact_mask_keeps_absolute_key_blocks(full_blocks):
    adjacency = torch.zeros(4, 4, dtype=torch.bool)
    adjacency[:, 3] = True

    def visible(b, h, q, k):
        return k // 4 == 3

    mask = _assemble_tf_block_mask(
        adjacency,
        16,
        4,
        torch.device("cpu"),
        visible,
        adjacency if full_blocks else None,
    )
    # A one-entry list refers to key block 3. Its width is not the KV length.
    sliced = _slice_q_blocks(mask, 4, 12)
    assert sliced.seq_lengths == (8, 16)
    assert sliced.q_num_blocks is None
    q = torch.arange(8)[:, None]
    k = torch.arange(16)[None, :]
    assert torch.equal(sliced.mask_mod(0, 0, q, k), visible(0, 0, q + 4, k))
    counts = sliced.full_kv_num_blocks if full_blocks else sliced.kv_num_blocks
    indices = sliced.full_kv_indices if full_blocks else sliced.kv_indices
    assert torch.equal(counts, torch.ones_like(counts))
    assert torch.equal(indices, torch.full_like(indices, 3))


def test_cache_row_layout_preserves_batches_and_gradients():
    store = {}
    first = KVSplice(store, 0, None)
    assert first.full_rows() is None
    k = torch.randn(2, 1, 7, 8)
    first(k, k.clone())
    reuse = KVSplice(store, 0, (2, 5))
    assert reuse.full_rows() == (2, 5, 7)
    x = torch.randn(2, 3, 16, requires_grad=True)
    padded = _pad_context_rows(x, reuse.full_rows())
    assert padded.shape == (2, 7, 16)
    assert torch.equal(padded[:, 2:5], x)
    assert not padded[:, :2].count_nonzero()
    assert not padded[:, 5:].count_nonzero()
    padded.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("backend", ["flash", "triton", "cudnn"])
@pytest.mark.parametrize("full_rows", [False, True])
def test_gate_restores_full_row_count(monkeypatch, backend, full_rows):
    monkeypatch.setattr(layers, "CTX_FULL_ROWS", full_rows)
    from praxis import attention

    def fake_attention(q, k, v, **kwargs):
        return torch.zeros_like(q)

    monkeypatch.setattr(
        attention,
        "optional_attention",
        lambda backend, *a, **kw: fake_attention(*a, **kw),
    )
    monkeypatch.setattr(layers, "_compiled_flex_attention", fake_attention)
    module = Attention(16, 8, 1, backend)
    inputs = []
    module.gate_proj.register_forward_pre_hook(lambda m, a: inputs.append(a[0].clone()))
    k = torch.zeros(2, 1, 7, 8)
    kv = KVSplice({0: (k, k.clone())}, 0, (2, 5))
    x = torch.randn(2, 3, 16)
    with torch.inference_mode():
        out = module(x, None, None, kv)
    assert out.shape == x.shape
    expected = _pad_context_rows(x, (2, 5, 7)) if backend == "flash" or full_rows else x
    assert torch.equal(inputs[0], expected)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("start", [0, 4, 8])
def test_router_keeps_full_matrix_and_chunk_boundaries(monkeypatch, batch, start):
    monkeypatch.setattr(layers, "_ROUTER_CHUNK_ROWS", 5)
    module = MoEFeedForward(16, 8, 4, 2)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.01)
    x = torch.randn(batch, 3, 16)
    rows = (start, start + 3, 11)
    padded = _pad_context_rows(x, rows)
    expected = layers._router_logits(module.router.weight, padded.reshape(-1, 16))
    expected = expected.reshape(batch, 11, 4)[:, start : start + 3].reshape(-1, 4)
    observed = []
    original = layers._router_logits

    def router(weight, value):
        observed.append(value.clone())
        return original(weight, value)

    monkeypatch.setattr(layers, "_router_logits", router)
    monkeypatch.setattr(
        module, "_routed_forward", lambda x, top_idx, gate: torch.zeros_like(x)
    )
    logits = []
    module.routing_hook = lambda m, value, indices: logits.append(value.clone())
    with torch.inference_mode():
        module(x, router_rows=rows)
    assert torch.equal(observed[0], padded.reshape(-1, 16))
    assert torch.equal(logits[0], expected)


@pytest.mark.parametrize("driver", ["solo", "pair"])
@pytest.mark.parametrize("backend", ["flash", "triton", "cudnn"])
@pytest.mark.parametrize("batch", [1, 2])
def test_generator_preserves_cached_router_rows(monkeypatch, driver, backend, batch):
    from praxis.layers import TransformerBlock, run_pair, run_solo

    monkeypatch.setattr(layers, "FUSED_NORM", False)
    monkeypatch.setattr(layers, "_ROUTER_CHUNK_ROWS", 5)
    module = MoEFeedForward(16, 8, 4, 2)
    block = TransformerBlock(module, 16, 8, 1, attention_backend=backend)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-0.1, 0.1)
    monkeypatch.setattr(block.mha, "forward", lambda x, *args: torch.zeros_like(x))
    k = torch.zeros(batch, 1, 11, 8)
    seen = []
    original = layers._router_logits

    def router(weight, value):
        seen.append(value.shape[0])
        return original(weight, value)

    monkeypatch.setattr(layers, "_router_logits", router)
    xs = [torch.randn(batch, 3, 16), torch.randn(batch, 3, 16)]
    kvs = [KVSplice({0: (k, k.clone())}, 0, (4, 7)) for _ in xs]
    with torch.inference_mode():
        expected = [block(x, None, kv=kv) for x, kv in zip(xs, kvs, strict=True)]
        seen.clear()
        gens = [
            block.forward_gen(x, None, kv=kv) for x, kv in zip(xs, kvs, strict=True)
        ]
        actual = run_pair(gens) if driver == "pair" else [run_solo(g) for g in gens]
    assert seen == [batch * 11, batch * 11]
    assert not layers._ROUTER_ROWS
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))


@pytest.mark.parametrize("driver", ["solo", "pair"])
@pytest.mark.parametrize("nvtx", [False, True])
def test_cached_row_registration_is_cleared_after_driver_failure(
    monkeypatch, driver, nvtx
):
    from praxis import nvtx as NV
    from praxis.layers import TransformerBlock, run_pair, run_solo

    monkeypatch.setattr(layers, "FUSED_NORM", False)
    monkeypatch.setattr(NV, "ON", nvtx)
    module = MoEFeedForward(16, 8, 4, 2)
    block = TransformerBlock(module, 16, 8, 1, attention_backend="triton")
    monkeypatch.setattr(block.mha, "forward", lambda x, *args: torch.zeros_like(x))

    def fail(*args):
        raise RuntimeError("router failed")

    monkeypatch.setattr(module, "forward", fail)
    k = torch.zeros(1, 1, 11, 8)
    kv = KVSplice({0: (k, k.clone())}, 0, (4, 7))
    gens = [block.forward_gen(torch.randn(1, 3, 16), None, kv=kv)]
    gens = [NV.tag_gen(g, "enc", "g") for g in gens]
    with torch.inference_mode(), pytest.raises(RuntimeError, match="router failed"):
        if driver == "pair":
            run_pair(gens)
        else:
            run_solo(gens[0])
    assert not layers._ROUTER_ROWS


def test_paired_driver_routes_with_full_rows_without_wrapping_moe(monkeypatch):
    from praxis.layers import TransformerBlock, run_pair

    monkeypatch.setattr(layers, "FUSED_NORM", False)
    module = MoEFeedForward(16, 8, 4, 2)
    block = TransformerBlock(module, 16, 8, 1, attention_backend="triton")
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-0.1, 0.1)
    monkeypatch.setattr(block.mha, "forward", lambda x, *args: torch.zeros_like(x))
    k = torch.zeros(1, 1, 11, 8)
    xs = [torch.randn(1, 3, 16), torch.randn(1, 5, 16)]
    kvs = [KVSplice({0: (k, k.clone())}, 0, rows) for rows in [(4, 7), (1, 6)]]
    seen = []
    router = layers._router_logits

    def observe(weight, value):
        seen.append(value.shape[0])
        return router(weight, value)

    monkeypatch.setattr(layers, "_router_logits", observe)
    paired_calls = []

    def paired(values):
        paired_calls.append(len(values))
        out = []
        for value in values:
            flat = value.reshape(-1, value.size(-1))
            indices, gates = module._route(flat)
            out.append(
                module._routed_tail(
                    flat, indices, gates, module.shared(flat)
                ).reshape_as(value)
            )
        return out

    monkeypatch.setattr(module, "forward_pair", paired)
    with torch.inference_mode():
        expected = [block(x, None, kv=kv) for x, kv in zip(xs, kvs, strict=True)]
        seen.clear()
        actual = run_pair(
            [block.forward_gen(x, None, kv=kv) for x, kv in zip(xs, kvs, strict=True)]
        )
    assert paired_calls == [2]
    assert seen == [11, 11]
    assert not layers._ROUTER_ROWS
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
