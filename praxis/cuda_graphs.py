"""CUDA Graphs for frozen text, action-expert and NVFP4 MoE inference.

CPU preparation stays eager. Each graph owns stable inputs and outputs; callers
receive independent copies so replay cannot change clean history or saved taps.
Memory pools are shared only under serialized execution on one CUDA stream.
The window-joint action expert uses private pools and events across streams.
"""

from __future__ import annotations

import logging
from dataclasses import fields, is_dataclass
from functools import wraps
from threading import Lock

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _map(value, tensor_fn):
    if isinstance(value, Tensor):
        return tensor_fn(value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{f.name: _map(getattr(value, f.name), tensor_fn) for f in fields(value)}
        )
    if isinstance(value, dict):
        return {k: _map(v, tensor_fn) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_map(v, tensor_fn) for v in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"CUDA Graph boundary cannot hold {type(value).__name__}")


def _signature(value):
    if isinstance(value, Tensor):
        if not value.is_cuda:
            raise ValueError("CUDA Graph inputs must be CUDA tensors")
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )
    if isinstance(value, dict):
        return ("dict", tuple((k, _signature(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return (type(value), tuple(_signature(v) for v in value))
    if value is None or isinstance(value, str | int | float | bool):
        return (type(value), value)
    raise TypeError(f"CUDA Graph input cannot hold {type(value).__name__}")


def _tensors(value):
    result = []
    _map(value, lambda x: result.append(x))
    return result


class CudaGraphPool:
    """One pipeline's serialized graph cache.

    Inference only. Module weights must stay frozen and on the same device.
    A new shape is captured lazily until :meth:`freeze` prohibits new captures.
    This makes unexpected capture during timed serving a clear error.
    Once a stage holds ``max_graphs`` shapes, new shapes run eagerly. The first
    such call logs a warning, and :meth:`stats` counts them as ``eager``.
    """

    def __init__(
        self, device, *, warmup=3, max_graphs=8, validate=True, allow_streams=False
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA Graphs require a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if warmup < 1 or max_graphs < 1:
            raise ValueError("warmup and max_graphs must be positive")
        with torch.cuda.device(self.device):
            self.pool = torch.cuda.graph_pool_handle()
            self.capture_stream = torch.cuda.Stream()
        self.warmup = warmup
        self.max_graphs = max_graphs
        self.validate = validate
        self.allow_streams = allow_streams
        self._completion = None
        self.frozen = False
        self._lock = Lock()
        self._execution_stream = None
        self._stages = {}
        self._originals = []

    def wrap(self, owner, attr, name):
        original = getattr(owner, attr)
        cache = {}
        stats = {"captures": 0, "replays": 0, "validated": 0, "eager": 0}
        self._stages[name] = stats

        @wraps(original)
        def call(*args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "CUDA Graph inference requires no_grad/inference_mode"
                )
            if not self._lock.acquire(blocking=False):
                raise RuntimeError(
                    "CUDA Graph pipeline does not support concurrent calls"
                )
            try:
                with torch.cuda.device(self.device):
                    stream = torch.cuda.current_stream()
                    if self._execution_stream is None:
                        self._execution_stream = stream.cuda_stream
                    if (
                        not self.allow_streams
                        and stream.cuda_stream != self._execution_stream
                    ):
                        raise RuntimeError(
                            "CUDA Graph pipeline must run on one CUDA stream"
                        )
                    if self._completion is not None:
                        stream.wait_event(self._completion)
                    inputs = (args, kwargs)
                    if any(
                        x.is_cuda and x.device != self.device for x in _tensors(inputs)
                    ):
                        raise ValueError(
                            "CUDA Graph inputs must use the pipeline's CUDA device"
                        )
                    key = (
                        _signature(inputs),
                        torch.is_inference_mode_enabled(),
                        torch.is_autocast_enabled("cuda"),
                        torch.get_autocast_dtype("cuda"),
                    )
                    if key not in cache:
                        if self.frozen:
                            raise RuntimeError(f"Unwarmed CUDA Graph shape in {name}")
                        if len(cache) >= self.max_graphs:
                            if not stats["eager"]:
                                logger.warning(
                                    "%s holds %d CUDA Graph shapes; new shapes "
                                    "run eagerly",
                                    name,
                                    self.max_graphs,
                                )
                            stats["eager"] += 1
                            return original(*args, **kwargs)
                        logger.info(
                            "Capturing %s specialization %d", name, len(cache) + 1
                        )
                        static = _map(inputs, lambda x: x.detach().clone())
                        reference = original(*args, **kwargs) if self.validate else None
                        self.capture_stream.wait_stream(stream)
                        with torch.cuda.stream(self.capture_stream):
                            for _ in range(self.warmup):
                                warm_output = original(*static[0], **static[1])
                            del warm_output
                        stream.wait_stream(self.capture_stream)
                        torch.cuda.synchronize(self.device)
                        # Uncaptured warmup can leave unused allocations in
                        # the default allocator cache. Return those unused
                        # blocks before allocating from the graph-private pool.
                        torch.cuda.empty_cache()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            graph,
                            pool=None if self.allow_streams else self.pool,
                            stream=self.capture_stream,
                        ):
                            output = original(*static[0], **static[1])
                        # Keep the graph's static tensor tree alive for its lifetime.
                        stats["captures"] += 1
                        if self.validate:
                            graph.replay()
                            actual, expected = _tensors(output), _tensors(reference)
                            if len(actual) != len(expected) or any(
                                a.shape != b.shape
                                or a.dtype != b.dtype
                                or not torch.equal(
                                    a.contiguous().reshape(-1).view(torch.uint8),
                                    b.contiguous().reshape(-1).view(torch.uint8),
                                )
                                for a, b in zip(actual, expected, strict=True)
                            ):
                                raise RuntimeError(
                                    f"CUDA Graph changed output bits in {name}"
                                )
                            stats["validated"] += 1
                        del reference
                        cache[key] = (graph, static, output)
                        logger.info("Captured %s specialization %d", name, len(cache))
                    graph, static, output = cache[key]
                    incoming = iter(_tensors(inputs))
                    _map(static, lambda target: target.copy_(next(incoming)))
                    graph.replay()
                    stats["replays"] += 1
                    result = _map(output, lambda x: x.clone())
                    if self.allow_streams:
                        # Another policy call can use a new side stream. Keep
                        # static inputs and outputs alive until their copies
                        # finish before that stream reuses the same graph.
                        if self._completion is None:
                            self._completion = torch.cuda.Event()
                        self._completion.record(stream)
                    return result
            finally:
                self._lock.release()

        setattr(owner, attr, call)
        self._originals.append((owner, attr, original))

    def freeze(self):
        """Reject new shapes after the application's warmup phase."""
        self.frozen = True

    def stats(self):
        return {name: dict(values) for name, values in self._stages.items()}


def enable_nvfp4_moe_cuda_graphs(pipeline, *, warmup=3, max_graphs=8, validate=True):
    """Capture SM100 NVFP4 MoEs, including routing, packing and reduction.

    Shape-specific capture happens on first use. Each capture checks output
    bytes against eager execution. Weights and MoE settings must stay fixed.
    The shared pool runs on the backbone stream; action graphs stay separate.
    """
    from praxis.layers import MoEFeedForward

    if hasattr(pipeline, "nvfp4_moe_cuda_graphs"):
        raise ValueError("NVFP4 MoE CUDA Graphs are already enabled")
    if torch.device(pipeline.device).type != "cuda" or torch.cuda.get_device_capability(
        pipeline.device
    ) != (10, 0):
        raise ValueError("NVFP4 MoE CUDA Graphs require SM100 CUDA")
    if (
        pipeline.variant.backbone_layout != "chunk_causal"
        or pipeline.sampler.layout != "window_joint"
        or getattr(pipeline.policy, "routed_quantization", {}).get("format") != "nvfp4"
    ):
        raise ValueError("NVFP4 MoE CUDA Graphs require window_joint NVFP4")
    targets = [
        (name, module)
        for name, module in pipeline.policy.backbone.named_modules()
        if isinstance(module, MoEFeedForward)
        and getattr(module.routed, "format", None) == "nvfp4"
    ]
    if not targets or any(
        module.training or any(p.requires_grad for p in module.parameters())
        for _, module in targets
    ):
        raise ValueError("NVFP4 MoE CUDA Graphs require frozen eval MoEs")
    pool = CudaGraphPool(
        pipeline.device, warmup=warmup, max_graphs=max_graphs, validate=validate
    )
    for name, module in targets:
        pool.wrap(module, "forward", name)
    pipeline.nvfp4_moe_cuda_graphs = pool
    return pool


def enable_action_cuda_graphs(pipeline, *, warmup=3, max_graphs=8, validate=True):
    """Capture the chunk-causal action expert with changing tensor inputs.

    Each context depth and prompt length gets a graph. Input preparation,
    DiT, text and VAE stay eager. Guided sampling can replay on its side
    stream. Graphs own separate memory pools and return independent outputs.
    Weights and captured settings must stay fixed after enabling graphs.
    """
    if hasattr(pipeline, "action_cuda_graphs") or hasattr(pipeline, "cuda_graphs"):
        raise ValueError("CUDA Graphs are already enabled")
    if (
        pipeline.variant.backbone_layout != "chunk_causal"
        or pipeline.sampler.layout != "window_joint"
    ):
        raise ValueError("Action CUDA Graphs require the window_joint sampler")
    expert = pipeline.policy.action_expert
    if expert.training or any(p.requires_grad for p in expert.parameters()):
        raise ValueError("Action CUDA Graphs require frozen eval weights")
    pool = CudaGraphPool(
        pipeline.device,
        warmup=warmup,
        max_graphs=max_graphs,
        validate=validate,
        allow_streams=True,
    )
    pool.wrap(expert, "forward_trunk_rope", "action_expert")
    pipeline.action_cuda_graphs = pool
    return pool


def enable_cuda_graphs(pipeline, *, warmup=3, max_graphs=8, validate=True):
    """Graph text and action-expert stages of the bimanual pipeline.

    Install after optional VAE compilation and before opening any sessions.
    Shape-specific setup and capture happen on first use. DiT and VAE stages
    execute directly, outside the graph pool.
    """
    if hasattr(pipeline, "cuda_graphs") or hasattr(pipeline, "action_cuda_graphs"):
        raise ValueError("CUDA Graphs are already enabled")
    if (
        pipeline.embodiment != "bimanual"
        or pipeline.sampler.layout != "chunk"
        or pipeline.sampler.guided
    ):
        raise ValueError(
            "CUDA Graphs currently support the unguided bimanual chunk serve path"
        )
    if pipeline.variant.backbone_layout != "serve":
        raise ValueError("CUDA Graphs require the serve backbone")
    if any(
        p.requires_grad
        for model in (pipeline.policy.action_expert, pipeline.text.model)
        for p in model.parameters()
    ):
        raise ValueError("CUDA Graphs require frozen inference weights")
    language = getattr(pipeline.text.model, "language_model", None)
    if language is None or not hasattr(language, "layers"):
        raise ValueError("CUDA Graph text capture requires the Qwen3.5 language model")
    pool = CudaGraphPool(
        pipeline.device, warmup=warmup, max_graphs=max_graphs, validate=validate
    )
    # HF's mask preparation reads GPU values on the CPU. Keep that preparation
    # eager and capture each text transformer layer, embedding, RoPE and final
    # norm. Hooks collecting hidden_states then also run outside capture.
    pool.wrap(language.embed_tokens, "forward", "text_embedding")
    pool.wrap(language.rotary_emb, "forward", "text_rotary")
    for index, layer in enumerate(language.layers):
        pool.wrap(layer, "forward", f"text_layer_{index}")
    pool.wrap(language.norm, "forward", "text_norm")
    pool.wrap(pipeline.policy.action_expert, "forward", "action_expert")
    pipeline.cuda_graphs = pool
    return pool
