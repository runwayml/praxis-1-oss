"""Stream a strict checkpoint into routed NVFP4 banks and native dense tensors."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn

from praxis.checkpoint import (
    CHECKSUM_NAME,
    INDEX_NAME,
    CheckpointError,
    expected_dtype,
    weight_map,
)
from praxis.layers import MoEFeedForward, RoutedExperts

_CACHE_VERSION = 2
_LOG = logging.getLogger(__name__)


def _digest(value: Tensor) -> str:
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _checkpoint_id(root: Path, files: dict[str, str]) -> str:
    """Hash the shard checksums, else the index, else the sorted weight map."""
    for name in (CHECKSUM_NAME, INDEX_NAME):
        if (root / name).is_file():
            return hashlib.sha256((root / name).read_bytes()).hexdigest()[:16]
    raw = json.dumps(files, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _torchao_version() -> str | None:
    """The installed TorchAO version; packing needs TorchAO, the cache key not."""
    try:
        return importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return None


def _cached_bank(gate_up: Tensor, down: Tensor, path: Path | None):
    from praxis.quantization.nvfp4 import (
        PACKING_RECIPE,
        NVFP4RoutedExperts,
        pack_routed_experts,
    )

    source = {
        "schema": _CACHE_VERSION,
        "torchao": _torchao_version(),
        "recipe": PACKING_RECIPE,
        "gate_up_shape": list(gate_up.shape),
        "down_shape": list(down.shape),
        "gate_up_sha256": _digest(gate_up),
        "down_sha256": _digest(down),
    }
    if path is not None and path.is_file():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(data, dict)
            or set(data) != {"source", "buffers", "sha256"}
            or data["source"] != source
            or not isinstance(data["buffers"], dict)
            or not isinstance(data["sha256"], dict)
            or set(data["buffers"]) != set(data["sha256"])
        ):
            raise CheckpointError(
                f"NVFP4 cache source or schema mismatch: {path}; delete this cache "
                "file to repack it"
            )
        for name, value in data["buffers"].items():
            if not isinstance(value, Tensor) or _digest(value) != data["sha256"][name]:
                raise CheckpointError(f"NVFP4 cache checksum mismatch: {path}: {name}")
        bank = NVFP4RoutedExperts.from_packed(data["buffers"])
        return bank, True, source

    bank = pack_routed_experts(gate_up, down)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        buffers = bank.state_dict()
        data = {
            "source": source,
            "buffers": buffers,
            "sha256": {name: _digest(value) for name, value in buffers.items()},
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(data, stream)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return bank, False, source


def check_nvfp4_device(device: str | torch.device) -> None:
    """Refuse a CUDA device that cannot run the NVFP4 experts, before loading.

    The grouped NVFP4 GEMM runs on SM120 GPUs (RTX 5090, RTX PRO 6000
    Blackwell) only, and needs FlashInfer from the ``nvfp4`` extra. A CPU
    device is allowed, so that packing and the CPU tests can run.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(
            "routed_precision='nvfp4' needs an SM120 GPU (RTX 5090, RTX PRO 6000 "
            f"Blackwell); {torch.cuda.get_device_name(device)} is "
            f"SM{capability[0]}{capability[1]}. Use routed_precision='bf16'."
        )
    if importlib.util.find_spec("flashinfer") is None:
        raise ImportError(
            "routed_precision='nvfp4' needs FlashInfer: pip install -e '.[nvfp4]'"
        )


def load_routed_nvfp4(
    module: nn.Module,
    path: str | Path,
    *,
    device: str | torch.device,
    cache: str | Path | None = None,
) -> None:
    """Load native tensors and pack one CPU expert bank at a time.

    Validate every checkpoint name, shape and dtype before transferring any
    tensor. The checkpoint must use the chunked, FP32 routed combine. The
    optional cache stores banks under a checkpoint-specific directory and binds
    packed bytes to the original tensor bytes, the TorchAO version and the
    packing recipe; a stale or corrupt entry raises instead of silently loading
    other weights.
    """
    root = Path(path)
    files = weight_map(root)
    expected = module.state_dict(keep_vars=True)
    if set(files) != set(expected):
        missing = sorted(set(expected) - set(files))
        extra = sorted(set(files) - set(expected))
        raise CheckpointError(
            f"checkpoint names differ: missing {missing}, extra {extra}"
        )
    if any(not value.is_meta for value in expected.values()):
        raise ValueError("routed NVFP4 loading requires an unmaterialized meta model")
    banks = {
        name: layer
        for name, layer in module.named_modules()
        if isinstance(layer, MoEFeedForward)
    }
    if not banks:
        raise ValueError("checkpoint has no routed expert banks")
    for name, layer in banks.items():
        if not isinstance(layer.routed, RoutedExperts):
            raise ValueError(f"routed bank is already replaced: {name}")
        if layer.dispatch_chunk_size is None:
            raise ValueError(f"NVFP4 needs chunked routed dispatch: {name}")
    routed_names = {
        f"{name}.routed.{projection}_weight"
        for name in banks
        for projection in ("gate_up", "down")
    }
    by_file: dict[str, set[str]] = {}
    for name, file in files.items():
        if Path(file).name != file:
            raise CheckpointError(f"checkpoint shard must be a file name: {file}")
        by_file.setdefault(file, set()).add(name)

    # Safetensors slices expose the header without materializing the banks.
    dtype_names = {torch.bfloat16: "BF16", torch.float32: "F32"}
    for file, names in by_file.items():
        with safe_open(str(root / file), framework="pt", device="cpu") as reader:
            if set(reader.keys()) != names:
                raise CheckpointError(f"{file}: shard tensors differ from the index")
            for name in names:
                view = reader.get_slice(name)
                if tuple(view.get_shape()) != tuple(expected[name].shape):
                    raise CheckpointError(f"{name}: checkpoint shape mismatch")
                dtype = expected_dtype(module.config, name)
                if dtype not in dtype_names or view.get_dtype() != dtype_names[dtype]:
                    raise CheckpointError(f"{name}: checkpoint dtype mismatch")

    native = {}
    for file, names in by_file.items():
        with safe_open(str(root / file), framework="pt", device="cpu") as reader:
            for name in sorted(names - routed_names):
                native[name] = reader.get_tensor(name).to(device=device)
    result = module.load_state_dict(native, strict=False, assign=True)
    if set(result.missing_keys) != routed_names or result.unexpected_keys:
        raise CheckpointError(
            "native tensor assignment did not match the routed exclusion"
        )
    del native, expected

    receipts = []
    cache_root = None if cache is None else Path(cache) / _checkpoint_id(root, files)
    for index, (name, layer) in enumerate(banks.items(), 1):
        values = []
        for projection in ("gate_up", "down"):
            key = f"{name}.routed.{projection}_weight"
            with safe_open(
                str(root / files[key]), framework="pt", device="cpu"
            ) as reader:
                values.append(reader.get_tensor(key))
        cache_path = None if cache_root is None else cache_root / f"{name}.pt"
        _LOG.info("Loading routed NVFP4 bank %d/%d: %s", index, len(banks), name)
        bank, hit, source = _cached_bank(*values, cache_path)
        original_bytes = sum(v.numel() * v.element_size() for v in values)
        del values
        layer.routed = bank.to(device=device)
        receipts.append(
            {
                "name": name,
                "cache_hit": hit,
                "source": source,
                "original_bytes": original_bytes,
                "packed_bytes": sum(
                    value.numel() * value.element_size() for value in bank.buffers()
                ),
            }
        )
    if any(value.is_meta for value in module.state_dict().values()):
        raise CheckpointError("meta tensors remain after routed NVFP4 loading")
    module.routed_quantization = {
        "format": "nvfp4",
        "scope": "routed experts only",
        "activation_scale": "dynamic per expert (QAT recipe)",
        "banks": receipts,
        "native_tensors": len(files) - len(routed_names),
        "non_routed_checkpoint_dtypes_preserved": True,
    }
