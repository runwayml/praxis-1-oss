"""Device guards for optional Jetson Thor kernels."""

import torch


def is_thor(device: torch.device | str) -> bool:
    """Match SM110 exactly; SM100 B200 and SM90 H100 keep their own paths."""
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (11, 0)


def is_blackwell_tcgen05(device: torch.device | str) -> bool:
    """Blackwell GPUs with tcgen05 Quack GEMMs: SM100/SM103 (B200/B300) and SM110."""
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] in (
        10,
        11,
    )


def require_thor(device: torch.device | str) -> None:
    """Reject other devices before importing optional kernel dependencies."""
    if not is_thor(device):
        raise RuntimeError("This optional kernel requires Jetson Thor (SM110)")
