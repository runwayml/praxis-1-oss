"""Optional inference quantization. GPU dependencies load only when used."""

from praxis.quantization.nvfp4 import NVFP4RoutedExperts, pack_routed_experts

__all__ = ["NVFP4RoutedExperts", "pack_routed_experts"]
