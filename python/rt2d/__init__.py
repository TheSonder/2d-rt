from __future__ import annotations

import torch

from .loader import load_library
from .scene import load_scene


def raytrace(
    origins: torch.Tensor,
    directions: torch.Tensor,
    sphere_center: torch.Tensor,
    sphere_radius: float = 1.0,
) -> torch.Tensor:
    load_library()
    return torch.ops.rt2d.raytrace(origins, directions, sphere_center, float(sphere_radius))


__all__ = ["load_library", "load_scene", "raytrace"]
