from __future__ import annotations

from .boundary import (
    build_geometry,
    compute_visible_subsegments,
    export_boundaries_json,
    extract_los_boundaries,
    extract_reflection_boundaries,
    extract_scene_boundaries,
    is_visible,
    merge_and_label_boundaries,
)
from .coverage import (
    build_rx_visibility_runtime,
    compute_rx_visibility,
    compute_rx_visibility_runtime,
    warm_rx_visibility_runtime,
)
from .path_family import (
    build_path_family_runtime,
    compute_rx_partition as compute_path_family_partition,
    compute_rx_partition_runtime as compute_path_family_partition_runtime,
    compute_rx_rays_runtime,
)
from .loader import load_library
from .scene import load_scene


def raytrace(
    origins,
    directions,
    sphere_center,
    sphere_radius: float = 1.0,
) -> object:
    import torch

    load_library()
    return torch.ops.rt2d.raytrace(origins, directions, sphere_center, float(sphere_radius))


__all__ = [
    "build_geometry",
    "build_path_family_runtime",
    "build_rx_visibility_runtime",
    "compute_visible_subsegments",
    "compute_path_family_partition",
    "compute_path_family_partition_runtime",
    "compute_rx_rays_runtime",
    "compute_rx_visibility",
    "compute_rx_visibility_runtime",
    "warm_rx_visibility_runtime",
    "export_boundaries_json",
    "extract_los_boundaries",
    "extract_reflection_boundaries",
    "extract_scene_boundaries",
    "is_visible",
    "load_library",
    "load_scene",
    "merge_and_label_boundaries",
    "raytrace",
]
