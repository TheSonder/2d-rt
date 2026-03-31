from __future__ import annotations

from typing import Any

from ..boundary import _add, _ray_segment_hit_parameter, _scale, _sub
from ..coverage import _state_reaches_rx, _torch_evaluate_state_hits_from_indices
from .runtime import PathFamilyRuntime, build_path_family_runtime
from .types import DiffractionInteractionRef, PathFamily, RayHit


def _family_hits_mask(
    runtime: PathFamilyRuntime,
    family: PathFamily,
    *,
    torch_state_chunk_size: int = 16,
    torch_point_chunk_size: int = 4096,
    torch_edge_chunk_size: int = 64,
) -> list[bool]:
    rx_runtime = runtime.rx_runtime
    if rx_runtime.acceleration_backend == "torch" and rx_runtime.torch_geom is not None:
        point_indices = list(range(len(rx_runtime.outdoor_points)))
        hits, _sequence_hits = _torch_evaluate_state_hits_from_indices(
            [family.state],
            point_indices,
            rx_runtime,
            torch_state_chunk_size,
            torch_point_chunk_size,
            torch_edge_chunk_size,
        )
        return hits

    return [
        _state_reaches_rx(family.state, point, runtime.geometry)
        for _, _, point in rx_runtime.outdoor_points
    ]


def _reconstruct_reflection_point(
    runtime: PathFamilyRuntime,
    family: PathFamily,
    rx_point: tuple[float, float],
) -> tuple[float, float] | None:
    if family.interaction_ref is None:
        return None

    edge = runtime.geometry.edges[family.interaction_ref.edge_id]
    origin = family.state.source_point
    direction = _sub(rx_point, origin)
    t = _ray_segment_hit_parameter(origin, direction, edge.a, edge.b, runtime.geometry.epsilon)
    if t is None:
        return None
    return _add(origin, _scale(direction, t))


def compute_rx_rays_runtime(
    runtime: PathFamilyRuntime,
    *,
    torch_state_chunk_size: int = 16,
    torch_point_chunk_size: int = 4096,
    torch_edge_chunk_size: int = 64,
) -> dict[str, Any]:
    families = []
    if runtime.los_family is not None:
        families.append(runtime.los_family)
    families.extend(runtime.reflection_families)
    families.extend(runtime.diffraction_families)

    rx_hits: dict[tuple[int, int], list[RayHit]] = {}

    for family in families:
        hits = _family_hits_mask(
            runtime,
            family,
            torch_state_chunk_size=torch_state_chunk_size,
            torch_point_chunk_size=torch_point_chunk_size,
            torch_edge_chunk_size=torch_edge_chunk_size,
        )
        for index, hit in enumerate(hits):
            if not hit:
                continue
            row, col, rx_point = runtime.rx_runtime.outdoor_points[index]

            if family.sequence == "L":
                ray_hit = RayHit(
                    family_id=family.family_id,
                    sequence="L",
                    rx_row=row,
                    rx_col=col,
                    rx_point=rx_point,
                    interaction_types=(),
                    interaction_points=(),
                    path_points=(runtime.tx_point, rx_point),
                )
            elif family.sequence == "R":
                reflection_point = _reconstruct_reflection_point(runtime, family, rx_point)
                if reflection_point is None:
                    continue
                ray_hit = RayHit(
                    family_id=family.family_id,
                    sequence=family.sequence,
                    rx_row=row,
                    rx_col=col,
                    rx_point=rx_point,
                    interaction_types=("R",),
                    interaction_points=(reflection_point,),
                    path_points=(runtime.tx_point, reflection_point, rx_point),
                )
            else:
                interaction_ref = family.interaction_ref
                if not isinstance(interaction_ref, DiffractionInteractionRef):
                    continue
                diffraction_point = interaction_ref.point
                ray_hit = RayHit(
                    family_id=family.family_id,
                    sequence=family.sequence,
                    rx_row=row,
                    rx_col=col,
                    rx_point=rx_point,
                    interaction_types=("D",),
                    interaction_points=(diffraction_point,),
                    path_points=(runtime.tx_point, diffraction_point, rx_point),
                )

            rx_hits.setdefault((row, col), []).append(ray_hit)

    return {
        "scene_id": runtime.scene_id,
        "tx_id": runtime.tx_id,
        "rx_hits": rx_hits,
    }


def compute_rx_partition_runtime(
    runtime: PathFamilyRuntime,
    *,
    torch_state_chunk_size: int = 16,
    torch_point_chunk_size: int = 4096,
    torch_edge_chunk_size: int = 64,
) -> dict[str, Any]:
    rays_payload = compute_rx_rays_runtime(
        runtime,
        torch_state_chunk_size=torch_state_chunk_size,
        torch_point_chunk_size=torch_point_chunk_size,
        torch_edge_chunk_size=torch_edge_chunk_size,
    )

    height = len(runtime.rx_runtime.y_coords)
    width = len(runtime.rx_runtime.x_coords)
    grid = [
        ["blocked" if not runtime.rx_runtime.outdoor_mask[row][col] else "unreachable" for col in range(width)]
        for row in range(height)
    ]

    counts = {
        "blocked": 0,
        "unreachable": 0,
        "L": 0,
        "R": 0,
        "D": 0,
    }
    for row in range(height):
        for col in range(width):
            if not runtime.rx_runtime.outdoor_mask[row][col]:
                counts["blocked"] += 1

    for (row, col), hits in rays_payload["rx_hits"].items():
        sequences = {hit.sequence for hit in hits}
        if "L" in sequences:
            grid[row][col] = "L"
            counts["L"] += 1
            continue
        if "R" in sequences:
            grid[row][col] = "R"
            counts["R"] += 1
            continue
        if "D" in sequences:
            grid[row][col] = "D"
            counts["D"] += 1

    counts["unreachable"] = sum(1 for row in grid for label in row if label == "unreachable")
    return {
        "scene_id": runtime.scene_id,
        "tx_id": runtime.tx_id,
        "partition_grid": grid,
        "counts": counts,
        "rx_hits": rays_payload["rx_hits"],
    }


def compute_rx_partition(
    scene: dict[str, Any] | str | int,
    *,
    root_dir: str | None = None,
    tx_id: int = 0,
    grid_step: float = 1.0,
    bounds: tuple[float, float, float, float] | None = None,
    epsilon: float = 1.0e-6,
    acceleration_backend: str = "cpu",
    torch_device: str | None = None,
    torch_state_chunk_size: int = 16,
    torch_point_chunk_size: int = 4096,
    torch_edge_chunk_size: int = 64,
) -> dict[str, Any]:
    runtime = build_path_family_runtime(
        scene,
        root_dir=root_dir,
        tx_id=tx_id,
        grid_step=grid_step,
        bounds=bounds,
        epsilon=epsilon,
        acceleration_backend=acceleration_backend,
        torch_device=torch_device,
    )
    return compute_rx_partition_runtime(
        runtime,
        torch_state_chunk_size=torch_state_chunk_size,
        torch_point_chunk_size=torch_point_chunk_size,
        torch_edge_chunk_size=torch_edge_chunk_size,
    )
