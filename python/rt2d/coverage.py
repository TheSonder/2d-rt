from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .boundary import (
    GeometryIndex,
    Point,
    VertexRecord,
    _add,
    _compute_visible_subsegments_for_state,
    _distance,
    _extend_to_bounds,
    _has_outward_departure,
    _is_reflective_front_face,
    _lerp,
    _normalize,
    _order_points_by_angle,
    _point_in_polygon,
    _point_in_tube,
    _point_on_segment,
    _query_edge_candidates,
    _ray_segment_hit_parameter,
    _reflect_point,
    _same_point,
    _scale,
    _sub,
    build_geometry,
    is_visible,
    is_visible_with_exclusions,
)
from .scene import load_scene


@dataclass(frozen=True)
class PropagationState:
    sequence: str
    source_point: Point
    apex: Point
    left_dir: Point | None
    right_dir: Point | None
    boundary_points: tuple[Point, Point] | None
    exclude_edge_ids: tuple[int, ...] = ()
    interaction_kind: str = "root"
    source_poly_id: int | None = None
    source_vertex_id: int | None = None


@dataclass(frozen=True)
class SequenceCostConfig:
    distance_weight: float = 0.85
    reflection_penalty: float = 18.0
    diffraction_penalty: float = 26.0
    extra_interaction_penalty: float = 9.0
    max_cost: float = 170.0


@dataclass(frozen=True)
class TorchGeometryCache:
    edge_a: Any
    edge_b: Any
    edge_bbox: Any
    poly_edge_a: Any
    poly_edge_b: Any
    poly_edge_valid: Any
    sample_step: float


@dataclass(frozen=True)
class TorchStateBatch:
    source_points: Any
    left_dirs: Any
    right_dirs: Any
    has_tube_dirs: Any
    diffraction_mask: Any
    source_poly_ids: Any
    exclude_mask: Any
    sequences: tuple[str, ...]


@dataclass
class RxVisibilityRuntime:
    scene_id: str
    geom: GeometryIndex
    grid: dict[str, Any]
    x_coords: tuple[float, ...]
    y_coords: tuple[float, ...]
    outdoor_mask: list[list[bool]]
    outdoor_points: tuple[tuple[int, int, Point], ...]
    outdoor_point_lookup: dict[tuple[int, int], int]
    building_mask_path: str | None
    outdoor_mask_source: str
    acceleration_backend: str
    torch_device: str | None
    torch_geom: TorchGeometryCache | None = None
    torch_outdoor_points: Any = None
    state_cache: dict[tuple[int, int, bool, bool], tuple[dict[int, list[PropagationState]], dict[int, dict[str, list[PropagationState]]]]] = field(
        default_factory=dict
    )
    torch_state_batch_cache: dict[tuple[int, int, bool, bool], dict[int, TorchStateBatch]] = field(
        default_factory=dict
    )
    torch_los_state_batch_cache: dict[int, TorchStateBatch] = field(default_factory=dict)


def _import_torch() -> Any:
    import torch

    return torch


def _resolve_acceleration_backend(
    acceleration_backend: str,
    torch_device: str | None,
) -> tuple[str, str | None]:
    backend = acceleration_backend.strip().lower()
    if backend == "cpu":
        return ("cpu", None)

    torch = _import_torch()
    if backend == "auto":
        if torch.cuda.is_available():
            return ("torch", torch_device or "cuda")
        return ("cpu", None)

    if backend == "torch":
        if torch_device is not None:
            return ("torch", torch_device)
        if torch.cuda.is_available():
            return ("torch", "cuda")
        return ("torch", "cpu")

    raise ValueError("acceleration_backend must be one of: 'cpu', 'auto', 'torch'.")


def _build_torch_geometry_cache(
    geom: GeometryIndex,
    device: str,
) -> TorchGeometryCache:
    torch = _import_torch()
    dtype = torch.float32

    edge_a = torch.tensor([edge.a for edge in geom.edges], dtype=dtype, device=device)
    edge_b = torch.tensor([edge.b for edge in geom.edges], dtype=dtype, device=device)
    edge_bbox = torch.tensor([edge.bbox for edge in geom.edges], dtype=dtype, device=device)

    max_poly_edges = max((len(polygon.vertex_ids) for polygon in geom.polygons), default=1)
    poly_edge_a = torch.zeros((len(geom.polygons), max_poly_edges, 2), dtype=dtype, device=device)
    poly_edge_b = torch.zeros((len(geom.polygons), max_poly_edges, 2), dtype=dtype, device=device)
    poly_edge_valid = torch.zeros((len(geom.polygons), max_poly_edges), dtype=torch.bool, device=device)

    for polygon in geom.polygons:
        points = [geom.vertices[vertex_id].point for vertex_id in polygon.vertex_ids]
        for local_id, point in enumerate(points):
            nxt = points[(local_id + 1) % len(points)]
            poly_edge_a[polygon.poly_id, local_id] = torch.tensor(point, dtype=dtype, device=device)
            poly_edge_b[polygon.poly_id, local_id] = torch.tensor(nxt, dtype=dtype, device=device)
            poly_edge_valid[polygon.poly_id, local_id] = True

    span_x = geom.bounds[2] - geom.bounds[0]
    span_y = geom.bounds[3] - geom.bounds[1]
    sample_step = max(max(span_x, span_y) * 1.0e-4, geom.epsilon * 32.0, 1.0e-3)

    return TorchGeometryCache(
        edge_a=edge_a,
        edge_b=edge_b,
        edge_bbox=edge_bbox,
        poly_edge_a=poly_edge_a,
        poly_edge_b=poly_edge_b,
        poly_edge_valid=poly_edge_valid,
        sample_step=sample_step,
    )


def _build_outdoor_mask_and_points(
    x_coords: list[float],
    y_coords: list[float],
    geom: GeometryIndex,
) -> tuple[list[list[bool]], tuple[tuple[int, int, Point], ...], dict[tuple[int, int], int]]:
    outdoor_mask: list[list[bool]] = []
    outdoor_points: list[tuple[int, int, Point]] = []
    outdoor_point_lookup: dict[tuple[int, int], int] = {}

    for row, y in enumerate(y_coords):
        mask_row: list[bool] = []
        for col, x in enumerate(x_coords):
            point = (x, y)
            is_outdoor = _is_outdoor_point(point, geom)
            mask_row.append(is_outdoor)
            if is_outdoor:
                outdoor_point_lookup[(row, col)] = len(outdoor_points)
                outdoor_points.append((row, col, point))
        outdoor_mask.append(mask_row)

    return (outdoor_mask, tuple(outdoor_points), outdoor_point_lookup)


def _resolve_building_mask_path(scene_data: dict[str, Any]) -> Path | None:
    explicit = scene_data.get("building_mask_path")
    if explicit is not None:
        path = Path(str(explicit))
        if path.is_file():
            return path

    root_dir = scene_data.get("root_dir")
    scene_id = str(scene_data.get("scene_id", "")).strip()
    if root_dir and scene_id:
        candidate = Path(str(root_dir)) / "png" / "buildings_complete" / f"{scene_id}.png"
        if candidate.is_file():
            return candidate
    return None


def _build_outdoor_mask_from_building_png(
    x_coords: list[float],
    y_coords: list[float],
    building_mask_path: Path,
) -> tuple[list[list[bool]], tuple[tuple[int, int, Point], ...], dict[tuple[int, int], int]]:
    image = Image.open(building_mask_path).convert("L")
    mask = image.load()
    width, height = image.size

    outdoor_mask: list[list[bool]] = []
    outdoor_points: list[tuple[int, int, Point]] = []
    outdoor_point_lookup: dict[tuple[int, int], int] = {}

    for row, y in enumerate(y_coords):
        mask_row: list[bool] = []
        for col, x in enumerate(x_coords):
            px = int(round(float(x)))
            py = int(round(float(height - 1 - y)))
            is_outdoor = True
            if 0 <= px < width and 0 <= py < height:
                is_outdoor = int(mask[px, py]) == 0
            mask_row.append(is_outdoor)
            if is_outdoor:
                outdoor_point_lookup[(row, col)] = len(outdoor_points)
                outdoor_points.append((row, col, (x, y)))
        outdoor_mask.append(mask_row)

    return (outdoor_mask, tuple(outdoor_points), outdoor_point_lookup)


def _prepare_torch_state_batch(
    states: list[PropagationState],
    edge_count: int,
    device: str,
) -> TorchStateBatch:
    torch = _import_torch()
    dtype = torch.float32

    source_points = torch.tensor([state.source_point for state in states], dtype=dtype, device=device)
    left_dirs = torch.tensor(
        [state.left_dir if state.left_dir is not None else (0.0, 0.0) for state in states],
        dtype=dtype,
        device=device,
    )
    right_dirs = torch.tensor(
        [state.right_dir if state.right_dir is not None else (0.0, 0.0) for state in states],
        dtype=dtype,
        device=device,
    )
    has_tube_dirs = torch.tensor(
        [state.left_dir is not None and state.right_dir is not None for state in states],
        dtype=torch.bool,
        device=device,
    )
    diffraction_mask = torch.tensor(
        [state.interaction_kind == "diffraction" and state.source_poly_id is not None for state in states],
        dtype=torch.bool,
        device=device,
    )
    source_poly_ids = torch.tensor(
        [state.source_poly_id if state.source_poly_id is not None else 0 for state in states],
        dtype=torch.int64,
        device=device,
    )
    exclude_mask = torch.zeros((len(states), edge_count), dtype=torch.bool, device=device)
    for state_id, state in enumerate(states):
        if state.exclude_edge_ids:
            exclude_mask[state_id, list(state.exclude_edge_ids)] = True

    return TorchStateBatch(
        source_points=source_points,
        left_dirs=left_dirs,
        right_dirs=right_dirs,
        has_tube_dirs=has_tube_dirs,
        diffraction_mask=diffraction_mask,
        source_poly_ids=source_poly_ids,
        exclude_mask=exclude_mask,
        sequences=tuple(state.sequence for state in states),
    )


def build_rx_visibility_runtime(
    scene: dict[str, Any] | str | int,
    *,
    root_dir: str | None = None,
    grid_step: float = 1.0,
    bounds: tuple[float, float, float, float] | None = None,
    epsilon: float = 1.0e-6,
    acceleration_backend: str = "cpu",
    torch_device: str | None = None,
) -> RxVisibilityRuntime:
    if grid_step <= 0.0:
        raise ValueError("grid_step must be > 0.")

    if isinstance(scene, dict):
        scene_data = scene
    else:
        scene_data = load_scene(scene, root_dir=root_dir)

    geom = build_geometry(scene_data, epsilon=epsilon)
    resolved_backend, resolved_torch_device = _resolve_acceleration_backend(
        acceleration_backend,
        torch_device,
    )

    min_x, min_y, max_x, max_y = _resolve_bounds(geom, bounds, grid_step)
    x_coords = _build_axis(min_x, max_x, grid_step)
    y_coords = list(reversed(_build_axis(min_y, max_y, grid_step)))
    building_mask_path = _resolve_building_mask_path(scene_data)
    if building_mask_path is not None:
        outdoor_mask, outdoor_points, outdoor_point_lookup = _build_outdoor_mask_from_building_png(
            x_coords,
            y_coords,
            building_mask_path,
        )
        outdoor_mask_source = "building_png"
    else:
        outdoor_mask, outdoor_points, outdoor_point_lookup = _build_outdoor_mask_and_points(
            x_coords,
            y_coords,
            geom,
        )
        outdoor_mask_source = "geometry_polygon"

    grid = {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "step": grid_step,
        "width": len(x_coords),
        "height": len(y_coords),
        "row_major_y_descending": True,
        "labels": {
            "blocked": -2,
            "unreachable": -1,
            "los": 0,
            "order1": 1,
            "order2": 2,
            "order3": 3,
            "order4": 4,
        },
    }

    torch_geom = None
    torch_outdoor_points = None
    if resolved_backend == "torch" and resolved_torch_device is not None:
        torch_geom = _build_torch_geometry_cache(geom, resolved_torch_device)
        torch = _import_torch()
        torch_outdoor_points = torch.tensor(
            [point for _, _, point in outdoor_points],
            dtype=torch.float32,
            device=resolved_torch_device,
        )

    return RxVisibilityRuntime(
        scene_id=geom.scene_id,
        geom=geom,
        grid=grid,
        x_coords=tuple(x_coords),
        y_coords=tuple(y_coords),
        outdoor_mask=outdoor_mask,
        outdoor_points=outdoor_points,
        outdoor_point_lookup=outdoor_point_lookup,
        building_mask_path=str(building_mask_path) if building_mask_path is not None else None,
        outdoor_mask_source=outdoor_mask_source,
        acceleration_backend=resolved_backend,
        torch_device=resolved_torch_device,
        torch_geom=torch_geom,
        torch_outdoor_points=torch_outdoor_points,
    )


def _make_grid(height: int, width: int, value: int = 0) -> list[list[int]]:
    return [[value for _ in range(width)] for _ in range(height)]


def _make_state(
    sequence: str,
    source_point: Point,
    boundary_points: tuple[Point, Point] | None,
    *,
    exclude_edge_ids: tuple[int, ...] = (),
    interaction_kind: str = "root",
    source_poly_id: int | None = None,
    source_vertex_id: int | None = None,
) -> PropagationState:
    if boundary_points is None:
        return PropagationState(
            sequence=sequence,
            source_point=source_point,
            apex=source_point,
            left_dir=None,
            right_dir=None,
            boundary_points=None,
            exclude_edge_ids=exclude_edge_ids,
            interaction_kind=interaction_kind,
            source_poly_id=source_poly_id,
            source_vertex_id=source_vertex_id,
        )

    left_point, right_point = _order_points_by_angle(source_point, boundary_points[0], boundary_points[1])
    left_dir = _normalize(_sub(left_point, source_point), 1.0e-12)
    right_dir = _normalize(_sub(right_point, source_point), 1.0e-12)
    if left_dir is None or right_dir is None:
        left_dir = None
        right_dir = None

    return PropagationState(
        sequence=sequence,
        source_point=source_point,
        apex=source_point,
        left_dir=left_dir,
        right_dir=right_dir,
        boundary_points=(left_point, right_point),
        exclude_edge_ids=exclude_edge_ids,
        interaction_kind=interaction_kind,
        source_poly_id=source_poly_id,
        source_vertex_id=source_vertex_id,
    )


def _state_key(state: PropagationState, eps: float) -> tuple[Any, ...]:
    scale = max(1.0 / max(eps, 1.0e-9), 1.0e3)

    def q(point: Point | None) -> tuple[int, int] | None:
        if point is None:
            return None
        return (int(round(point[0] * scale)), int(round(point[1] * scale)))

    boundary_points = state.boundary_points or (None, None)
    return (
        state.sequence,
        state.interaction_kind,
        q(state.source_point),
        q(boundary_points[0]),
        q(boundary_points[1]),
        tuple(sorted(state.exclude_edge_ids)),
        state.source_poly_id,
        state.source_vertex_id,
    )


def _dedupe_states(states: list[PropagationState], eps: float) -> list[PropagationState]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[PropagationState] = []
    for state in states:
        key = _state_key(state, eps)
        if key in seen:
            continue
        seen.add(key)
        unique.append(state)
    return unique


def _torch_cross(a: Any, b: Any) -> Any:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _torch_point_on_segment(p: Any, a: Any, b: Any, eps: float) -> Any:
    torch = _import_torch()
    ab = b - a
    ap = p - a
    cross = torch.abs(_torch_cross(ab, ap)) <= eps
    dot = (ap * ab).sum(dim=-1)
    length_sq = (ab * ab).sum(dim=-1)
    return cross & (dot >= -eps) & (dot <= length_sq + eps)


def _torch_points_outside_polygons(
    sample_points: Any,
    poly_ids: Any,
    torch_geom: TorchGeometryCache,
    eps: float,
) -> Any:
    torch = _import_torch()
    edge_a = torch_geom.poly_edge_a[poly_ids]
    edge_b = torch_geom.poly_edge_b[poly_ids]
    edge_valid = torch_geom.poly_edge_valid[poly_ids]

    points = sample_points[:, :, None, :]
    a = edge_a[:, None, :, :]
    b = edge_b[:, None, :, :]
    valid = edge_valid[:, None, :]

    on_segment = _torch_point_on_segment(points, a, b, eps) & valid
    on_edge = on_segment.any(dim=-1)

    yi_above = a[..., 1] > points[..., 1]
    yj_above = b[..., 1] > points[..., 1]
    crosses = (yi_above != yj_above) & valid

    denom = b[..., 1] - a[..., 1]
    denom = torch.where(torch.abs(denom) <= eps, torch.ones_like(denom), denom)
    x_cross = a[..., 0] + (points[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) / denom
    toggles = crosses & (x_cross >= points[..., 0] - eps)
    inside = (toggles.to(torch.int16).sum(dim=-1) % 2) == 1
    return ~(on_edge | inside)


def _torch_visibility_with_exclusions(
    source_points: Any,
    points: Any,
    exclude_mask: Any,
    torch_geom: TorchGeometryCache,
    eps: float,
    edge_chunk_size: int,
) -> Any:
    torch = _import_torch()
    state_count = source_points.shape[0]
    point_count = points.shape[0]
    if state_count == 0 or point_count == 0:
        return torch.zeros((state_count, point_count), dtype=torch.bool, device=source_points.device)

    tx = source_points[:, None, :]
    pts = points[None, :, :]
    seg_min = torch.minimum(tx, pts)
    seg_max = torch.maximum(tx, pts)
    q = pts - tx
    visible = torch.ones((state_count, point_count), dtype=torch.bool, device=source_points.device)
    eps_sq = eps * eps

    for edge_start in range(0, torch_geom.edge_a.shape[0], edge_chunk_size):
        edge_stop = min(edge_start + edge_chunk_size, torch_geom.edge_a.shape[0])
        a = torch_geom.edge_a[edge_start:edge_stop][None, None, :, :]
        b = torch_geom.edge_b[edge_start:edge_stop][None, None, :, :]
        bbox = torch_geom.edge_bbox[edge_start:edge_stop]
        excluded = exclude_mask[:, None, edge_start:edge_stop]

        bbox_ok = ~(
            (seg_max[:, :, 0:1] < (bbox[None, None, :, 0] - eps))
            | ((bbox[None, None, :, 2] + eps) < seg_min[:, :, 0:1])
            | (seg_max[:, :, 1:2] < (bbox[None, None, :, 1] - eps))
            | ((bbox[None, None, :, 3] + eps) < seg_min[:, :, 1:2])
        )
        if not bool(bbox_ok.any().item()):
            continue

        points_expanded = points[None, :, None, :]
        source_expanded = source_points[:, None, None, :]
        point_on_seg = _torch_point_on_segment(points_expanded, a, b, eps)
        same_a = ((points_expanded - a) ** 2).sum(dim=-1) <= eps_sq
        same_b = ((points_expanded - b) ** 2).sum(dim=-1) <= eps_sq

        r0 = a - source_expanded
        r1 = b - source_expanded
        len_ok = ((r0 * r0).sum(dim=-1) > eps_sq) & ((r1 * r1).sum(dim=-1) > eps_sq)

        cross_r = _torch_cross(r0, r1)
        swap = cross_r < 0.0
        a_ray = torch.where(swap[..., None], r1, r0)
        b_ray = torch.where(swap[..., None], r0, r1)
        q_expanded = q[:, :, None, :]
        in_wedge = (_torch_cross(a_ray, q_expanded) >= -eps) & (_torch_cross(q_expanded, b_ray) >= -eps)

        edge_vec = b - a
        tx_side = _torch_cross(edge_vec, source_expanded - a)
        point_side = _torch_cross(edge_vec, points_expanded - a)
        blocked = (
            bbox_ok
            & (~excluded)
            & len_ok
            & (torch.abs(tx_side) > eps)
            & in_wedge
            & (~point_on_seg)
            & (~same_a)
            & (~same_b)
            & ((tx_side * point_side) <= eps)
        )
        if bool(blocked.any().item()):
            visible &= ~blocked.any(dim=-1)
            if not bool(visible.any().item()):
                break

    return visible


def _torch_state_chunk_reaches_points(
    state_batch: TorchStateBatch,
    point_tensor: Any,
    torch_geom: TorchGeometryCache,
    eps: float,
    edge_chunk_size: int,
) -> Any:
    torch = _import_torch()
    state_count = state_batch.source_points.shape[0]
    point_count = point_tensor.shape[0]
    if state_count == 0 or point_count == 0:
        return torch.zeros((state_count, point_count), dtype=torch.bool, device=point_tensor.device)

    q = point_tensor[None, :, :] - state_batch.source_points[:, None, :]
    if bool(state_batch.has_tube_dirs.any().item()):
        tube_ok = torch.ones((state_count, point_count), dtype=torch.bool, device=point_tensor.device)
        masked_indices = torch.nonzero(state_batch.has_tube_dirs, as_tuple=False).flatten()
        left_dirs = state_batch.left_dirs[masked_indices][:, None, :]
        right_dirs = state_batch.right_dirs[masked_indices][:, None, :]
        q_masked = q[masked_indices]
        tube_ok_masked = (_torch_cross(left_dirs, q_masked) >= -eps) & (_torch_cross(q_masked, right_dirs) >= -eps)
        tube_ok[masked_indices] = tube_ok_masked
    else:
        tube_ok = torch.ones((state_count, point_count), dtype=torch.bool, device=point_tensor.device)

    departure_ok = torch.ones((state_count, point_count), dtype=torch.bool, device=point_tensor.device)
    if bool(state_batch.diffraction_mask.any().item()):
        diffraction_indices = torch.nonzero(state_batch.diffraction_mask, as_tuple=False).flatten()
        q_diffraction = q[diffraction_indices]
        lengths = torch.sqrt((q_diffraction * q_diffraction).sum(dim=-1))
        unit = torch.zeros_like(q_diffraction)
        valid = lengths > eps
        unit[valid] = q_diffraction[valid] / lengths[valid][:, None]
        sample = state_batch.source_points[diffraction_indices][:, None, :] + unit * torch_geom.sample_step
        outside = _torch_points_outside_polygons(
            sample,
            state_batch.source_poly_ids[diffraction_indices],
            torch_geom,
            eps,
        )
        departure_ok_diffraction = outside & valid
        departure_ok[diffraction_indices] = departure_ok_diffraction

    prelim = tube_ok & departure_ok
    if not bool(prelim.any().item()):
        return prelim

    visible = _torch_visibility_with_exclusions(
        state_batch.source_points,
        point_tensor,
        state_batch.exclude_mask,
        torch_geom,
        eps,
        edge_chunk_size,
    )
    return prelim & visible


def _torch_evaluate_state_hits(
    states: list[PropagationState],
    points: list[Point],
    torch_geom: TorchGeometryCache,
    device: str,
    eps: float,
    state_chunk_size: int,
    point_chunk_size: int,
    edge_chunk_size: int,
) -> tuple[list[bool], dict[str, list[bool]]]:
    torch = _import_torch()
    if not states or not points:
        return ([False for _ in points], {})

    with torch.no_grad():
        point_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        state_batch = _prepare_torch_state_batch(states, torch_geom.edge_a.shape[0], device)
        overall_hits: list[bool] = [False for _ in points]
        sequence_hits: dict[str, list[bool]] = {sequence: [False for _ in points] for sequence in set(state_batch.sequences)}

        for point_start in range(0, point_tensor.shape[0], point_chunk_size):
            point_stop = min(point_start + point_chunk_size, point_tensor.shape[0])
            chunk_points = point_tensor[point_start:point_stop]
            chunk_overall = torch.zeros((chunk_points.shape[0],), dtype=torch.bool, device=device)
            chunk_sequences = {
                sequence: torch.zeros((chunk_points.shape[0],), dtype=torch.bool, device=device)
                for sequence in sequence_hits
            }

            for state_start in range(0, state_batch.source_points.shape[0], state_chunk_size):
                state_stop = min(state_start + state_chunk_size, state_batch.source_points.shape[0])
                chunk_state_batch = TorchStateBatch(
                    source_points=state_batch.source_points[state_start:state_stop],
                    left_dirs=state_batch.left_dirs[state_start:state_stop],
                    right_dirs=state_batch.right_dirs[state_start:state_stop],
                    has_tube_dirs=state_batch.has_tube_dirs[state_start:state_stop],
                    diffraction_mask=state_batch.diffraction_mask[state_start:state_stop],
                    source_poly_ids=state_batch.source_poly_ids[state_start:state_stop],
                    exclude_mask=state_batch.exclude_mask[state_start:state_stop],
                    sequences=state_batch.sequences[state_start:state_stop],
                )
                hit_matrix = _torch_state_chunk_reaches_points(
                    chunk_state_batch,
                    chunk_points,
                    torch_geom,
                    eps,
                    edge_chunk_size,
                )
                if hit_matrix.shape[0] == 0:
                    continue

                chunk_overall |= hit_matrix.any(dim=0)
                local_sequences = sorted(set(chunk_state_batch.sequences))
                for sequence in local_sequences:
                    state_ids = [index for index, value in enumerate(chunk_state_batch.sequences) if value == sequence]
                    if state_ids:
                        chunk_sequences[sequence] |= hit_matrix[state_ids].any(dim=0)

                if bool(chunk_overall.all().item()) and all(bool(value.all().item()) for value in chunk_sequences.values()):
                    break

            overall_cpu = chunk_overall.to("cpu").tolist()
            for offset, value in enumerate(overall_cpu):
                overall_hits[point_start + offset] = bool(value)

            for sequence, hits in chunk_sequences.items():
                hits_cpu = hits.to("cpu").tolist()
                for offset, value in enumerate(hits_cpu):
                    sequence_hits[sequence][point_start + offset] = bool(value)

    return (overall_hits, sequence_hits)


def _torch_evaluate_state_hits_from_indices(
    states: list[PropagationState] | None,
    point_indices: list[int],
    runtime: RxVisibilityRuntime,
    state_chunk_size: int,
    point_chunk_size: int,
    edge_chunk_size: int,
    state_batch: TorchStateBatch | None = None,
) -> tuple[list[bool], dict[str, list[bool]]]:
    torch = _import_torch()
    if state_batch is None and not states:
        return ([False for _ in point_indices], {})
    if not point_indices:
        return ([False for _ in point_indices], {})
    if runtime.torch_outdoor_points is None or runtime.torch_geom is None or runtime.torch_device is None:
        raise ValueError("runtime does not contain torch geometry cache.")

    with torch.no_grad():
        index_tensor = torch.tensor(point_indices, dtype=torch.int64, device=runtime.torch_device)
        point_tensor = runtime.torch_outdoor_points.index_select(0, index_tensor)
        prepared_state_batch = state_batch or _prepare_torch_state_batch(
            states or [],
            runtime.torch_geom.edge_a.shape[0],
            runtime.torch_device,
        )
        overall_hits: list[bool] = [False for _ in point_indices]
        sequence_hits: dict[str, list[bool]] = {
            sequence: [False for _ in point_indices]
            for sequence in set(prepared_state_batch.sequences)
        }

        for point_start in range(0, point_tensor.shape[0], point_chunk_size):
            point_stop = min(point_start + point_chunk_size, point_tensor.shape[0])
            chunk_points = point_tensor[point_start:point_stop]
            chunk_overall = torch.zeros((chunk_points.shape[0],), dtype=torch.bool, device=runtime.torch_device)
            chunk_sequences = {
                sequence: torch.zeros((chunk_points.shape[0],), dtype=torch.bool, device=runtime.torch_device)
                for sequence in sequence_hits
            }

            for state_start in range(0, prepared_state_batch.source_points.shape[0], state_chunk_size):
                state_stop = min(state_start + state_chunk_size, prepared_state_batch.source_points.shape[0])
                chunk_state_batch = TorchStateBatch(
                    source_points=prepared_state_batch.source_points[state_start:state_stop],
                    left_dirs=prepared_state_batch.left_dirs[state_start:state_stop],
                    right_dirs=prepared_state_batch.right_dirs[state_start:state_stop],
                    has_tube_dirs=prepared_state_batch.has_tube_dirs[state_start:state_stop],
                    diffraction_mask=prepared_state_batch.diffraction_mask[state_start:state_stop],
                    source_poly_ids=prepared_state_batch.source_poly_ids[state_start:state_stop],
                    exclude_mask=prepared_state_batch.exclude_mask[state_start:state_stop],
                    sequences=prepared_state_batch.sequences[state_start:state_stop],
                )
                hit_matrix = _torch_state_chunk_reaches_points(
                    chunk_state_batch,
                    chunk_points,
                    runtime.torch_geom,
                    runtime.geom.epsilon,
                    edge_chunk_size,
                )
                if hit_matrix.shape[0] == 0:
                    continue

                chunk_overall |= hit_matrix.any(dim=0)
                local_sequences = sorted(set(chunk_state_batch.sequences))
                for sequence in local_sequences:
                    state_ids = [index for index, value in enumerate(chunk_state_batch.sequences) if value == sequence]
                    if state_ids:
                        chunk_sequences[sequence] |= hit_matrix[state_ids].any(dim=0)

            overall_cpu = chunk_overall.to("cpu").tolist()
            for offset, value in enumerate(overall_cpu):
                overall_hits[point_start + offset] = bool(value)

            for sequence, hits in chunk_sequences.items():
                hits_cpu = hits.to("cpu").tolist()
                for offset, value in enumerate(hits_cpu):
                    sequence_hits[sequence][point_start + offset] = bool(value)

    return (overall_hits, sequence_hits)


def _resolve_bounds(
    geom: GeometryIndex,
    bounds: tuple[float, float, float, float] | None,
    grid_step: float,
) -> tuple[float, float, float, float]:
    if bounds is None:
        min_x = math.floor(geom.bounds[0] / grid_step) * grid_step
        min_y = math.floor(geom.bounds[1] / grid_step) * grid_step
        max_x = math.ceil(geom.bounds[2] / grid_step) * grid_step
        max_y = math.ceil(geom.bounds[3] / grid_step) * grid_step
        return (min_x, min_y, max_x, max_y)

    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    if max_x < min_x or max_y < min_y:
        raise ValueError("bounds must satisfy min <= max on both axes.")
    return (min_x, min_y, max_x, max_y)


def _build_axis(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step))
    values = [start + i * step for i in range(count + 1)]
    values[-1] = end
    return values


def _is_outdoor_point(point: Point, geom: GeometryIndex) -> bool:
    for polygon in geom.polygons:
        polygon_points = [geom.vertices[vertex_id].point for vertex_id in polygon.vertex_ids]
        if _point_in_polygon(point, polygon_points, geom.epsilon):
            return False
    return True


def _trace_to_first_collision_with_exclusions(
    origin: Point,
    direction: Point,
    geom: GeometryIndex,
    exclude_edge_ids: tuple[int, ...] = (),
) -> tuple[Point, int | None]:
    eps = geom.epsilon
    unit = _normalize(direction, eps)
    if unit is None:
        return (origin, None)

    ray_end = _extend_to_bounds(origin, unit, geom)
    best_distance = _distance(origin, ray_end)
    best_edge_id: int | None = None
    excluded = set(exclude_edge_ids)

    candidates = _query_edge_candidates(origin, ray_end, geom)
    for edge_id in candidates:
        if edge_id in excluded:
            continue

        edge = geom.edges[edge_id]
        if _point_on_segment(origin, edge.a, edge.b, eps):
            continue

        hit_t = _ray_segment_hit_parameter(origin, unit, edge.a, edge.b, eps)
        if hit_t is None:
            continue
        if hit_t >= best_distance - eps:
            continue

        best_distance = hit_t
        best_edge_id = edge_id

    return _add(origin, _scale(unit, best_distance)), best_edge_id


def _first_hit_edge_id_with_exclusions(
    source_point: Point,
    theta: float,
    geom: GeometryIndex,
    exclude_edge_ids: tuple[int, ...],
) -> int | None:
    direction = (math.cos(theta), math.sin(theta))
    _, edge_id = _trace_to_first_collision_with_exclusions(
        source_point,
        direction,
        geom,
        exclude_edge_ids=exclude_edge_ids,
    )
    return edge_id


def _is_state_vertex_critical(
    state: PropagationState,
    vertex: VertexRecord,
    geom: GeometryIndex,
) -> bool:
    if _same_point(state.source_point, vertex.point, geom.epsilon):
        return False
    if not _point_in_tube(state, vertex.point, geom.epsilon):
        return False
    if not is_visible_with_exclusions(
        state.source_point,
        vertex.point,
        geom,
        state.exclude_edge_ids,
    ):
        return False

    theta = math.atan2(vertex.point[1] - state.source_point[1], vertex.point[0] - state.source_point[0])
    delta = geom.angle_epsilon
    left = _first_hit_edge_id_with_exclusions(
        state.source_point,
        theta - delta,
        geom,
        state.exclude_edge_ids,
    )
    right = _first_hit_edge_id_with_exclusions(
        state.source_point,
        theta + delta,
        geom,
        state.exclude_edge_ids,
    )
    return left != right


def _expand_reflection_successors(
    state: PropagationState,
    geom: GeometryIndex,
) -> list[PropagationState]:
    children: list[PropagationState] = []
    sequence = f"{state.sequence}R" if state.sequence else "R"

    for edge in geom.edges:
        if edge.edge_id in state.exclude_edge_ids:
            continue
        if not _is_reflective_front_face(state.source_point, edge, geom):
            continue

        visible_subsegments = _compute_visible_subsegments_for_state(state, edge, geom)
        if not visible_subsegments:
            continue

        image_source = _reflect_point(state.source_point, edge.a, edge.b, geom.epsilon)
        exclude_edge_ids = tuple(
            sorted(set(state.exclude_edge_ids).union(geom.polygons[edge.poly_id].edge_ids))
        )
        for start, end in visible_subsegments:
            left_point = _lerp(edge.a, edge.b, start)
            right_point = _lerp(edge.a, edge.b, end)
            left_point, right_point = _order_points_by_angle(image_source, left_point, right_point)
            children.append(
                _make_state(
                    sequence,
                    image_source,
                    (left_point, right_point),
                    exclude_edge_ids=exclude_edge_ids,
                    interaction_kind="reflection",
                    source_poly_id=edge.poly_id,
                )
            )

    return _dedupe_states(children, geom.epsilon)


def _expand_diffraction_successors(
    state: PropagationState,
    geom: GeometryIndex,
) -> list[PropagationState]:
    children: list[PropagationState] = []
    sequence = f"{state.sequence}D" if state.sequence else "D"

    for vertex in geom.vertices:
        if vertex.vertex_id == state.source_vertex_id:
            continue
        if not _is_state_vertex_critical(state, vertex, geom):
            continue

        exclude_edge_ids = tuple(
            sorted(
                set(state.exclude_edge_ids).union(
                    {
                        vertex.prev_edge_id,
                        vertex.next_edge_id,
                    }
                )
            )
        )
        children.append(
            _make_state(
                sequence,
                vertex.point,
                None,
                exclude_edge_ids=exclude_edge_ids,
                interaction_kind="diffraction",
                source_poly_id=vertex.poly_id,
                source_vertex_id=vertex.vertex_id,
            )
        )

    return _dedupe_states(children, geom.epsilon)


def _get_or_build_state_expansion(
    runtime: RxVisibilityRuntime,
    tx_id: int,
    max_interactions: int,
    enable_reflection: bool,
    enable_diffraction: bool,
) -> tuple[dict[int, list[PropagationState]], dict[int, dict[str, list[PropagationState]]]]:
    cache_key = (tx_id, max_interactions, enable_reflection, enable_diffraction)
    cached = runtime.state_cache.get(cache_key)
    if cached is not None:
        return cached

    states_by_order: dict[int, list[PropagationState]] = {}
    sequence_groups_by_order: dict[int, dict[str, list[PropagationState]]] = {}
    frontier = [_make_state("", runtime.geom.antennas[tx_id], None)]

    for depth in range(1, max_interactions + 1):
        next_frontier: list[PropagationState] = []
        for state in frontier:
            if enable_reflection:
                next_frontier.extend(_expand_reflection_successors(state, runtime.geom))
            if enable_diffraction:
                next_frontier.extend(_expand_diffraction_successors(state, runtime.geom))
        frontier = _dedupe_states(next_frontier, runtime.geom.epsilon)
        states_by_order[depth] = frontier

        groups: dict[str, list[PropagationState]] = {}
        for state in frontier:
            groups.setdefault(state.sequence, []).append(state)
        sequence_groups_by_order[depth] = groups

    runtime.state_cache[cache_key] = (states_by_order, sequence_groups_by_order)
    return (states_by_order, sequence_groups_by_order)


def _get_or_build_torch_state_batches(
    runtime: RxVisibilityRuntime,
    tx_id: int,
    max_interactions: int,
    enable_reflection: bool,
    enable_diffraction: bool,
) -> dict[int, TorchStateBatch]:
    cache_key = (tx_id, max_interactions, enable_reflection, enable_diffraction)
    cached = runtime.torch_state_batch_cache.get(cache_key)
    if cached is not None:
        return cached
    if runtime.torch_geom is None or runtime.torch_device is None:
        raise ValueError("runtime does not contain torch geometry cache.")

    states_by_order, _sequence_groups_by_order = _get_or_build_state_expansion(
        runtime,
        tx_id,
        max_interactions,
        enable_reflection,
        enable_diffraction,
    )
    batches: dict[int, TorchStateBatch] = {}
    for depth, states in states_by_order.items():
        batches[depth] = _prepare_torch_state_batch(
            states,
            runtime.torch_geom.edge_a.shape[0],
            runtime.torch_device,
        )
    runtime.torch_state_batch_cache[cache_key] = batches
    return batches


def _get_or_build_torch_los_state_batch(
    runtime: RxVisibilityRuntime,
    tx_id: int,
) -> TorchStateBatch:
    cached = runtime.torch_los_state_batch_cache.get(tx_id)
    if cached is not None:
        return cached
    if runtime.torch_geom is None or runtime.torch_device is None:
        raise ValueError("runtime does not contain torch geometry cache.")

    batch = _prepare_torch_state_batch(
        [_make_state("L", runtime.geom.antennas[tx_id], None)],
        runtime.torch_geom.edge_a.shape[0],
        runtime.torch_device,
    )
    runtime.torch_los_state_batch_cache[tx_id] = batch
    return batch


def _state_reaches_rx(
    state: PropagationState,
    rx: Point,
    geom: GeometryIndex,
) -> bool:
    if not _point_in_tube(state, rx, geom.epsilon):
        return False

    if state.interaction_kind == "diffraction":
        if state.source_poly_id is None:
            return False
        if not _has_outward_departure(
            state.source_point,
            _sub(rx, state.source_point),
            state.source_poly_id,
            geom,
        ):
            return False

    return is_visible_with_exclusions(
        state.source_point,
        rx,
        geom,
        state.exclude_edge_ids,
    )


def _sequence_priority_key(sequence: str) -> tuple[int, int, tuple[int, ...]]:
    if len(sequence) == 1:
        if sequence == "R":
            return (10, 10, (1,))
        if sequence == "D":
            return (0, 0, (0,))

    reflection_count = sequence.count("R")
    ends_with_reflection = 1 if sequence.endswith("R") else 0
    char_rank = tuple(1 if char == "R" else 0 for char in sequence)
    return (reflection_count, ends_with_reflection, char_rank)


def _is_pure_diffraction(sequence: str) -> bool:
    return bool(sequence) and set(sequence) == {"D"}


def _can_override_sequence(existing_label: str, candidate_sequence: str) -> bool:
    if not _is_pure_diffraction(existing_label):
        return False
    if len(candidate_sequence) != len(existing_label) + 1:
        return False
    if len(candidate_sequence) == 2:
        return candidate_sequence == "RR"
    return candidate_sequence.endswith("R")


def build_layered_sequence_render_grid(
    sequence_hit_grids: dict[str, list[list[int]]],
    outdoor_mask: list[list[bool]],
) -> tuple[list[list[str]], dict[str, int]]:
    if not outdoor_mask or not outdoor_mask[0]:
        return [], {}

    height = len(outdoor_mask)
    width = len(outdoor_mask[0])
    result = [["blocked" if not outdoor_mask[row][col] else "unreachable" for col in range(width)] for row in range(height)]

    los_grid = sequence_hit_grids.get("L")
    if los_grid is not None:
        for row in range(height):
            for col in range(width):
                if outdoor_mask[row][col] and los_grid[row][col]:
                    result[row][col] = "L"

    max_depth = max((len(sequence) for sequence in sequence_hit_grids if sequence != "L"), default=0)
    for depth in range(1, max_depth + 1):
        sequences = [sequence for sequence in sequence_hit_grids if len(sequence) == depth]
        if not sequences:
            continue
        ordered_sequences = sorted(sequences, key=_sequence_priority_key, reverse=True)
        for row in range(height):
            for col in range(width):
                if not outdoor_mask[row][col]:
                    continue

                best_sequence: str | None = None
                for sequence in ordered_sequences:
                    if sequence_hit_grids[sequence][row][col]:
                        best_sequence = sequence
                        break

                if best_sequence is None:
                    continue

                current = result[row][col]
                if current == "unreachable":
                    result[row][col] = best_sequence
                    continue
                if _can_override_sequence(current, best_sequence):
                    result[row][col] = best_sequence

    counts: dict[str, int] = {}
    for row in result:
        for label in row:
            counts[label] = counts.get(label, 0) + 1

    return result, counts


def _sequence_cost(
    sequence: str,
    distance: float,
    config: SequenceCostConfig,
) -> float:
    if sequence == "L":
        return config.distance_weight * distance

    reflection_count = sequence.count("R")
    diffraction_count = sequence.count("D")
    return (
        config.distance_weight * distance
        + reflection_count * config.reflection_penalty
        + diffraction_count * config.diffraction_penalty
        + max(0, len(sequence) - 1) * config.extra_interaction_penalty
    )


def build_energy_pruned_sequence_render_grid(
    sequence_hit_grids: dict[str, list[list[int]]],
    outdoor_mask: list[list[bool]],
    tx_point: Point,
    grid: dict[str, Any],
    config: SequenceCostConfig | None = None,
) -> tuple[list[list[str]], dict[str, int]]:
    if not outdoor_mask or not outdoor_mask[0]:
        return [], {}

    cfg = config or SequenceCostConfig()
    height = len(outdoor_mask)
    width = len(outdoor_mask[0])
    step = float(grid["step"])
    min_x = float(grid["min_x"])
    max_y = float(grid["max_y"])

    result = [["blocked" if not outdoor_mask[row][col] else "unreachable" for col in range(width)] for row in range(height)]
    sequences = sorted(sequence_hit_grids.keys(), key=_sequence_priority_key, reverse=True)

    for row in range(height):
        y = max_y - row * step
        for col in range(width):
            if not outdoor_mask[row][col]:
                continue

            x = min_x + col * step
            distance = math.hypot(x - tx_point[0], y - tx_point[1])
            best_sequence: str | None = None
            best_cost: float | None = None

            for sequence in sequences:
                if not sequence_hit_grids[sequence][row][col]:
                    continue
                cost = _sequence_cost(sequence, distance, cfg)
                if cost > cfg.max_cost:
                    continue
                if best_cost is None or cost < best_cost - 1.0e-9:
                    best_sequence = sequence
                    best_cost = cost
                    continue
                if best_cost is not None and abs(cost - best_cost) <= 1.0e-9:
                    if best_sequence is None or _sequence_priority_key(sequence) > _sequence_priority_key(best_sequence):
                        best_sequence = sequence
                        best_cost = cost

            if best_sequence is not None:
                result[row][col] = best_sequence

    counts: dict[str, int] = {}
    for row in result:
        for label in row:
            counts[label] = counts.get(label, 0) + 1
    return result, counts


def _export_rx_visibility_json(
    payload: dict[str, Any],
    output_path: str | Path | None,
) -> dict[str, Any]:
    if output_path is None:
        return payload

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def compute_rx_visibility_runtime(
    runtime: RxVisibilityRuntime,
    *,
    tx_ids: list[int] | None = None,
    max_interactions: int = 1,
    enable_reflection: bool = True,
    enable_diffraction: bool = True,
    include_sequence_render_grid: bool = False,
    include_sequence_hit_grids: bool = False,
    torch_state_chunk_size: int = 16,
    torch_point_chunk_size: int = 4096,
    torch_edge_chunk_size: int = 64,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if max_interactions < 0:
        raise ValueError("max_interactions must be >= 0.")
    if max_interactions > 4:
        raise NotImplementedError("max_interactions > 4 is not implemented yet.")
    if not enable_reflection and not enable_diffraction and max_interactions > 0:
        max_interactions = 0

    indices = tx_ids if tx_ids is not None else list(range(len(runtime.geom.antennas)))
    antenna_count = len(runtime.geom.antennas)
    for tx_id in indices:
        if tx_id < 0 or tx_id >= antenna_count:
            raise ValueError(f"tx_id out of range: {tx_id}. Valid range: 0..{antenna_count - 1}")

    payload: dict[str, Any] = {
        "scene_id": runtime.scene_id,
        "grid": runtime.grid,
        "tx_results": [],
    }

    height = len(runtime.y_coords)
    width = len(runtime.x_coords)

    for tx_id in indices:
        tx = runtime.geom.antennas[tx_id]
        states_by_order, sequence_groups_by_order = _get_or_build_state_expansion(
            runtime,
            tx_id,
            max_interactions,
            enable_reflection,
            enable_diffraction,
        )
        torch_state_batches = (
            _get_or_build_torch_state_batches(
                runtime,
                tx_id,
                max_interactions,
                enable_reflection,
                enable_diffraction,
            )
            if runtime.acceleration_backend == "torch" and runtime.torch_geom is not None
            else {}
        )

        visibility_order_grid = [
            [-2 if not runtime.outdoor_mask[row][col] else -1 for col in range(width)]
            for row in range(height)
        ]
        counts = {
            "blocked": 0,
            "unreachable": 0,
            "los": 0,
            "order1": 0,
            "order2": 0,
            "order3": 0,
            "order4": 0,
        }
        for row in range(height):
            for col in range(width):
                if runtime.outdoor_mask[row][col]:
                    continue
                counts["blocked"] += 1

        sequence_hit_grids: dict[str, list[list[int]]] | None = None
        if include_sequence_render_grid:
            sequence_hit_grids = {"L": _make_grid(height, width)}
            for groups in sequence_groups_by_order.values():
                for sequence in groups:
                    sequence_hit_grids.setdefault(sequence, _make_grid(height, width))

        outdoor_entries = list(runtime.outdoor_points)
        if runtime.acceleration_backend == "torch" and runtime.torch_geom is not None:
            point_indices = list(range(len(outdoor_entries)))
            los_hits, _los_sequence_hits = _torch_evaluate_state_hits_from_indices(
                None,
                point_indices,
                runtime,
                torch_state_chunk_size,
                torch_point_chunk_size,
                torch_edge_chunk_size,
                state_batch=_get_or_build_torch_los_state_batch(runtime, tx_id),
            )
        else:
            los_hits = [is_visible(tx, point, runtime.geom) for _, _, point in outdoor_entries]

        remaining: list[tuple[int, int, Point]] = []
        remaining_point_indices: list[int] = []
        for index, (row, col, point) in enumerate(outdoor_entries):
            if los_hits[index]:
                visibility_order_grid[row][col] = 0
                counts["los"] += 1
                if sequence_hit_grids is not None:
                    sequence_hit_grids["L"][row][col] = 1
            else:
                remaining.append((row, col, point))
                remaining_point_indices.append(index)

        for depth in range(1, max_interactions + 1):
            states = states_by_order.get(depth, [])
            sequence_groups = sequence_groups_by_order.get(depth, {})
            if not states or not remaining:
                continue

            if runtime.acceleration_backend == "torch" and runtime.torch_geom is not None:
                state_hits, sequence_hits = _torch_evaluate_state_hits_from_indices(
                    None,
                    remaining_point_indices,
                    runtime,
                    torch_state_chunk_size,
                    torch_point_chunk_size,
                    torch_edge_chunk_size,
                    state_batch=torch_state_batches.get(depth),
                )
                if sequence_hit_grids is not None and sequence_groups:
                    for sequence in sequence_groups:
                        hits = sequence_hits.get(sequence)
                        if hits is None:
                            continue
                        for index, (row, col, _point) in enumerate(remaining):
                            if hits[index]:
                                sequence_hit_grids[sequence][row][col] = 1

                next_remaining: list[tuple[int, int, Point]] = []
                next_remaining_indices: list[int] = []
                for index, (row, col, point) in enumerate(remaining):
                    if state_hits[index]:
                        visibility_order_grid[row][col] = depth
                        counts[f"order{depth}"] += 1
                    else:
                        next_remaining.append((row, col, point))
                        next_remaining_indices.append(remaining_point_indices[index])
                remaining = next_remaining
                remaining_point_indices = next_remaining_indices
                continue

            if sequence_hit_grids is not None and sequence_groups:
                for row, col, point in remaining:
                    for sequence, grouped_states in sequence_groups.items():
                        if any(_state_reaches_rx(state, point, runtime.geom) for state in grouped_states):
                            sequence_hit_grids[sequence][row][col] = 1

            next_remaining: list[tuple[int, int, Point]] = []
            next_remaining_indices: list[int] = []
            for index, (row, col, point) in enumerate(remaining):
                if any(_state_reaches_rx(state, point, runtime.geom) for state in states):
                    visibility_order_grid[row][col] = depth
                    counts[f"order{depth}"] += 1
                else:
                    next_remaining.append((row, col, point))
                    next_remaining_indices.append(remaining_point_indices[index])
            remaining = next_remaining
            remaining_point_indices = next_remaining_indices

        counts["unreachable"] = len(remaining)
        tx_result = {
            "tx_id": tx_id,
            "counts": counts,
            "state_counts": {
                f"order{depth}": len(states_by_order.get(depth, []))
                for depth in range(1, max_interactions + 1)
            },
            "visibility_order_grid": visibility_order_grid,
        }
        if sequence_hit_grids is not None:
            layered_grid, layered_counts = build_layered_sequence_render_grid(
                sequence_hit_grids,
                runtime.outdoor_mask,
            )
            tx_result["layered_sequence_grid"] = layered_grid
            tx_result["layered_sequence_counts"] = layered_counts
            if include_sequence_hit_grids:
                tx_result["sequence_hit_grids"] = sequence_hit_grids
        payload["tx_results"].append(tx_result)

    return _export_rx_visibility_json(payload, output_path)


def warm_rx_visibility_runtime(
    runtime: RxVisibilityRuntime,
    *,
    tx_ids: list[int] | None = None,
    max_interactions: int = 1,
    enable_reflection: bool = True,
    enable_diffraction: bool = True,
) -> None:
    indices = tx_ids if tx_ids is not None else list(range(len(runtime.geom.antennas)))
    antenna_count = len(runtime.geom.antennas)
    for tx_id in indices:
        if tx_id < 0 or tx_id >= antenna_count:
            raise ValueError(f"tx_id out of range: {tx_id}. Valid range: 0..{antenna_count - 1}")

    for tx_id in indices:
        _get_or_build_state_expansion(
            runtime,
            tx_id,
            max_interactions,
            enable_reflection,
            enable_diffraction,
        )
        if runtime.acceleration_backend == "torch" and runtime.torch_geom is not None:
            _get_or_build_torch_los_state_batch(runtime, tx_id)
            _get_or_build_torch_state_batches(
                runtime,
                tx_id,
                max_interactions,
                enable_reflection,
                enable_diffraction,
            )


def compute_rx_visibility(
    scene: dict[str, Any] | str | int,
    *,
    tx_ids: list[int] | None = None,
    root_dir: str | None = None,
    max_interactions: int = 1,
    grid_step: float = 1.0,
    bounds: tuple[float, float, float, float] | None = None,
    epsilon: float = 1.0e-6,
    enable_reflection: bool = True,
    enable_diffraction: bool = True,
    include_sequence_render_grid: bool = False,
    include_sequence_hit_grids: bool = False,
    acceleration_backend: str = "cpu",
    torch_device: str | None = None,
    torch_state_chunk_size: int = 16,
    torch_point_chunk_size: int = 4096,
    torch_edge_chunk_size: int = 64,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    runtime = build_rx_visibility_runtime(
        scene,
        root_dir=root_dir,
        grid_step=grid_step,
        bounds=bounds,
        epsilon=epsilon,
        acceleration_backend=acceleration_backend,
        torch_device=torch_device,
    )
    return compute_rx_visibility_runtime(
        runtime,
        tx_ids=tx_ids,
        max_interactions=max_interactions,
        enable_reflection=enable_reflection,
        enable_diffraction=enable_diffraction,
        include_sequence_render_grid=include_sequence_render_grid,
        include_sequence_hit_grids=include_sequence_hit_grids,
        torch_state_chunk_size=torch_state_chunk_size,
        torch_point_chunk_size=torch_point_chunk_size,
        torch_edge_chunk_size=torch_edge_chunk_size,
        output_path=output_path,
    )
