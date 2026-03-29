from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if max_interactions < 0:
        raise ValueError("max_interactions must be >= 0.")
    if max_interactions > 4:
        raise NotImplementedError("max_interactions > 4 is not implemented yet.")
    if grid_step <= 0.0:
        raise ValueError("grid_step must be > 0.")
    if not enable_reflection and not enable_diffraction and max_interactions > 0:
        max_interactions = 0

    if isinstance(scene, dict):
        scene_data = scene
    else:
        scene_data = load_scene(scene, root_dir=root_dir)

    geom = build_geometry(scene_data, epsilon=epsilon)
    indices = tx_ids if tx_ids is not None else list(range(len(scene_data["antenna"])))
    antenna_count = len(geom.antennas)
    for tx_id in indices:
        if tx_id < 0 or tx_id >= antenna_count:
            raise ValueError(f"tx_id out of range: {tx_id}. Valid range: 0..{antenna_count - 1}")

    min_x, min_y, max_x, max_y = _resolve_bounds(geom, bounds, grid_step)
    x_coords = _build_axis(min_x, max_x, grid_step)
    y_coords = list(reversed(_build_axis(min_y, max_y, grid_step)))

    payload: dict[str, Any] = {
        "scene_id": geom.scene_id,
        "grid": {
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
        },
        "tx_results": [],
    }

    outdoor_mask = [
        [_is_outdoor_point((x, y), geom) for x in x_coords]
        for y in y_coords
    ]
    height = len(y_coords)
    width = len(x_coords)

    for tx_id in indices:
        tx = geom.antennas[tx_id]
        states_by_order: dict[int, list[PropagationState]] = {}
        sequence_groups_by_order: dict[int, dict[str, list[PropagationState]]] = {}
        frontier = [_make_state("", tx, None)]

        for depth in range(1, max_interactions + 1):
            next_frontier: list[PropagationState] = []
            for state in frontier:
                if enable_reflection:
                    next_frontier.extend(_expand_reflection_successors(state, geom))
                if enable_diffraction:
                    next_frontier.extend(_expand_diffraction_successors(state, geom))
            frontier = _dedupe_states(next_frontier, geom.epsilon)
            states_by_order[depth] = frontier
            groups: dict[str, list[PropagationState]] = {}
            for state in frontier:
                groups.setdefault(state.sequence, []).append(state)
            sequence_groups_by_order[depth] = groups

        visibility_order_grid: list[list[int]] = []
        remaining: list[tuple[int, int, Point]] = []
        sequence_hit_grids: dict[str, list[list[int]]] | None = None
        if include_sequence_render_grid:
            sequence_hit_grids = {"L": _make_grid(height, width)}
            for groups in sequence_groups_by_order.values():
                for sequence in groups:
                    sequence_hit_grids.setdefault(sequence, _make_grid(height, width))
        counts = {
            "blocked": 0,
            "unreachable": 0,
            "los": 0,
            "order1": 0,
            "order2": 0,
            "order3": 0,
            "order4": 0,
        }

        for row, y in enumerate(y_coords):
            grid_row: list[int] = []
            for col, x in enumerate(x_coords):
                point = (x, y)
                if not outdoor_mask[row][col]:
                    grid_row.append(-2)
                    counts["blocked"] += 1
                    continue

                if is_visible(tx, point, geom):
                    grid_row.append(0)
                    counts["los"] += 1
                    if sequence_hit_grids is not None:
                        sequence_hit_grids["L"][row][col] = 1
                    continue

                grid_row.append(-1)
                remaining.append((row, col, point))
            visibility_order_grid.append(grid_row)

        for depth in range(1, max_interactions + 1):
            states = states_by_order.get(depth, [])
            sequence_groups = sequence_groups_by_order.get(depth, {})
            if sequence_hit_grids is not None and sequence_groups:
                for row, col, point in remaining:
                    for sequence, grouped_states in sequence_groups.items():
                        if any(_state_reaches_rx(state, point, geom) for state in grouped_states):
                            sequence_hit_grids[sequence][row][col] = 1

            if not states or not remaining:
                continue

            next_remaining: list[tuple[int, int, Point]] = []
            for row, col, point in remaining:
                if any(_state_reaches_rx(state, point, geom) for state in states):
                    visibility_order_grid[row][col] = depth
                    counts[f"order{depth}"] += 1
                    continue
                next_remaining.append((row, col, point))
            remaining = next_remaining

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
                outdoor_mask,
            )
            tx_result["layered_sequence_grid"] = layered_grid
            tx_result["layered_sequence_counts"] = layered_counts
        payload["tx_results"].append(tx_result)

    return _export_rx_visibility_json(payload, output_path)
