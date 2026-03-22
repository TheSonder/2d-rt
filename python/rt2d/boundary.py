from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .scene import load_scene

Point = tuple[float, float]


def _point(value: Iterable[float]) -> Point:
    x, y = value
    return (float(x), float(y))


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _scale(v: Point, factor: float) -> Point:
    return (v[0] * factor, v[1] * factor)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _orient(a: Point, b: Point, c: Point) -> float:
    return _cross(_sub(b, a), _sub(c, a))


def _length(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _distance(a: Point, b: Point) -> float:
    return _length(_sub(a, b))


def _almost_equal(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps


def _same_point(a: Point, b: Point, eps: float) -> bool:
    return _distance(a, b) <= eps


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _bbox_of_points(points: Iterable[Point]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    eps: float,
) -> bool:
    return not (
        a[2] < b[0] - eps
        or b[2] < a[0] - eps
        or a[3] < b[1] - eps
        or b[3] < a[1] - eps
    )


def _remove_closing_point(points: list[list[float]], eps: float) -> list[Point]:
    result = [_point(p) for p in points]
    if len(result) >= 2 and _same_point(result[0], result[-1], eps):
        return result[:-1]
    return result


def _polygon_area(points: list[Point]) -> float:
    area = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        area += p[0] * q[1] - q[0] * p[1]
    return area * 0.5


def _point_on_segment(p: Point, a: Point, b: Point, eps: float) -> bool:
    ab = _sub(b, a)
    ap = _sub(p, a)
    if abs(_cross(ab, ap)) > eps:
        return False
    dot = _dot(ap, ab)
    if dot < -eps:
        return False
    if dot > _dot(ab, ab) + eps:
        return False
    return True


def _point_in_polygon(point: Point, polygon: list[Point], eps: float) -> bool:
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if _point_on_segment(point, a, b, eps):
            return True

    inside = False
    x, y = point
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        yi_above = a[1] > y
        yj_above = b[1] > y
        if yi_above == yj_above:
            continue

        x_cross = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        if x_cross >= x - eps:
            inside = not inside

    return inside


def _clip_interval_geq(
    interval: tuple[float, float] | None,
    slope: float,
    bias: float,
    eps: float,
) -> tuple[float, float] | None:
    if interval is None:
        return None

    t0, t1 = interval
    if abs(slope) <= eps:
        return interval if bias >= -eps else None

    threshold = (-eps - bias) / slope
    if slope > 0:
        t0 = max(t0, threshold)
    else:
        t1 = min(t1, threshold)

    if t0 > t1 + eps:
        return None
    return (max(0.0, t0), min(1.0, t1))


def _clip_interval_leq(
    interval: tuple[float, float] | None,
    slope: float,
    bias: float,
    eps: float,
) -> tuple[float, float] | None:
    return _clip_interval_geq(interval, -slope, -bias, eps)


def _merge_intervals(intervals: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if not intervals:
        return []

    ordered = sorted((max(0.0, a), min(1.0, b)) for a, b in intervals if b - a > eps)
    if not ordered:
        return []

    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        current = merged[-1]
        if start <= current[1] + eps:
            current[1] = max(current[1], end)
            continue
        merged.append([start, end])

    return [(start, end) for start, end in merged]


def _complement_intervals(intervals: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if not intervals:
        return [(0.0, 1.0)]

    visible: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in intervals:
        if start > cursor + eps:
            visible.append((cursor, start))
        cursor = max(cursor, end)

    if cursor < 1.0 - eps:
        visible.append((cursor, 1.0))

    return [(a, b) for a, b in visible if b - a > eps]


def _unique_intervals(intervals: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    merged = _merge_intervals(intervals, eps)
    return [(round(a, 12), round(b, 12)) for a, b in merged]


def _normalize(v: Point, eps: float) -> Point | None:
    length = _length(v)
    if length <= eps:
        return None
    return (v[0] / length, v[1] / length)


def _line_intersection_parameter(a: Point, b: Point, p: Point, direction: Point, eps: float) -> float | None:
    edge = _sub(b, a)
    denom = _cross(direction, edge)
    if abs(denom) <= eps:
        return None
    delta = _sub(a, p)
    ray_t = _cross(delta, edge) / denom
    seg_t = _cross(delta, direction) / denom
    if ray_t <= eps:
        return None
    if seg_t < -eps or seg_t > 1.0 + eps:
        return None
    return ray_t


def _ray_segment_hit_parameter(
    origin: Point,
    direction: Point,
    a: Point,
    b: Point,
    eps: float,
) -> float | None:
    return _line_intersection_parameter(a, b, origin, direction, eps)


def _reflect_point(point: Point, a: Point, b: Point, eps: float) -> Point:
    edge = _sub(b, a)
    length_sq = _dot(edge, edge)
    if length_sq <= eps:
        return point
    ap = _sub(point, a)
    proj = _dot(ap, edge) / length_sq
    foot = _add(a, _scale(edge, proj))
    return _add(foot, _sub(foot, point))


@dataclass(frozen=True)
class VertexRecord:
    vertex_id: int
    poly_id: int
    local_id: int
    point: Point
    prev_edge_id: int
    next_edge_id: int


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: int
    poly_id: int
    local_id: int
    start_vertex_id: int
    end_vertex_id: int
    a: Point
    b: Point
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PolygonRecord:
    poly_id: int
    vertex_ids: tuple[int, ...]
    edge_ids: tuple[int, ...]
    area: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class BoundaryRecord:
    type: str
    p0: Point
    p1: Point
    source: dict[str, int | None]
    mechanism: str
    sequence: str
    scene_id: str
    tx_id: int


@dataclass(frozen=True)
class InteractionSeed:
    point: Point
    sequence: str
    source: dict[str, int | None]
    exclude_edge_ids: tuple[int, ...] = ()


class UniformGrid:
    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        cell_size: float,
    ) -> None:
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[int]] = {}

    def _cell_index(self, value: float, base: float) -> int:
        return math.floor((value - base) / self.cell_size)

    def _cells_for_bbox(self, bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
        min_x, min_y, max_x, max_y = bbox
        ix0 = self._cell_index(min_x, self.min_x)
        iy0 = self._cell_index(min_y, self.min_y)
        ix1 = self._cell_index(max_x, self.min_x)
        iy1 = self._cell_index(max_y, self.min_y)

        cells: list[tuple[int, int]] = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cells.append((ix, iy))
        return cells

    def insert(self, bbox: tuple[float, float, float, float], edge_id: int) -> None:
        for cell in self._cells_for_bbox(bbox):
            bucket = self.cells.setdefault(cell, [])
            bucket.append(edge_id)

    def query_bbox(self, bbox: tuple[float, float, float, float]) -> set[int]:
        result: set[int] = set()
        for cell in self._cells_for_bbox(bbox):
            result.update(self.cells.get(cell, ()))
        return result


@dataclass(frozen=True)
class GeometryIndex:
    scene_id: str
    epsilon: float
    bounds: tuple[float, float, float, float]
    extended_bounds: tuple[float, float, float, float]
    antennas: tuple[Point, ...]
    vertices: tuple[VertexRecord, ...]
    edges: tuple[EdgeRecord, ...]
    polygons: tuple[PolygonRecord, ...]
    grid: UniformGrid
    angle_epsilon: float


def build_geometry(scene: dict[str, Any], epsilon: float = 1.0e-6) -> GeometryIndex:
    antennas = tuple(_point(p) for p in scene["antenna"])

    vertices: list[VertexRecord] = []
    edges: list[EdgeRecord] = []
    polygons: list[PolygonRecord] = []

    all_points: list[Point] = list(antennas)

    for poly_id, raw_polygon in enumerate(scene["polygons"]):
        polygon = _remove_closing_point(raw_polygon, epsilon)
        if len(polygon) < 3:
            raise ValueError(f"polygon[{poly_id}] must contain at least 3 unique points.")

        start_vertex = len(vertices)
        start_edge = len(edges)
        area = _polygon_area(polygon)
        bbox = _bbox_of_points(polygon)
        all_points.extend(polygon)

        vertex_ids: list[int] = []
        edge_ids: list[int] = []
        count = len(polygon)

        for local_id, point in enumerate(polygon):
            vertex_id = start_vertex + local_id
            prev_edge_id = start_edge + (local_id - 1) % count
            next_edge_id = start_edge + local_id
            vertices.append(
                VertexRecord(
                    vertex_id=vertex_id,
                    poly_id=poly_id,
                    local_id=local_id,
                    point=point,
                    prev_edge_id=prev_edge_id,
                    next_edge_id=next_edge_id,
                )
            )
            vertex_ids.append(vertex_id)

        for local_id, point in enumerate(polygon):
            a = point
            b = polygon[(local_id + 1) % count]
            edge_id = start_edge + local_id
            edges.append(
                EdgeRecord(
                    edge_id=edge_id,
                    poly_id=poly_id,
                    local_id=local_id,
                    start_vertex_id=start_vertex + local_id,
                    end_vertex_id=start_vertex + (local_id + 1) % count,
                    a=a,
                    b=b,
                    bbox=_bbox_of_points([a, b]),
                )
            )
            edge_ids.append(edge_id)

        polygons.append(
            PolygonRecord(
                poly_id=poly_id,
                vertex_ids=tuple(vertex_ids),
                edge_ids=tuple(edge_ids),
                area=area,
                bbox=bbox,
            )
        )

    if not all_points:
        raise ValueError("scene does not contain geometry points.")

    bounds = _bbox_of_points(all_points)
    span_x = max(bounds[2] - bounds[0], 1.0)
    span_y = max(bounds[3] - bounds[1], 1.0)
    margin = max(span_x, span_y) * 0.2 + 1.0
    extended_bounds = (
        bounds[0] - margin,
        bounds[1] - margin,
        bounds[2] + margin,
        bounds[3] + margin,
    )

    resolution = max(4, int(math.sqrt(max(len(edges), 1))))
    cell_size = max(span_x, span_y) / resolution
    cell_size = max(cell_size, 1.0)

    grid = UniformGrid(extended_bounds, cell_size)
    for edge in edges:
        grid.insert(edge.bbox, edge.edge_id)

    return GeometryIndex(
        scene_id=str(scene["scene_id"]),
        epsilon=epsilon,
        bounds=bounds,
        extended_bounds=extended_bounds,
        antennas=antennas,
        vertices=tuple(vertices),
        edges=tuple(edges),
        polygons=tuple(polygons),
        grid=grid,
        angle_epsilon=1.0e-3,
    )


def _query_edge_candidates(
    p0: Point,
    p1: Point,
    geom: GeometryIndex,
) -> set[int]:
    bbox = _bbox_of_points([p0, p1])
    return geom.grid.query_bbox(bbox)


def _polygon_points(poly_id: int, geom: GeometryIndex) -> list[Point]:
    polygon = geom.polygons[poly_id]
    return [geom.vertices[vertex_id].point for vertex_id in polygon.vertex_ids]


def _has_outward_departure(p0: Point, direction: Point, poly_id: int, geom: GeometryIndex) -> bool:
    eps = geom.epsilon
    unit = _normalize(direction, eps)
    if unit is None:
        return False

    span_x = geom.bounds[2] - geom.bounds[0]
    span_y = geom.bounds[3] - geom.bounds[1]
    sample_step = max(max(span_x, span_y) * 1.0e-4, eps * 32.0, 1.0e-3)
    sample = _add(p0, _scale(unit, sample_step))
    polygon = _polygon_points(poly_id, geom)
    return not _point_in_polygon(sample, polygon, eps)


def _point_shadowed_by_edge(
    tx: Point,
    p: Point,
    edge: EdgeRecord,
    geom: GeometryIndex,
) -> bool:
    eps = geom.epsilon
    if _point_on_segment(p, edge.a, edge.b, eps):
        return False
    if _same_point(p, edge.a, eps) or _same_point(p, edge.b, eps):
        return False

    r0 = _sub(edge.a, tx)
    r1 = _sub(edge.b, tx)
    if _length(r0) <= eps or _length(r1) <= eps:
        return False

    a_ray = r0
    b_ray = r1
    if _cross(a_ray, b_ray) < 0.0:
        a_ray, b_ray = b_ray, a_ray

    q = _sub(p, tx)
    if _cross(a_ray, q) < -eps:
        return False
    if _cross(q, b_ray) < -eps:
        return False

    tx_side = _orient(edge.a, edge.b, tx)
    point_side = _orient(edge.a, edge.b, p)
    if tx_side == 0.0:
        return False
    return tx_side * point_side <= eps


def is_visible(tx: Point, p: Point, geom: GeometryIndex) -> bool:
    candidates = _query_edge_candidates(tx, p, geom)
    for edge_id in candidates:
        edge = geom.edges[edge_id]
        if _point_shadowed_by_edge(tx, p, edge, geom):
            return False
    return True


def _shadow_interval_on_edge(
    tx: Point,
    target_edge: EdgeRecord,
    occluder: EdgeRecord,
    geom: GeometryIndex,
) -> tuple[float, float] | None:
    eps = geom.epsilon
    if target_edge.edge_id == occluder.edge_id:
        return None

    a = target_edge.a
    b = target_edge.b
    edge_vec = _sub(b, a)
    r0 = _sub(occluder.a, tx)
    r1 = _sub(occluder.b, tx)

    if _length(r0) <= eps or _length(r1) <= eps:
        return None
    if abs(_cross(r0, r1)) <= eps:
        return None

    if _cross(r0, r1) < 0.0:
        r0, r1 = r1, r0

    interval: tuple[float, float] | None = (0.0, 1.0)
    q0 = _sub(a, tx)

    interval = _clip_interval_geq(interval, _cross(r0, edge_vec), _cross(r0, q0), eps)
    interval = _clip_interval_geq(interval, _cross(edge_vec, r1), _cross(q0, r1), eps)
    if interval is None:
        return None

    tx_side = _orient(occluder.a, occluder.b, tx)
    point_bias = _orient(occluder.a, occluder.b, a)
    point_slope = _orient(occluder.a, occluder.b, b) - point_bias
    interval = _clip_interval_leq(interval, tx_side * point_slope, tx_side * point_bias, eps)
    if interval is None:
        return None

    start, end = interval
    if end - start <= eps:
        return None
    return (max(0.0, start), min(1.0, end))


def compute_visible_subsegments(
    tx: Point,
    edge: EdgeRecord,
    geom: GeometryIndex,
) -> list[tuple[float, float]]:
    candidates = _query_edge_candidates(tx, edge.a, geom)
    candidates.update(_query_edge_candidates(tx, edge.b, geom))
    triangle_bbox = _bbox_of_points([tx, edge.a, edge.b])
    candidates.update(geom.grid.query_bbox(triangle_bbox))

    blocked: list[tuple[float, float]] = []
    for edge_id in candidates:
        occluder = geom.edges[edge_id]
        if occluder.edge_id == edge.edge_id:
            continue
        if not _bbox_intersects(triangle_bbox, occluder.bbox, geom.epsilon):
            continue
        interval = _shadow_interval_on_edge(tx, edge, occluder, geom)
        if interval is not None:
            blocked.append(interval)

    merged_blocked = _merge_intervals(blocked, geom.epsilon)
    visible = _complement_intervals(merged_blocked, geom.epsilon)

    filtered: list[tuple[float, float]] = []
    for start, end in visible:
        mid = (start + end) * 0.5
        if is_visible(tx, _lerp(edge.a, edge.b, mid), geom):
            filtered.append((start, end))

    return _unique_intervals(filtered, geom.epsilon)


def _extend_to_bounds(origin: Point, direction: Point, geom: GeometryIndex) -> Point:
    eps = geom.epsilon
    unit = _normalize(direction, eps)
    if unit is None:
        return origin

    min_x, min_y, max_x, max_y = geom.extended_bounds
    params: list[float] = []

    if abs(unit[0]) > eps:
        params.extend([(min_x - origin[0]) / unit[0], (max_x - origin[0]) / unit[0]])
    if abs(unit[1]) > eps:
        params.extend([(min_y - origin[1]) / unit[1], (max_y - origin[1]) / unit[1]])

    hits: list[Point] = []
    for t in params:
        if t <= eps:
            continue
        p = _add(origin, _scale(unit, t))
        if p[0] < min_x - eps or p[0] > max_x + eps:
            continue
        if p[1] < min_y - eps or p[1] > max_y + eps:
            continue
        hits.append(p)

    if not hits:
        return origin

    hits.sort(key=lambda p: _distance(origin, p))
    return hits[0]


def _trace_to_first_collision(origin: Point, direction: Point, geom: GeometryIndex) -> tuple[Point, int | None]:
    eps = geom.epsilon
    unit = _normalize(direction, eps)
    if unit is None:
        return (origin, None)

    ray_end = _extend_to_bounds(origin, unit, geom)
    best_distance = _distance(origin, ray_end)
    best_edge_id: int | None = None

    candidates = _query_edge_candidates(origin, ray_end, geom)
    for edge_id in candidates:
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

    return (_add(origin, _scale(unit, best_distance)), best_edge_id)


def _first_hit_edge_id(tx: Point, theta: float, geom: GeometryIndex) -> int | None:
    direction = (math.cos(theta), math.sin(theta))
    _, edge_id = _trace_to_first_collision(tx, direction, geom)
    return edge_id


def _is_vertex_critical(tx: Point, vertex: VertexRecord, geom: GeometryIndex) -> bool:
    if not is_visible(tx, vertex.point, geom):
        return False

    theta = math.atan2(vertex.point[1] - tx[1], vertex.point[0] - tx[0])
    delta = geom.angle_epsilon
    left = _first_hit_edge_id(tx, theta - delta, geom)
    right = _first_hit_edge_id(tx, theta + delta, geom)
    return left != right


def _is_reflective_front_face(tx: Point, edge: EdgeRecord, geom: GeometryIndex) -> bool:
    polygon = geom.polygons[edge.poly_id]
    winding = 1.0 if polygon.area >= 0.0 else -1.0
    return winding * _orient(edge.a, edge.b, tx) < -geom.epsilon


def _source_for_edge_point(edge: EdgeRecord, t: float, geom: GeometryIndex) -> dict[str, int | None]:
    source: dict[str, int | None] = {
        "poly_id": edge.poly_id,
        "edge_id": edge.edge_id,
        "vertex_id": None,
    }

    if _almost_equal(t, 0.0, geom.epsilon):
        source["vertex_id"] = edge.start_vertex_id
    elif _almost_equal(t, 1.0, geom.epsilon):
        source["vertex_id"] = edge.end_vertex_id
    return source


def _boundary_type_for_sequence(sequence: str) -> str:
    if sequence == "L":
        return "los"
    if not sequence:
        raise ValueError("sequence must not be empty.")

    kinds = set(sequence)
    if kinds == {"R"}:
        return "reflection"
    if kinds == {"D"}:
        return "diffraction"
    return "mixed"


def _mechanism_for_sequence(sequence: str) -> str:
    if sequence == "L":
        return "los"

    labels = {"R": "reflection", "D": "diffraction"}
    return "+".join(labels[item] for item in sequence)


def _make_boundary_record(
    sequence: str,
    p0: Point,
    p1: Point,
    source: dict[str, int | None],
    scene_id: str,
    tx_id: int,
) -> BoundaryRecord:
    return BoundaryRecord(
        type=_boundary_type_for_sequence(sequence),
        p0=p0,
        p1=p1,
        source=source,
        mechanism=_mechanism_for_sequence(sequence),
        sequence=sequence,
        scene_id=scene_id,
        tx_id=tx_id,
    )


def _extract_reflection_boundaries_from_point(
    source_point: Point,
    geom: GeometryIndex,
    tx_id: int,
    *,
    sequence_prefix: str = "",
    exclude_edge_ids: tuple[int, ...] = (),
) -> list[BoundaryRecord]:
    boundaries: list[BoundaryRecord] = []
    sequence = f"{sequence_prefix}R" if sequence_prefix else "R"

    for edge in geom.edges:
        if edge.edge_id in exclude_edge_ids:
            continue
        if not _is_reflective_front_face(source_point, edge, geom):
            continue

        visible_subsegments = compute_visible_subsegments(source_point, edge, geom)
        if not visible_subsegments:
            continue

        image_source = _reflect_point(source_point, edge.a, edge.b, geom.epsilon)
        for start, end in visible_subsegments:
            for t in (start, end):
                p0 = _lerp(edge.a, edge.b, t)
                direction = _sub(p0, image_source)
                if not _has_outward_departure(p0, direction, edge.poly_id, geom):
                    continue
                p1, _ = _trace_to_first_collision(p0, direction, geom)
                boundaries.append(
                    _make_boundary_record(
                        sequence=sequence,
                        p0=p0,
                        p1=p1,
                        source=_source_for_edge_point(edge, t, geom),
                        scene_id=geom.scene_id,
                        tx_id=tx_id,
                    )
                )

    return boundaries


def _extract_diffraction_boundaries_from_point(
    source_point: Point,
    geom: GeometryIndex,
    tx_id: int,
    *,
    sequence_prefix: str = "",
) -> list[BoundaryRecord]:
    boundaries: list[BoundaryRecord] = []
    sequence = f"{sequence_prefix}D" if sequence_prefix else "D"

    for vertex in geom.vertices:
        if not _is_vertex_critical(source_point, vertex, geom):
            continue

        direction = _sub(vertex.point, source_point)
        if not _has_outward_departure(vertex.point, direction, vertex.poly_id, geom):
            continue
        p1, _ = _trace_to_first_collision(vertex.point, direction, geom)
        boundaries.append(
            _make_boundary_record(
                sequence=sequence,
                p0=vertex.point,
                p1=p1,
                source={
                    "poly_id": vertex.poly_id,
                    "edge_id": None,
                    "vertex_id": vertex.vertex_id,
                },
                scene_id=geom.scene_id,
                tx_id=tx_id,
            )
        )

    return boundaries


def extract_los_boundaries(tx: Point, geom: GeometryIndex, tx_id: int = 0) -> list[BoundaryRecord]:
    boundaries: list[BoundaryRecord] = []
    critical_vertices = {
        vertex.vertex_id
        for vertex in geom.vertices
        if _is_vertex_critical(tx, vertex, geom)
    }

    for edge in geom.edges:
        visible_subsegments = compute_visible_subsegments(tx, edge, geom)
        for start, end in visible_subsegments:
            for t in (start, end):
                source = _source_for_edge_point(edge, t, geom)
                vertex_id = source["vertex_id"]
                if vertex_id is not None and vertex_id not in critical_vertices:
                    continue

                p0 = _lerp(edge.a, edge.b, t)
                direction = _sub(p0, tx)
                if not _has_outward_departure(p0, direction, edge.poly_id, geom):
                    continue
                p1, _ = _trace_to_first_collision(p0, direction, geom)
                boundaries.append(
                    _make_boundary_record(
                        sequence="L",
                        p0=p0,
                        p1=p1,
                        source=source,
                        scene_id=geom.scene_id,
                        tx_id=tx_id,
                    )
                )

    return boundaries


def extract_reflection_boundaries(
    tx: Point,
    geom: GeometryIndex,
    tx_id: int = 0,
) -> list[BoundaryRecord]:
    return _extract_reflection_boundaries_from_point(tx, geom, tx_id)


def extract_diffraction_events(
    tx: Point,
    geom: GeometryIndex,
    tx_id: int = 0,
) -> list[BoundaryRecord]:
    return _extract_diffraction_boundaries_from_point(tx, geom, tx_id)


def _segment_key(boundary: BoundaryRecord, eps: float) -> tuple[Any, ...]:
    scale = max(1.0 / max(eps, 1.0e-9), 1.0e3)

    def q(point: Point) -> tuple[int, int]:
        return (int(round(point[0] * scale)), int(round(point[1] * scale)))

    return (
        boundary.type,
        boundary.sequence,
        q(boundary.p0),
        q(boundary.p1),
        boundary.source.get("poly_id"),
        boundary.source.get("edge_id"),
        boundary.source.get("vertex_id"),
    )


def _geometry_only_key(boundary: BoundaryRecord, eps: float) -> tuple[Any, ...]:
    scale = max(1.0 / max(eps, 1.0e-9), 1.0e3)

    def q(point: Point) -> tuple[int, int]:
        return (int(round(point[0] * scale)), int(round(point[1] * scale)))

    return (q(boundary.p0), q(boundary.p1))


def merge_and_label_boundaries(
    *boundary_groups: list[BoundaryRecord],
    epsilon: float,
) -> list[BoundaryRecord]:
    merged: list[BoundaryRecord] = []
    seen: dict[tuple[Any, ...], BoundaryRecord] = {}

    for group in boundary_groups:
        for boundary in group:
            if _distance(boundary.p0, boundary.p1) <= epsilon:
                continue

            key = _segment_key(boundary, epsilon)
            if key in seen:
                continue
            seen[key] = boundary
            merged.append(boundary)

    return merged


def export_boundaries_json(
    scene_id: str,
    boundaries: list[BoundaryRecord],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = {
        "scene_id": scene_id,
        "boundary_count": len(boundaries),
        "boundaries": [
            {
                "type": boundary.type,
                "p0": [boundary.p0[0], boundary.p0[1]],
                "p1": [boundary.p1[0], boundary.p1[1]],
                "source": boundary.source,
                "mechanism": boundary.mechanism,
                "sequence": boundary.sequence,
                "scene_id": boundary.scene_id,
                "tx_id": boundary.tx_id,
            }
            for boundary in boundaries
        ],
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return payload


def extract_scene_boundaries(
    scene: dict[str, Any] | str | int,
    *,
    tx_ids: list[int] | None = None,
    root_dir: str | None = None,
    max_interactions: int = 1,
    epsilon: float = 1.0e-6,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if max_interactions < 0:
        raise ValueError("max_interactions must be >= 0.")
    if max_interactions > 2:
        raise NotImplementedError("max_interactions > 2 is not implemented yet.")

    if isinstance(scene, dict):
        scene_data = scene
    else:
        scene_data = load_scene(scene, root_dir=root_dir)

    geom = build_geometry(scene_data, epsilon=epsilon)
    indices = tx_ids if tx_ids is not None else list(range(len(scene_data["antenna"])))

    boundaries: list[BoundaryRecord] = []
    for tx_id in indices:
        tx = geom.antennas[tx_id]
        los = extract_los_boundaries(tx, geom, tx_id=tx_id)
        reflection: list[BoundaryRecord] = []
        diffraction: list[BoundaryRecord] = []
        rr: list[BoundaryRecord] = []
        rd: list[BoundaryRecord] = []
        dr: list[BoundaryRecord] = []

        if max_interactions >= 1:
            reflection = extract_reflection_boundaries(tx, geom, tx_id=tx_id)
            diffraction = extract_diffraction_events(tx, geom, tx_id=tx_id)

        if max_interactions >= 2:
            for boundary in reflection:
                edge_id = boundary.source.get("edge_id")
                exclude = (int(edge_id),) if edge_id is not None else ()
                seed = InteractionSeed(
                    point=boundary.p0,
                    sequence=boundary.sequence,
                    source=boundary.source,
                    exclude_edge_ids=exclude,
                )
                rr.extend(
                    _extract_reflection_boundaries_from_point(
                        seed.point,
                        geom,
                        tx_id,
                        sequence_prefix=seed.sequence,
                        exclude_edge_ids=seed.exclude_edge_ids,
                    )
                )
                rd.extend(
                    _extract_diffraction_boundaries_from_point(
                        seed.point,
                        geom,
                        tx_id,
                        sequence_prefix=seed.sequence,
                    )
                )

            for boundary in diffraction:
                seed = InteractionSeed(
                    point=boundary.p0,
                    sequence=boundary.sequence,
                    source=boundary.source,
                )
                dr.extend(
                    _extract_reflection_boundaries_from_point(
                        seed.point,
                        geom,
                        tx_id,
                        sequence_prefix=seed.sequence,
                    )
                )

        boundaries.extend(
            merge_and_label_boundaries(
                los,
                reflection,
                diffraction,
                rr,
                rd,
                dr,
                epsilon=geom.epsilon,
            )
        )

    return export_boundaries_json(geom.scene_id, boundaries, output_path=output_path)
