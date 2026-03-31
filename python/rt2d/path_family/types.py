from __future__ import annotations

from dataclasses import dataclass

from ..coverage import PropagationState


@dataclass(frozen=True)
class ReflectionInteractionRef:
    edge_id: int
    poly_id: int
    subsegment_t0: float
    subsegment_t1: float
    p0: tuple[float, float]
    p1: tuple[float, float]


@dataclass(frozen=True)
class PathFamily:
    family_id: int
    sequence: str
    order: int
    parent_family_id: int | None
    interaction_kind: str
    interaction_ref: ReflectionInteractionRef | None
    state: PropagationState


@dataclass(frozen=True)
class RayHit:
    family_id: int
    sequence: str
    rx_row: int
    rx_col: int
    rx_point: tuple[float, float]
    interaction_types: tuple[str, ...]
    interaction_points: tuple[tuple[float, float], ...]
    path_points: tuple[tuple[float, float], ...]
