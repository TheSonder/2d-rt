from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..boundary import (
    GeometryIndex,
    _compute_visible_subsegments_for_state,
    _is_reflective_front_face,
    _lerp,
)
from ..coverage import (
    PropagationState,
    _get_or_build_state_expansion,
    _make_state,
    build_rx_visibility_runtime,
)
from .types import DiffractionInteractionRef, PathFamily, ReflectionInteractionRef


@dataclass
class PathFamilyRuntime:
    scene_id: str
    tx_id: int
    tx_point: tuple[float, float]
    rx_runtime: Any
    geometry: GeometryIndex
    max_interactions: int
    los_family: PathFamily | None = None
    reflection_families: tuple[PathFamily, ...] = ()
    diffraction_families: tuple[PathFamily, ...] = ()
    second_order_families: tuple[PathFamily, ...] = ()


def _build_reflection_interaction_refs(
    tx: tuple[float, float],
    geom: GeometryIndex,
) -> list[ReflectionInteractionRef]:
    root_state = _make_state("", tx, None)
    refs: list[ReflectionInteractionRef] = []

    for edge in geom.edges:
        if not _is_reflective_front_face(tx, edge, geom):
            continue
        visible_subsegments = _compute_visible_subsegments_for_state(root_state, edge, geom)
        if not visible_subsegments:
            continue
        for start, end in visible_subsegments:
            refs.append(
                ReflectionInteractionRef(
                    edge_id=edge.edge_id,
                    poly_id=edge.poly_id,
                    subsegment_t0=float(start),
                    subsegment_t1=float(end),
                    p0=_lerp(edge.a, edge.b, start),
                    p1=_lerp(edge.a, edge.b, end),
                )
            )
    return refs


def build_path_family_runtime(
    scene: dict[str, Any] | str | int,
    *,
    root_dir: str | None = None,
    tx_id: int = 0,
    max_interactions: int = 2,
    grid_step: float = 1.0,
    bounds: tuple[float, float, float, float] | None = None,
    epsilon: float = 1.0e-6,
    acceleration_backend: str = "cpu",
    torch_device: str | None = None,
) -> PathFamilyRuntime:
    rx_runtime = build_rx_visibility_runtime(
        scene,
        root_dir=root_dir,
        grid_step=grid_step,
        bounds=bounds,
        epsilon=epsilon,
        acceleration_backend=acceleration_backend,
        torch_device=torch_device,
    )
    geom = rx_runtime.geom
    if tx_id < 0 or tx_id >= len(geom.antennas):
        raise ValueError(f"tx_id out of range: {tx_id}")

    tx = geom.antennas[tx_id]
    los_state = _make_state("L", tx, None)
    los_family = PathFamily(
        family_id=0,
        sequence="L",
        order=0,
        parent_family_id=None,
        interaction_kind="los",
        interaction_ref=None,
        state=los_state,
    )

    states_by_order, _sequence_groups_by_order = _get_or_build_state_expansion(
        rx_runtime,
        tx_id,
        max_interactions,
        enable_reflection=True,
        enable_diffraction=True,
    )
    reflection_states = [state for state in states_by_order.get(1, []) if state.sequence == "R"]
    diffraction_states = [state for state in states_by_order.get(1, []) if state.sequence == "D"]
    interaction_refs = _build_reflection_interaction_refs(tx, geom)

    reflection_families: list[PathFamily] = []
    for family_id, (state, interaction_ref) in enumerate(
        zip(reflection_states, interaction_refs),
        start=1,
    ):
        reflection_families.append(
            PathFamily(
                family_id=family_id,
                sequence="R",
                order=1,
                parent_family_id=0,
                interaction_kind="reflection",
                interaction_ref=interaction_ref,
                state=state,
            )
        )

    next_family_id = len(reflection_families) + 1
    diffraction_families: list[PathFamily] = []
    for offset, state in enumerate(diffraction_states):
        if state.source_vertex_id is None or state.source_poly_id is None:
            continue
        diffraction_families.append(
            PathFamily(
                family_id=next_family_id + offset,
                sequence="D",
                order=1,
                parent_family_id=0,
                interaction_kind="diffraction",
                interaction_ref=DiffractionInteractionRef(
                    vertex_id=state.source_vertex_id,
                    poly_id=state.source_poly_id,
                    point=state.source_point,
                ),
                state=state,
            )
        )

    next_family_id = len(reflection_families) + len(diffraction_families) + 1
    second_order_families: list[PathFamily] = []
    for offset, state in enumerate(states_by_order.get(2, [])):
        if state.sequence not in {"RR", "RD", "DR", "DD"}:
            continue
        second_order_families.append(
            PathFamily(
                family_id=next_family_id + offset,
                sequence=state.sequence,
                order=2,
                parent_family_id=None,
                interaction_kind=state.interaction_kind,
                interaction_ref=None,
                state=state,
            )
        )

    return PathFamilyRuntime(
        scene_id=str(geom.scene_id),
        tx_id=tx_id,
        tx_point=tx,
        rx_runtime=rx_runtime,
        geometry=geom,
        max_interactions=max_interactions,
        los_family=los_family,
        reflection_families=tuple(reflection_families),
        diffraction_families=tuple(diffraction_families),
        second_order_families=tuple(second_order_families),
    )
