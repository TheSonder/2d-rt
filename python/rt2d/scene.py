from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

_SCENE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _normalize_scene_id(scene_id: int | str) -> str:
    value = str(scene_id).strip()
    if not value:
        raise ValueError("scene_id must not be empty.")
    if not _SCENE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "scene_id contains invalid characters. Allowed: letters, digits, '_' and '-'."
        )
    return value


def _default_environment_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "environment"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc


def _as_float_pair(value: Any, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{context} must be [x, y]. Got: {value!r}")

    x_raw, y_raw = value
    if not isinstance(x_raw, (int, float)) or not isinstance(y_raw, (int, float)):
        raise ValueError(f"{context} coordinates must be numeric. Got: {value!r}")

    x = float(x_raw)
    y = float(y_raw)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{context} coordinates must be finite. Got: {value!r}")

    return [x, y]


def _validate_antenna(data: Any, path: Path) -> list[list[float]]:
    if not isinstance(data, list):
        raise ValueError(f"'{path}' must be a JSON array of [x, y] points.")

    points: list[list[float]] = []
    for i, point in enumerate(data):
        points.append(_as_float_pair(point, f"antenna[{i}]"))

    return points


def _validate_polygons(data: Any, path: Path) -> list[list[list[float]]]:
    if not isinstance(data, list):
        raise ValueError(f"'{path}' must be a JSON array of polygons.")

    polygons: list[list[list[float]]] = []
    for poly_idx, poly in enumerate(data):
        if not isinstance(poly, list):
            raise ValueError(f"polygon[{poly_idx}] must be an array of [x, y] points.")
        if len(poly) < 3:
            raise ValueError(
                f"polygon[{poly_idx}] must contain at least 3 points, got {len(poly)}."
            )

        points: list[list[float]] = []
        for pt_idx, pt in enumerate(poly):
            points.append(_as_float_pair(pt, f"polygon[{poly_idx}][{pt_idx}]"))
        polygons.append(points)

    return polygons


def load_scene(
    scene_id: int | str,
    root_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    scene_name = _normalize_scene_id(scene_id)
    root = Path(root_dir) if root_dir is not None else _default_environment_root()

    antenna_path = root / "antenna" / f"{scene_name}.json"
    polygon_path = root / "polygon" / f"{scene_name}.json"

    if not antenna_path.is_file():
        raise FileNotFoundError(f"Antenna file not found: '{antenna_path}'")
    if not polygon_path.is_file():
        raise FileNotFoundError(f"Polygon file not found: '{polygon_path}'")

    antenna = _validate_antenna(_read_json(antenna_path), antenna_path)
    polygons = _validate_polygons(_read_json(polygon_path), polygon_path)

    return {
        "scene_id": scene_name,
        "root_dir": str(root.resolve()),
        "antenna_path": str(antenna_path.resolve()),
        "polygon_path": str(polygon_path.resolve()),
        "antenna": antenna,
        "polygons": polygons,
    }

