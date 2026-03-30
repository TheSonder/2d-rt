from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d
from rt2d.coverage import SequenceCostConfig, build_energy_pruned_sequence_render_grid

GRID_BOUNDS = (0.0, 0.0, 255.0, 255.0)
DEFAULT_SAMPLES = ["0:0", "0:1", "0:2"]
REFERENCE_PERCENTILES = (60.0, 70.0, 80.0, 90.0, 95.0, 97.0)
PARTITION_EDGE_WIDTH_PX = 2

PARTITION_COLORS = {
    "blocked": (17, 24, 39),
    "unreachable": (15, 15, 15),
    "L": (22, 163, 74),
    "I1": (245, 158, 11),
    "I2": (59, 130, 246),
}


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    target: str
    mode: str
    max_order: int = 0
    sequences: tuple[str, ...] = ()


DPM_CANDIDATES = (
    CandidateConfig("minimal_order1", "dpm", "minimal", max_order=1),
    CandidateConfig("layered_reflection", "dpm", "layered", sequences=("L", "R")),
    CandidateConfig("layered_diffraction", "dpm", "layered", sequences=("L", "D")),
    CandidateConfig("layered_full", "dpm", "layered", sequences=("L", "R", "D")),
)

IRT2_CANDIDATES = (
    CandidateConfig("minimal_order2", "irt2", "minimal", max_order=2),
    CandidateConfig("layered_rr", "irt2", "layered", sequences=("RR",)),
    CandidateConfig("layered_dd", "irt2", "layered", sequences=("DD",)),
    CandidateConfig("layered_rd", "irt2", "layered", sequences=("RD",)),
    CandidateConfig("layered_dr", "irt2", "layered", sequences=("DR",)),
    CandidateConfig("layered_order2_no_rr", "irt2", "layered", sequences=("DD", "RD", "DR")),
    CandidateConfig("layered_order2_no_dd", "irt2", "layered", sequences=("RR", "RD", "DR")),
    CandidateConfig("layered_order2_full", "irt2", "layered", sequences=("RR", "DD", "RD", "DR")),
    CandidateConfig("energy_pruned_order2", "irt2", "energy", sequences=("RR", "DD", "RD", "DR")),
)


def _parse_sample(value: str) -> tuple[int, int]:
    raw = value.strip()
    if ":" not in raw:
        raise ValueError(f"invalid sample '{value}', expected '<map_id>:<tx_id>'")
    map_id_raw, tx_id_raw = raw.split(":", 1)
    return int(map_id_raw), int(tx_id_raw)


def _load_scene(radiomapseer_root: Path, map_id: int) -> dict[str, object]:
    antenna_path = radiomapseer_root / "antenna" / f"{map_id}.json"
    polygon_path = radiomapseer_root / "polygon" / "buildings_complete" / f"{map_id}.json"
    return {
        "scene_id": str(map_id),
        "root_dir": str(radiomapseer_root),
        "antenna_path": str(antenna_path),
        "polygon_path": str(polygon_path),
        "antenna": json.loads(antenna_path.read_text(encoding="utf-8")),
        "polygons": json.loads(polygon_path.read_text(encoding="utf-8")),
    }


def _load_gray_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32)
    pil_image = Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    blurred = pil_image.filter(ImageFilter.BoxBlur(radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _gain_edge_strength(gain_image: np.ndarray, outdoor_mask: np.ndarray) -> np.ndarray:
    image = gain_image.astype(np.float32) / 255.0
    strength = np.zeros(image.shape, dtype=np.float32)

    diff_h = np.abs(image[:, 1:] - image[:, :-1])
    valid_h = outdoor_mask[:, 1:] & outdoor_mask[:, :-1]
    diff_h = diff_h * valid_h
    strength[:, :-1] = np.maximum(strength[:, :-1], diff_h)
    strength[:, 1:] = np.maximum(strength[:, 1:], diff_h)

    diff_v = np.abs(image[1:, :] - image[:-1, :])
    valid_v = outdoor_mask[1:, :] & outdoor_mask[:-1, :]
    diff_v = diff_v * valid_v
    strength[:-1, :] = np.maximum(strength[:-1, :], diff_v)
    strength[1:, :] = np.maximum(strength[1:, :], diff_v)

    return strength


def _radial_front_response(
    gain_image: np.ndarray,
    tx_point: tuple[float, float],
    outdoor_mask: np.ndarray,
) -> np.ndarray:
    image = gain_image.astype(np.float32) / 255.0
    height, width = image.shape
    tx_x = float(tx_point[0])
    tx_y = float(height - 1 - tx_point[1])

    ys, xs = np.indices(image.shape, dtype=np.float32)
    dx = xs - tx_x
    dy = ys - tx_y
    norm = np.hypot(dx, dy)
    norm = np.where(norm < 1.0e-6, 1.0, norm)
    ux = dx / norm
    uy = dy / norm

    inner_x = np.clip(np.rint(xs - ux).astype(np.int32), 0, width - 1)
    inner_y = np.clip(np.rint(ys - uy).astype(np.int32), 0, height - 1)
    outer_x = np.clip(np.rint(xs + ux).astype(np.int32), 0, width - 1)
    outer_y = np.clip(np.rint(ys + uy).astype(np.int32), 0, height - 1)

    response = np.abs(image[outer_y, outer_x] - image[inner_y, inner_x])
    response *= outdoor_mask
    return response.astype(np.float32)


def _normalize_response(response: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = response[mask]
    if values.size == 0:
        return np.zeros(response.shape, dtype=np.float32)
    high = float(np.percentile(values, 97.0))
    if high <= 1.0e-9:
        return np.zeros(response.shape, dtype=np.float32)
    return np.clip(response / high, 0.0, 1.0).astype(np.float32)


def _texture_response(
    gain_image: np.ndarray,
    outdoor_mask: np.ndarray,
    tx_point: tuple[float, float],
) -> np.ndarray:
    image = gain_image.astype(np.float32) / 255.0
    edge = _gain_edge_strength(gain_image, outdoor_mask)
    blur_1 = _box_blur(image, radius=1)
    blur_3 = _box_blur(image, radius=3)
    blur_6 = _box_blur(image, radius=6)
    blur_12 = _box_blur(image, radius=12)
    front_12 = _gain_edge_strength(
        np.clip(blur_12 * 255.0, 0.0, 255.0).astype(np.uint8),
        outdoor_mask,
    )
    detail_1 = np.abs(image - blur_1)
    detail_3 = np.abs(image - blur_3)
    detail_6 = np.abs(image - blur_6)
    detail_12 = np.abs(blur_3 - blur_12)
    radial = _radial_front_response(gain_image, tx_point, outdoor_mask)

    combined = (
        0.18 * edge
        + 0.16 * front_12
        + 0.16 * radial
        + 0.16 * detail_1
        + 0.14 * detail_3
        + 0.10 * detail_6
        + 0.10 * detail_12
    )
    combined *= outdoor_mask
    return _normalize_response(combined, outdoor_mask)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros(mask.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            view = padded[
                radius + dy : radius + dy + mask.shape[0],
                radius + dx : radius + dx + mask.shape[1],
            ]
            result |= view
    return result


def _precision_recall_f1(pred_mask: np.ndarray, ref_mask: np.ndarray) -> tuple[float, float, float]:
    pred_count = int(pred_mask.sum())
    ref_count = int(ref_mask.sum())
    if pred_count == 0 or ref_count == 0:
        return 0.0, 0.0, 0.0

    ref_dilated = _dilate_mask(ref_mask, radius=1)
    pred_dilated = _dilate_mask(pred_mask, radius=1)
    precision_hits = int(np.count_nonzero(pred_mask & ref_dilated))
    recall_hits = int(np.count_nonzero(ref_mask & pred_dilated))
    precision = precision_hits / pred_count
    recall = recall_hits / ref_count
    if precision + recall <= 0.0:
        return precision, recall, 0.0
    return precision, recall, (2.0 * precision * recall) / (precision + recall)


def _score_partition_against_gain(
    boundary_mask: np.ndarray,
    gain_strength: np.ndarray,
    scoring_mask: np.ndarray,
) -> dict[str, float]:
    pred = boundary_mask & scoring_mask
    positive_scores = gain_strength[scoring_mask & (gain_strength > 0.0)]
    background_scores = gain_strength[scoring_mask]
    total_energy = float(positive_scores.sum()) if positive_scores.size > 0 else 0.0

    metrics: dict[str, float] = {
        "predicted_pixels": float(pred.sum()),
        "edge_lift": 0.0,
        "energy_capture": 0.0,
    }
    if pred.any() and background_scores.size > 0 and float(background_scores.mean()) > 0.0:
        metrics["edge_lift"] = float(gain_strength[pred].mean() / background_scores.mean())
    if pred.any() and total_energy > 0.0:
        metrics["energy_capture"] = float(gain_strength[pred].sum() / total_energy)

    if positive_scores.size == 0:
        for percentile in REFERENCE_PERCENTILES:
            metrics[f"f1_p{int(percentile)}"] = 0.0
            metrics[f"precision_p{int(percentile)}"] = 0.0
            metrics[f"recall_p{int(percentile)}"] = 0.0
        metrics["f1_mean"] = 0.0
        return metrics

    f1_values: list[float] = []
    for percentile in REFERENCE_PERCENTILES:
        threshold = float(np.percentile(positive_scores, percentile))
        ref = scoring_mask & (gain_strength >= threshold) & (gain_strength > 0.0)
        precision, recall, f1 = _precision_recall_f1(pred, ref)
        metrics[f"precision_p{int(percentile)}"] = precision
        metrics[f"recall_p{int(percentile)}"] = recall
        metrics[f"f1_p{int(percentile)}"] = f1
        f1_values.append(f1)

    metrics["f1_mean"] = float(sum(f1_values) / len(f1_values))
    return metrics


def _normalize_to_u8(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image.astype(np.uint8)
    max_value = float(image.max())
    if max_value <= 0.0:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image / max_value) * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def _reference_mask(
    gain_strength: np.ndarray,
    scoring_mask: np.ndarray,
    percentile: float = 85.0,
) -> np.ndarray:
    positive_scores = gain_strength[scoring_mask & (gain_strength > 0.0)]
    if positive_scores.size == 0:
        return np.zeros(gain_strength.shape, dtype=bool)
    threshold = float(np.percentile(positive_scores, percentile))
    return scoring_mask & (gain_strength >= threshold) & (gain_strength > 0.0)


def _overlay_mask(base_gray: np.ndarray, ref_mask: np.ndarray, pred_mask: np.ndarray) -> Image.Image:
    rgb = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.uint8)
    rgb[ref_mask] = np.array([255, 64, 64], dtype=np.uint8)
    rgb[pred_mask] = np.array([64, 224, 208], dtype=np.uint8)
    rgb[ref_mask & pred_mask] = np.array([255, 214, 10], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _grouped_grid_to_image(grouped_grid: list[list[str]]) -> Image.Image:
    height = len(grouped_grid)
    width = len(grouped_grid[0]) if height else 0
    image = Image.new("RGB", (width, height), PARTITION_COLORS["blocked"])
    pixels = image.load()
    for row in range(height):
        for col in range(width):
            pixels[col, row] = PARTITION_COLORS[grouped_grid[row][col]]
    return image


def _tile_images_with_titles(
    tiles: list[tuple[str, Image.Image]],
    output_path: Path,
) -> None:
    if not tiles:
        return

    label_height = 22
    width = tiles[0][1].width
    height = tiles[0][1].height + label_height
    canvas = Image.new("RGB", (width * len(tiles), height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    for index, (title, image) in enumerate(tiles):
        x = index * width
        canvas.paste(image.convert("RGB"), (x, label_height))
        draw.text((x + 6, 4), title, fill=(24, 24, 27))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "target",
        "candidate",
        "mode",
        "f1_mean",
        "edge_lift",
        "energy_capture",
        "predicted_pixels",
    ]
    for percentile in REFERENCE_PERCENTILES:
        fieldnames.append(f"f1_p{int(percentile)}")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_from_minimal_grid(
    visibility_order_grid: list[list[int]],
    target: str,
    max_order: int,
) -> list[list[str]]:
    grouped: list[list[str]] = []
    for row in visibility_order_grid:
        grouped_row: list[str] = []
        for value in row:
            label = int(value)
            if label == -2:
                grouped_row.append("blocked")
            elif label == -1:
                grouped_row.append("unreachable")
            elif label == 0:
                grouped_row.append("L")
            elif label == 1 and max_order >= 1:
                grouped_row.append("I1")
            elif label == 2 and target == "irt2" and max_order >= 2:
                grouped_row.append("I2")
            else:
                grouped_row.append("unreachable")
        grouped.append(grouped_row)
    return grouped


def _group_from_layered_grid(
    layered_grid: list[list[str]],
    target: str,
    included_sequences: set[str],
) -> list[list[str]]:
    grouped: list[list[str]] = []
    for row in layered_grid:
        grouped_row: list[str] = []
        for label in row:
            raw = str(label)
            if raw in {"blocked", "unreachable"}:
                grouped_row.append(raw)
                continue
            if raw == "L":
                grouped_row.append("L")
                continue
            if len(raw) == 1 and set(raw) <= {"R", "D"}:
                if target == "dpm" and raw in included_sequences:
                    grouped_row.append("I1")
                elif target == "irt2":
                    grouped_row.append("I1")
                else:
                    grouped_row.append("unreachable")
                continue
            if len(raw) == 2 and set(raw) <= {"R", "D"}:
                if target == "irt2" and raw in included_sequences:
                    grouped_row.append("I2")
                else:
                    grouped_row.append("unreachable")
                continue
            grouped_row.append("unreachable")
        grouped.append(grouped_row)
    return grouped


def _group_from_energy_grid(
    tx_result: dict[str, Any],
    scene: dict[str, object],
    tx_id: int,
    outdoor_mask: np.ndarray,
    grid_meta: dict[str, Any],
    included_sequences: set[str],
) -> list[list[str]]:
    tx_point = (
        float(scene["antenna"][tx_id][0]),
        float(scene["antenna"][tx_id][1]),
    )
    pruned_grid, _counts = build_energy_pruned_sequence_render_grid(
        tx_result["sequence_hit_grids"],
        outdoor_mask.tolist(),
        tx_point,
        grid_meta,
        SequenceCostConfig(),
    )
    return _group_from_layered_grid(pruned_grid, "irt2", included_sequences)


def _candidate_partition_grid(
    candidate: CandidateConfig,
    tx_result: dict[str, Any],
    scene: dict[str, object],
    tx_id: int,
    outdoor_mask: np.ndarray,
    grid_meta: dict[str, Any],
) -> list[list[str]]:
    if candidate.mode == "minimal":
        return _group_from_minimal_grid(
            tx_result["visibility_order_grid"],
            candidate.target,
            candidate.max_order,
        )
    if candidate.mode == "layered":
        return _group_from_layered_grid(
            tx_result["layered_sequence_grid"],
            candidate.target,
            set(candidate.sequences),
        )
    if candidate.mode == "energy":
        return _group_from_energy_grid(
            tx_result,
            scene,
            tx_id,
            outdoor_mask,
            grid_meta,
            set(candidate.sequences),
        )
    raise ValueError(f"unsupported mode: {candidate.mode}")


def _boundary_segments_from_label_grid(
    label_grid: list[list[str]],
    outdoor_mask: np.ndarray,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    labels = np.asarray(label_grid, dtype=object)
    height, width = labels.shape
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []

    valid_h = outdoor_mask[:, :-1] & outdoor_mask[:, 1:]
    diff_h = valid_h & (labels[:, :-1] != labels[:, 1:])
    for col in range(width - 1):
        row = 0
        while row < height:
            if not diff_h[row, col]:
                row += 1
                continue
            start_row = row
            while row + 1 < height and diff_h[row + 1, col]:
                row += 1
            end_row = row
            x = col + 1
            y0 = start_row
            y1 = min(end_row + 1, height - 1)
            segments.append(((x, y0), (x, y1)))
            row += 1

    valid_v = outdoor_mask[:-1, :] & outdoor_mask[1:, :]
    diff_v = valid_v & (labels[:-1, :] != labels[1:, :])
    for row in range(height - 1):
        col = 0
        while col < width:
            if not diff_v[row, col]:
                col += 1
                continue
            start_col = col
            while col + 1 < width and diff_v[row, col + 1]:
                col += 1
            end_col = col
            y = row + 1
            x0 = start_col
            x1 = min(end_col + 1, width - 1)
            segments.append(((x0, y), (x1, y)))
            col += 1

    return segments


def _rasterize_boundary_segments(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for (x0, y0), (x1, y1) in segments:
        draw.line(
            (
                max(0, min(width - 1, x0)),
                max(0, min(height - 1, y0)),
                max(0, min(width - 1, x1)),
                max(0, min(height - 1, y1)),
            ),
            fill=255,
            width=PARTITION_EDGE_WIDTH_PX,
        )
    return np.asarray(image, dtype=np.uint8) > 0


def _boundary_mask_from_label_grid(label_grid: list[list[str]], outdoor_mask: np.ndarray) -> np.ndarray:
    segments = _boundary_segments_from_label_grid(label_grid, outdoor_mask)
    return _rasterize_boundary_segments(segments, outdoor_mask.shape) & outdoor_mask


def _evaluate_candidates(
    candidates: tuple[CandidateConfig, ...],
    tx_result: dict[str, Any],
    scene: dict[str, object],
    tx_id: int,
    outdoor_mask: np.ndarray,
    grid_meta: dict[str, Any],
    gain_strength: np.ndarray,
    scoring_mask: np.ndarray,
    sample_name: str,
) -> tuple[list[dict[str, object]], dict[str, list[list[str]]], dict[str, np.ndarray]]:
    ranking: list[dict[str, object]] = []
    grouped_grids: dict[str, list[list[str]]] = {}
    boundary_masks: dict[str, np.ndarray] = {}

    for candidate in candidates:
        grouped_grid = _candidate_partition_grid(
            candidate,
            tx_result,
            scene,
            tx_id,
            outdoor_mask,
            grid_meta,
        )
        boundary_mask = _boundary_mask_from_label_grid(grouped_grid, outdoor_mask)
        grouped_grids[candidate.name] = grouped_grid
        boundary_masks[candidate.name] = boundary_mask
        metrics = _score_partition_against_gain(boundary_mask, gain_strength, scoring_mask)
        ranking.append(
            {
                "sample": sample_name,
                "target": candidate.target,
                "candidate": candidate.name,
                "mode": candidate.mode,
                "metrics": metrics,
            }
        )

    ranking.sort(
        key=lambda item: (
            item["metrics"]["f1_mean"],
            item["metrics"]["energy_capture"],
            item["metrics"]["edge_lift"],
        ),
        reverse=True,
    )
    return ranking, grouped_grids, boundary_masks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RT partition renders against RadioMapSeer DPM / IRT2 gain maps."
    )
    parser.add_argument(
        "--radiomapseer-root",
        type=Path,
        default=Path(r"D:\TheSonder\0.8 Resources\data\RadioMapSeer"),
        help="RadioMapSeer root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build") / "radiomapseer_partition_compare",
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=DEFAULT_SAMPLES,
        help="Sample list formatted as '<map_id>:<tx_id>'.",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=2,
        help="Maximum interaction order for RT visibility expansion.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached RX visibility payloads and recompute them.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    image_dir = args.output_dir / "images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    sample_pairs = [_parse_sample(value) for value in args.samples]
    dpm_rows: list[dict[str, object]] = []
    irt2_rows: list[dict[str, object]] = []
    sample_results: list[dict[str, object]] = []

    for map_id, tx_id in sample_pairs:
        sample_name = f"{map_id}_{tx_id}"
        print(f"[sample] {sample_name}")
        scene = _load_scene(args.radiomapseer_root, map_id)
        cache_path = cache_dir / f"{sample_name}_order{args.max_interactions}.json"

        if args.force_recompute or not cache_path.is_file():
            payload = rt2d.compute_rx_visibility(
                scene,
                tx_ids=[tx_id],
                max_interactions=args.max_interactions,
                bounds=GRID_BOUNDS,
                include_sequence_render_grid=True,
                include_sequence_hit_grids=True,
                output_path=cache_path,
            )
        else:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))

        tx_result = payload["tx_results"][0]
        outdoor_mask = np.asarray(
            [[label != "blocked" for label in row] for row in tx_result["layered_sequence_grid"]],
            dtype=bool,
        )
        building_mask = _load_gray_image(
            args.radiomapseer_root / "png" / "buildings_complete" / f"{map_id}.png"
        ) > 0
        scoring_mask = outdoor_mask & (~building_mask)
        tx_point = (
            float(scene["antenna"][tx_id][0]),
            float(scene["antenna"][tx_id][1]),
        )

        dpm_image = _load_gray_image(args.radiomapseer_root / "gain" / "DPM" / f"{sample_name}.png")
        irt2_image = _load_gray_image(args.radiomapseer_root / "gain" / "IRT2" / f"{sample_name}.png")
        dpm_strength = _texture_response(dpm_image, scoring_mask, tx_point)
        irt2_strength = _texture_response(irt2_image, scoring_mask, tx_point)

        dpm_ranking, dpm_grouped, dpm_boundaries = _evaluate_candidates(
            DPM_CANDIDATES,
            tx_result,
            scene,
            tx_id,
            outdoor_mask,
            payload["grid"],
            dpm_strength,
            scoring_mask,
            sample_name,
        )
        irt2_ranking, irt2_grouped, irt2_boundaries = _evaluate_candidates(
            IRT2_CANDIDATES,
            tx_result,
            scene,
            tx_id,
            outdoor_mask,
            payload["grid"],
            irt2_strength,
            scoring_mask,
            sample_name,
        )

        for item in dpm_ranking:
            row = {
                "sample": sample_name,
                "target": "dpm",
                "candidate": item["candidate"],
                "mode": item["mode"],
                "f1_mean": f"{item['metrics']['f1_mean']:.6f}",
                "edge_lift": f"{item['metrics']['edge_lift']:.6f}",
                "energy_capture": f"{item['metrics']['energy_capture']:.6f}",
                "predicted_pixels": int(item["metrics"]["predicted_pixels"]),
            }
            for percentile in REFERENCE_PERCENTILES:
                row[f"f1_p{int(percentile)}"] = f"{item['metrics'][f'f1_p{int(percentile)}']:.6f}"
            dpm_rows.append(row)

        for item in irt2_ranking:
            row = {
                "sample": sample_name,
                "target": "irt2",
                "candidate": item["candidate"],
                "mode": item["mode"],
                "f1_mean": f"{item['metrics']['f1_mean']:.6f}",
                "edge_lift": f"{item['metrics']['edge_lift']:.6f}",
                "energy_capture": f"{item['metrics']['energy_capture']:.6f}",
                "predicted_pixels": int(item["metrics"]["predicted_pixels"]),
            }
            for percentile in REFERENCE_PERCENTILES:
                row[f"f1_p{int(percentile)}"] = f"{item['metrics'][f'f1_p{int(percentile)}']:.6f}"
            irt2_rows.append(row)

        dpm_best = dpm_ranking[0]
        irt2_best = irt2_ranking[0]
        irt2_current = next(item for item in irt2_ranking if item["candidate"] == "energy_pruned_order2")
        dpm_ref = _reference_mask(dpm_strength, scoring_mask, percentile=85.0)
        irt2_ref = _reference_mask(irt2_strength, scoring_mask, percentile=85.0)

        dpm_panel = [
            ("DPM", Image.fromarray(dpm_image, mode="L").convert("RGB")),
            ("DPM-texture", Image.fromarray(_normalize_to_u8(dpm_strength), mode="L").convert("RGB")),
            ("Best-partition", _grouped_grid_to_image(dpm_grouped[str(dpm_best["candidate"])])),
            ("Best-overlay", _overlay_mask(dpm_image, dpm_ref, dpm_boundaries[str(dpm_best["candidate"])])),
        ]
        _tile_images_with_titles(dpm_panel, image_dir / f"{sample_name}_dpm_compare.png")

        irt2_panel = [
            ("IRT2", Image.fromarray(irt2_image, mode="L").convert("RGB")),
            ("IRT2-texture", Image.fromarray(_normalize_to_u8(irt2_strength), mode="L").convert("RGB")),
            ("Best-partition", _grouped_grid_to_image(irt2_grouped[str(irt2_best["candidate"])])),
            ("Best-overlay", _overlay_mask(irt2_image, irt2_ref, irt2_boundaries[str(irt2_best["candidate"])])),
            ("Energy-overlay", _overlay_mask(irt2_image, irt2_ref, irt2_boundaries[str(irt2_current["candidate"])])),
        ]
        _tile_images_with_titles(irt2_panel, image_dir / f"{sample_name}_irt2_compare.png")

        sample_results.append(
            {
                "sample": sample_name,
                "dpm_ranking": dpm_ranking,
                "irt2_ranking": irt2_ranking,
                "dpm_best_candidate": dpm_best["candidate"],
                "irt2_best_candidate": irt2_best["candidate"],
                "irt2_current_candidate": irt2_current["candidate"],
                "state_counts": tx_result["state_counts"],
                "counts": tx_result["counts"],
            }
        )

        print(
            f"  dpm_best={dpm_best['candidate']} f1_mean={dpm_best['metrics']['f1_mean']:.4f} "
            f"energy_capture={dpm_best['metrics']['energy_capture']:.4f}"
        )
        print(
            f"  irt2_best={irt2_best['candidate']} f1_mean={irt2_best['metrics']['f1_mean']:.4f} "
            f"energy_capture={irt2_best['metrics']['energy_capture']:.4f}"
        )
        print(
            f"  irt2_current={irt2_current['candidate']} f1_mean={irt2_current['metrics']['f1_mean']:.4f} "
            f"energy_capture={irt2_current['metrics']['energy_capture']:.4f}"
        )

    summary_json_path = args.output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "samples": sample_results,
                "dpm_candidate_order": [candidate.name for candidate in DPM_CANDIDATES],
                "irt2_candidate_order": [candidate.name for candidate in IRT2_CANDIDATES],
                "reference_percentiles": list(REFERENCE_PERCENTILES),
                "max_interactions": args.max_interactions,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dpm_summary_csv = args.output_dir / "summary_dpm.csv"
    irt2_summary_csv = args.output_dir / "summary_irt2.csv"
    _write_summary_csv(dpm_rows, dpm_summary_csv)
    _write_summary_csv(irt2_rows, irt2_summary_csv)

    print(f"summary_json: {summary_json_path}")
    print(f"summary_dpm_csv: {dpm_summary_csv}")
    print(f"summary_irt2_csv: {irt2_summary_csv}")
    print(f"image_dir: {image_dir}")


if __name__ == "__main__":
    main()
