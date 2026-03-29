from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d
from rt2d.coverage import (
    SequenceCostConfig,
    build_energy_pruned_sequence_render_grid,
    build_layered_sequence_render_grid,
)

GRID_BOUNDS = (0.0, 0.0, 255.0, 255.0)
DEFAULT_SAMPLES = ["0:0", "0:1", "1:0"]
REFERENCE_PERCENTILES = (90.0, 95.0, 97.0)


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    sequences: tuple[str, ...] | None
    energy_pruned: bool = False


CANDIDATE_CONFIGS = (
    CandidateConfig("los_only", ("L",)),
    CandidateConfig("order1_reflection", ("L", "R")),
    CandidateConfig("order1_diffraction", ("L", "D")),
    CandidateConfig("order1_reflection_diffraction", ("L", "R", "D")),
    CandidateConfig("reflection_family_order2", ("L", "R", "RR")),
    CandidateConfig("diffraction_family_order2", ("L", "D", "DD")),
    CandidateConfig("order2_no_rr", ("L", "R", "D", "RD", "DR", "DD")),
    CandidateConfig("order2_no_dd", ("L", "R", "D", "RR", "RD", "DR")),
    CandidateConfig("order2_full", ("L", "R", "D", "RR", "RD", "DR", "DD")),
    CandidateConfig("energy_pruned_order2", None, energy_pruned=True),
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


def _candidate_label_grid(
    tx_result: dict[str, Any],
    scene: dict[str, object],
    tx_id: int,
    candidate: CandidateConfig,
    outdoor_mask: np.ndarray,
    grid_meta: dict[str, Any],
) -> list[list[str]]:
    sequence_hit_grids = tx_result["sequence_hit_grids"]
    outdoor_rows = outdoor_mask.tolist()

    if candidate.energy_pruned:
        label_grid, _counts = build_energy_pruned_sequence_render_grid(
            sequence_hit_grids,
            outdoor_rows,
            (float(scene["antenna"][tx_id][0]), float(scene["antenna"][tx_id][1])),
            grid_meta,
            SequenceCostConfig(),
        )
        return label_grid

    restricted = {
        sequence: sequence_hit_grids[sequence]
        for sequence in candidate.sequences or ()
        if sequence in sequence_hit_grids
    }
    label_grid, _counts = build_layered_sequence_render_grid(restricted, outdoor_rows)
    return label_grid


def _boundary_mask_from_label_grid(label_grid: list[list[str]], outdoor_mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_grid, dtype=object)
    boundary = np.zeros(labels.shape, dtype=bool)

    valid_h = outdoor_mask[:, :-1] & outdoor_mask[:, 1:]
    diff_h = valid_h & (labels[:, :-1] != labels[:, 1:])
    boundary[:, :-1] |= diff_h
    boundary[:, 1:] |= diff_h

    valid_v = outdoor_mask[:-1, :] & outdoor_mask[1:, :]
    diff_v = valid_v & (labels[:-1, :] != labels[1:, :])
    boundary[:-1, :] |= diff_v
    boundary[1:, :] |= diff_v

    return boundary & outdoor_mask


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


def _precision_recall_f1(pred_mask: np.ndarray, ref_mask: np.ndarray) -> tuple[float, float, float]:
    pred_count = int(pred_mask.sum())
    ref_count = int(ref_mask.sum())
    if pred_count == 0 or ref_count == 0:
        return 0.0, 0.0, 0.0

    overlap = int(np.count_nonzero(pred_mask & ref_mask))
    precision = overlap / pred_count
    recall = overlap / ref_count
    if precision + recall <= 0.0:
        return precision, recall, 0.0
    return precision, recall, (2.0 * precision * recall) / (precision + recall)


def _score_candidate_against_gain(
    boundary_mask: np.ndarray,
    gain_strength: np.ndarray,
    scoring_mask: np.ndarray,
) -> dict[str, float]:
    pred = boundary_mask & scoring_mask
    positive_scores = gain_strength[scoring_mask & (gain_strength > 0.0)]
    background_scores = gain_strength[scoring_mask]

    metrics: dict[str, float] = {
        "predicted_pixels": float(pred.sum()),
        "edge_lift": 0.0,
    }
    if pred.any() and background_scores.size > 0 and float(background_scores.mean()) > 0.0:
        metrics["edge_lift"] = float(gain_strength[pred].mean() / background_scores.mean())

    if positive_scores.size == 0:
        for percentile in REFERENCE_PERCENTILES:
            key = f"f1_p{int(percentile)}"
            metrics[key] = 0.0
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


def _overlay_mask(base_gray: np.ndarray, ref_mask: np.ndarray, pred_mask: np.ndarray) -> Image.Image:
    rgb = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.uint8)
    rgb[ref_mask] = np.array([255, 64, 64], dtype=np.uint8)
    rgb[pred_mask] = np.array([64, 224, 208], dtype=np.uint8)
    rgb[ref_mask & pred_mask] = np.array([255, 214, 10], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


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
        "candidate",
        "dpm_f1_mean",
        "irt2_f1_mean",
        "avg_f1_mean",
        "dpm_edge_lift",
        "irt2_edge_lift",
        "avg_edge_lift",
        "predicted_pixels",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RT-derived feature boundary maps against RadioMapSeer gain textures."
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
        default=Path("build") / "radiomapseer_feature_compare",
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
    summary_rows: list[dict[str, object]] = []
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
        payload_outdoor = np.asarray(
            [[label != "blocked" for label in row] for row in tx_result["layered_sequence_grid"]],
            dtype=bool,
        )
        building_mask = _load_gray_image(
            args.radiomapseer_root / "png" / "buildings_complete" / f"{map_id}.png"
        ) > 0
        scoring_mask = payload_outdoor & (~building_mask)

        dpm_image = _load_gray_image(args.radiomapseer_root / "gain" / "DPM" / f"{sample_name}.png")
        irt2_image = _load_gray_image(args.radiomapseer_root / "gain" / "IRT2" / f"{sample_name}.png")
        dpm_strength = _gain_edge_strength(dpm_image, scoring_mask)
        irt2_strength = _gain_edge_strength(irt2_image, scoring_mask)

        per_candidate: list[dict[str, object]] = []
        candidate_masks: dict[str, np.ndarray] = {}
        candidate_ref_masks: dict[str, dict[str, np.ndarray]] = {}

        for candidate in CANDIDATE_CONFIGS:
            label_grid = _candidate_label_grid(
                tx_result,
                scene,
                tx_id,
                candidate,
                payload_outdoor,
                payload["grid"],
            )
            boundary_mask = _boundary_mask_from_label_grid(label_grid, payload_outdoor)
            candidate_masks[candidate.name] = boundary_mask

            dpm_metrics = _score_candidate_against_gain(boundary_mask, dpm_strength, scoring_mask)
            irt2_metrics = _score_candidate_against_gain(boundary_mask, irt2_strength, scoring_mask)

            ref_masks: dict[str, np.ndarray] = {}
            for target_name, strength in (("dpm", dpm_strength), ("irt2", irt2_strength)):
                positive_scores = strength[scoring_mask & (strength > 0.0)]
                if positive_scores.size == 0:
                    ref_masks[target_name] = np.zeros(strength.shape, dtype=bool)
                    continue
                threshold = float(np.percentile(positive_scores, 95.0))
                ref_masks[target_name] = scoring_mask & (strength >= threshold) & (strength > 0.0)
            candidate_ref_masks[candidate.name] = ref_masks

            result = {
                "sample": sample_name,
                "candidate": candidate.name,
                "dpm": dpm_metrics,
                "irt2": irt2_metrics,
                "avg_f1_mean": float((dpm_metrics["f1_mean"] + irt2_metrics["f1_mean"]) * 0.5),
                "avg_edge_lift": float((dpm_metrics["edge_lift"] + irt2_metrics["edge_lift"]) * 0.5),
            }
            per_candidate.append(result)
            summary_rows.append(
                {
                    "sample": sample_name,
                    "candidate": candidate.name,
                    "dpm_f1_mean": f"{dpm_metrics['f1_mean']:.6f}",
                    "irt2_f1_mean": f"{irt2_metrics['f1_mean']:.6f}",
                    "avg_f1_mean": f"{result['avg_f1_mean']:.6f}",
                    "dpm_edge_lift": f"{dpm_metrics['edge_lift']:.6f}",
                    "irt2_edge_lift": f"{irt2_metrics['edge_lift']:.6f}",
                    "avg_edge_lift": f"{result['avg_edge_lift']:.6f}",
                    "predicted_pixels": int(dpm_metrics["predicted_pixels"]),
                }
            )

        ranking = sorted(
            per_candidate,
            key=lambda item: (item["avg_f1_mean"], item["avg_edge_lift"]),
            reverse=True,
        )
        best = ranking[0]
        current = next(item for item in ranking if item["candidate"] == "energy_pruned_order2")

        best_mask = candidate_masks[str(best["candidate"])]
        current_mask = candidate_masks[str(current["candidate"])]
        best_refs = candidate_ref_masks[str(best["candidate"])]

        dpm_panel = [
            ("DPM", Image.fromarray(dpm_image, mode="L").convert("RGB")),
            ("DPM-edge", Image.fromarray(_normalize_to_u8(dpm_strength), mode="L").convert("RGB")),
            ("Best-boundary", Image.fromarray(best_mask.astype(np.uint8) * 255, mode="L").convert("RGB")),
            ("Best-overlay", _overlay_mask(dpm_image, best_refs["dpm"], best_mask)),
            ("Current-overlay", _overlay_mask(dpm_image, best_refs["dpm"], current_mask)),
        ]
        _tile_images_with_titles(dpm_panel, image_dir / f"{sample_name}_dpm_compare.png")

        irt2_panel = [
            ("IRT2", Image.fromarray(irt2_image, mode="L").convert("RGB")),
            ("IRT2-edge", Image.fromarray(_normalize_to_u8(irt2_strength), mode="L").convert("RGB")),
            ("Best-boundary", Image.fromarray(best_mask.astype(np.uint8) * 255, mode="L").convert("RGB")),
            ("Best-overlay", _overlay_mask(irt2_image, best_refs["irt2"], best_mask)),
            ("Current-overlay", _overlay_mask(irt2_image, best_refs["irt2"], current_mask)),
        ]
        _tile_images_with_titles(irt2_panel, image_dir / f"{sample_name}_irt2_compare.png")

        sample_results.append(
            {
                "sample": sample_name,
                "ranking": ranking,
                "best_candidate": best["candidate"],
                "current_candidate": current["candidate"],
                "state_counts": tx_result["state_counts"],
                "counts": tx_result["counts"],
            }
        )

        print(
            f"  best={best['candidate']} avg_f1_mean={best['avg_f1_mean']:.4f} "
            f"avg_edge_lift={best['avg_edge_lift']:.4f}"
        )
        print(
            f"  current={current['candidate']} avg_f1_mean={current['avg_f1_mean']:.4f} "
            f"avg_edge_lift={current['avg_edge_lift']:.4f}"
        )

    result_payload = {
        "samples": sample_results,
        "candidate_order": [candidate.name for candidate in CANDIDATE_CONFIGS],
        "reference_percentiles": list(REFERENCE_PERCENTILES),
        "max_interactions": args.max_interactions,
    }
    summary_json_path = args.output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary_csv(summary_rows, args.output_dir / "summary.csv")

    print(f"summary_json: {summary_json_path}")
    print(f"summary_csv: {args.output_dir / 'summary.csv'}")
    print(f"image_dir: {image_dir}")


if __name__ == "__main__":
    main()
