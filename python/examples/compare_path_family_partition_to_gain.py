from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d
import examples.compare_radiomapseer_feature_maps as compare

PATH_FAMILY_COLORS = {
    "blocked": (17, 24, 39),
    "unreachable": (15, 15, 15),
    "L": (22, 163, 74),
    "R": (245, 158, 11),
    "D": (59, 130, 246),
    "RR": (168, 85, 247),
    "RD": (236, 72, 153),
    "DR": (14, 165, 233),
    "DD": (99, 102, 241),
}


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


def _partition_boundary_mask(partition_grid: list[list[str]]) -> np.ndarray:
    outdoor_mask = np.asarray([[label != "blocked" for label in row] for row in partition_grid], dtype=bool)
    segments = compare._boundary_segments_from_label_grid(partition_grid, outdoor_mask)
    return compare._rasterize_boundary_segments(segments, outdoor_mask.shape) & outdoor_mask


def _partition_grid_to_image(partition_grid: list[list[str]]) -> Image.Image:
    height = len(partition_grid)
    width = len(partition_grid[0]) if height else 0
    image = Image.new("RGB", (width, height), PATH_FAMILY_COLORS["blocked"])
    pixels = image.load()
    for row in range(height):
        for col in range(width):
            pixels[col, row] = PATH_FAMILY_COLORS[str(partition_grid[row][col])]
    return image


def _report_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name}: "
        f"f1_mean={metrics['f1_mean']:.4f} "
        f"edge_lift={metrics['edge_lift']:.4f} "
        f"energy_capture={metrics['energy_capture']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare path-family first/second-order partition against RadioMapSeer DPM / IRT2."
    )
    parser.add_argument("map_id", type=int, help="RadioMapSeer map id.")
    parser.add_argument("--tx-id", type=int, default=0, help="TX index.")
    parser.add_argument(
        "--radiomapseer-root",
        type=Path,
        default=Path(r"D:\TheSonder\0.8 Resources\data\RadioMapSeer"),
        help="RadioMapSeer root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build") / "path_family_gain_compare",
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--acceleration-backend",
        choices=("cpu", "auto", "torch"),
        default="cpu",
        help="Backend used by the underlying path-family runtime.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Optional torch device, for example 'cuda' or 'cpu'.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = _load_scene(args.radiomapseer_root, args.map_id)
    tx_point = (
        float(scene["antenna"][args.tx_id][0]),
        float(scene["antenna"][args.tx_id][1]),
    )

    runtime = rt2d.build_path_family_runtime(
        scene,
        tx_id=args.tx_id,
        max_interactions=2,
        bounds=(0.0, 0.0, 255.0, 255.0),
        acceleration_backend=args.acceleration_backend,
        torch_device=args.torch_device,
    )

    dpm_partition = rt2d.compute_path_family_partition_runtime(runtime, max_order=1)
    irt2_partition = rt2d.compute_path_family_partition_runtime(runtime, max_order=2)

    building_mask = np.array(
        Image.open(args.radiomapseer_root / "png" / "buildings_complete" / f"{args.map_id}.png").convert("L")
    ) > 0
    scoring_mask = ~building_mask
    sample_name = f"{args.map_id}_{args.tx_id}"
    dpm_image = compare._load_gray_image(args.radiomapseer_root / "gain" / "DPM" / f"{sample_name}.png")
    irt2_image = compare._load_gray_image(args.radiomapseer_root / "gain" / "IRT2" / f"{sample_name}.png")
    dpm_strength = compare._texture_response(dpm_image, scoring_mask, tx_point)
    irt2_strength = compare._texture_response(irt2_image, scoring_mask, tx_point)

    dpm_boundary = _partition_boundary_mask(dpm_partition["partition_grid"])
    irt2_boundary = _partition_boundary_mask(irt2_partition["partition_grid"])

    dpm_metrics = compare._score_partition_against_gain(dpm_boundary, dpm_strength, scoring_mask)
    irt2_metrics = compare._score_partition_against_gain(irt2_boundary, irt2_strength, scoring_mask)

    dpm_ref = compare._reference_mask(dpm_strength, scoring_mask, percentile=85.0)
    irt2_ref = compare._reference_mask(irt2_strength, scoring_mask, percentile=85.0)

    dpm_panel = [
        ("DPM", Image.fromarray(dpm_image, mode="L").convert("RGB")),
        ("DPM-texture", Image.fromarray(compare._normalize_to_u8(dpm_strength), mode="L").convert("RGB")),
        ("PathFamily-order1", _partition_grid_to_image(dpm_partition["partition_grid"])),
        ("PathFamily-overlay", compare._overlay_mask(dpm_image, dpm_ref, dpm_boundary)),
    ]
    compare._tile_images_with_titles(
        dpm_panel,
        args.output_dir / f"path_family_dpm_map{args.map_id}_tx{args.tx_id}.png",
    )

    irt2_panel = [
        ("IRT2", Image.fromarray(irt2_image, mode="L").convert("RGB")),
        ("IRT2-texture", Image.fromarray(compare._normalize_to_u8(irt2_strength), mode="L").convert("RGB")),
        ("PathFamily-order2", _partition_grid_to_image(irt2_partition["partition_grid"])),
        ("PathFamily-overlay", compare._overlay_mask(irt2_image, irt2_ref, irt2_boundary)),
    ]
    compare._tile_images_with_titles(
        irt2_panel,
        args.output_dir / f"path_family_irt2_map{args.map_id}_tx{args.tx_id}.png",
    )

    print(f"scene_id: {args.map_id}")
    print(f"tx_id: {args.tx_id}")
    _report_metrics("DPM/order1", dpm_metrics)
    _report_metrics("IRT2/order2", irt2_metrics)


if __name__ == "__main__":
    main()
