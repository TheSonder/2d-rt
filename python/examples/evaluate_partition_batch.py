from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d
import rt2d.coverage  # noqa: F401
import examples.compare_radiomapseer_feature_maps as compare


def _evaluate_map(
    radiomapseer_root: str,
    output_dir: str,
    map_id: int,
    tx_ids: list[int],
    max_interactions: int,
    acceleration_backend: str,
    torch_device: str | None,
    torch_state_chunk_size: int,
    torch_point_chunk_size: int,
    torch_edge_chunk_size: int,
) -> list[dict[str, Any]]:
    root = Path(radiomapseer_root)
    outdir = Path(output_dir)
    cache_dir = outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    scene = compare._load_scene(root, map_id)
    tx_suffix = "_".join(str(tx_id) for tx_id in tx_ids)
    cache_path = cache_dir / f"{map_id}_tx_{tx_suffix}_order{max_interactions}.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=tx_ids,
            max_interactions=max_interactions,
            bounds=compare.GRID_BOUNDS,
            include_sequence_render_grid=True,
            include_sequence_hit_grids=True,
            acceleration_backend=acceleration_backend,
            torch_device=torch_device,
            torch_state_chunk_size=torch_state_chunk_size,
            torch_point_chunk_size=torch_point_chunk_size,
            torch_edge_chunk_size=torch_edge_chunk_size,
            output_path=cache_path,
        )

    building_mask = compare._load_gray_image(
        root / "png" / "buildings_complete" / f"{map_id}.png"
    ) > 0

    rows: list[dict[str, Any]] = []
    for tx_result in payload["tx_results"]:
        tx_id = int(tx_result["tx_id"])
        sample_name = f"{map_id}_{tx_id}"
        outdoor_mask = np.asarray(
            [[label != "blocked" for label in row] for row in tx_result["layered_sequence_grid"]],
            dtype=bool,
        )
        scoring_mask = outdoor_mask & (~building_mask)
        tx_point = (
            float(scene["antenna"][tx_id][0]),
            float(scene["antenna"][tx_id][1]),
        )

        dpm_image = compare._load_gray_image(root / "gain" / "DPM" / f"{sample_name}.png")
        irt2_image = compare._load_gray_image(root / "gain" / "IRT2" / f"{sample_name}.png")
        dpm_strength = compare._texture_response(dpm_image, scoring_mask, tx_point)
        irt2_strength = compare._texture_response(irt2_image, scoring_mask, tx_point)

        dpm_ranking, _dpm_grouped, _dpm_masks = compare._evaluate_candidates(
            compare.DPM_CANDIDATES,
            tx_result,
            scene,
            tx_id,
            outdoor_mask,
            payload["grid"],
            dpm_strength,
            scoring_mask,
            sample_name,
        )
        irt2_ranking, _irt2_grouped, _irt2_masks = compare._evaluate_candidates(
            compare.IRT2_CANDIDATES,
            tx_result,
            scene,
            tx_id,
            outdoor_mask,
            payload["grid"],
            irt2_strength,
            scoring_mask,
            sample_name,
        )

        rows.append(
            {
                "sample": sample_name,
                "map_id": map_id,
                "tx_id": tx_id,
                "dpm_best": dpm_ranking[0]["candidate"],
                "dpm_best_f1": dpm_ranking[0]["metrics"]["f1_mean"],
                "irt2_best": irt2_ranking[0]["candidate"],
                "irt2_best_f1": irt2_ranking[0]["metrics"]["f1_mean"],
                "irt2_rd_f1": next(
                    item["metrics"]["f1_mean"] for item in irt2_ranking if item["candidate"] == "layered_rd"
                ),
                "irt2_rr_f1": next(
                    item["metrics"]["f1_mean"] for item in irt2_ranking if item["candidate"] == "layered_rr"
                ),
                "irt2_dr_f1": next(
                    item["metrics"]["f1_mean"] for item in irt2_ranking if item["candidate"] == "layered_dr"
                ),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate RT partition candidates against RadioMapSeer DPM / IRT2."
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
        default=Path("build") / "radiomapseer_partition_eval",
        help="Directory for batch outputs.",
    )
    parser.add_argument("--map-start", type=int, default=0, help="First map id.")
    parser.add_argument("--map-end", type=int, default=19, help="Last map id.")
    parser.add_argument(
        "--tx-id",
        type=int,
        action="append",
        dest="tx_ids",
        default=None,
        help="Fixed tx ids used for every map. Repeat this option to add more tx ids.",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=2,
        help="Maximum interaction order used in RT visibility evaluation.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--acceleration-backend",
        choices=("cpu", "auto", "torch"),
        default="cpu",
        help="Backend used by compute_rx_visibility.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Optional torch device, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument("--torch-state-chunk-size", type=int, default=16)
    parser.add_argument("--torch-point-chunk-size", type=int, default=4096)
    parser.add_argument("--torch-edge-chunk-size", type=int, default=64)
    args = parser.parse_args()

    tx_ids = args.tx_ids or [0, 40]
    map_ids = list(range(args.map_start, args.map_end + 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(args.max_workers, 1)
    if args.acceleration_backend != "cpu":
        max_workers = 1

    all_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                _evaluate_map,
                str(args.radiomapseer_root),
                str(args.output_dir),
                map_id,
                tx_ids,
                args.max_interactions,
                args.acceleration_backend,
                args.torch_device,
                args.torch_state_chunk_size,
                args.torch_point_chunk_size,
                args.torch_edge_chunk_size,
            ): map_id
            for map_id in map_ids
        }

        for future in as_completed(future_map):
            map_id = future_map[future]
            rows = future.result()
            all_rows.extend(rows)
            print(f"map {map_id} done: {len(rows)} tx")

    all_rows.sort(key=lambda item: (item["map_id"], item["tx_id"]))

    samples_csv = args.output_dir / "partition_eval_samples.csv"
    with samples_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample",
                "map_id",
                "tx_id",
                "dpm_best",
                "dpm_best_f1",
                "irt2_best",
                "irt2_best_f1",
                "irt2_rd_f1",
                "irt2_rr_f1",
                "irt2_dr_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    dpm_counter = Counter(row["dpm_best"] for row in all_rows)
    rr_rd_dr_counter = Counter(
        max(
            [
                ("layered_rd", row["irt2_rd_f1"]),
                ("layered_rr", row["irt2_rr_f1"]),
                ("layered_dr", row["irt2_dr_f1"]),
            ],
            key=lambda item: item[1],
        )[0]
        for row in all_rows
    )

    rd_vals = [row["irt2_rd_f1"] for row in all_rows]
    rr_vals = [row["irt2_rr_f1"] for row in all_rows]
    dr_vals = [row["irt2_dr_f1"] for row in all_rows]

    summary = {
        "map_ids": map_ids,
        "tx_ids": tx_ids,
        "sample_count": len(all_rows),
        "dpm_winner_counts": dict(dpm_counter),
        "irt2_rr_rd_dr_winner_counts": dict(rr_rd_dr_counter),
        "irt2_rr_rd_dr": {
            "layered_rd": {
                "mean_f1": statistics.mean(rd_vals),
                "std_f1": statistics.pstdev(rd_vals),
                "min_f1": min(rd_vals),
                "max_f1": max(rd_vals),
            },
            "layered_rr": {
                "mean_f1": statistics.mean(rr_vals),
                "std_f1": statistics.pstdev(rr_vals),
                "min_f1": min(rr_vals),
                "max_f1": max(rr_vals),
            },
            "layered_dr": {
                "mean_f1": statistics.mean(dr_vals),
                "std_f1": statistics.pstdev(dr_vals),
                "min_f1": min(dr_vals),
                "max_f1": max(dr_vals),
            },
        },
    }

    summary_json = args.output_dir / "partition_eval_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"samples_csv: {samples_csv}")
    print(f"summary_json: {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
