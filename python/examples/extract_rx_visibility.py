from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RX grid visibility orders for a 2D RT scene.")
    parser.add_argument("scene_id", help="Scene id under data/environment")
    parser.add_argument(
        "--tx-id",
        type=int,
        action="append",
        dest="tx_ids",
        help="Optional TX index. Repeat to select multiple TX. Default: all TX.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-6,
        help="Geometry epsilon.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=1.0,
        help="RX grid step in world units.",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=1,
        help="Maximum interaction order. Supported: 0, 1, 2, 3, 4.",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
        default=None,
        help="Optional explicit RX grid bounds.",
    )
    parser.add_argument(
        "--disable-reflection",
        action="store_true",
        help="Disable reflection states.",
    )
    parser.add_argument(
        "--disable-diffraction",
        action="store_true",
        help="Disable diffraction states.",
    )
    args = parser.parse_args()

    payload = rt2d.compute_rx_visibility(
        args.scene_id,
        tx_ids=args.tx_ids,
        max_interactions=args.max_interactions,
        grid_step=args.grid_step,
        bounds=tuple(args.bounds) if args.bounds is not None else None,
        epsilon=args.epsilon,
        enable_reflection=not args.disable_reflection,
        enable_diffraction=not args.disable_diffraction,
        output_path=args.output,
    )

    print(f"scene_id: {payload['scene_id']}")
    grid = payload["grid"]
    print(f"grid: {grid['width']}x{grid['height']} step={grid['step']}")
    for tx_result in payload["tx_results"]:
        print(f"tx_id: {tx_result['tx_id']}")
        counts = tx_result["counts"]
        print(
            "counts: "
            f"blocked={counts['blocked']} "
            f"unreachable={counts['unreachable']} "
            f"los={counts['los']} "
            f"order1={counts['order1']} "
            f"order2={counts['order2']} "
            f"order3={counts['order3']} "
            f"order4={counts['order4']}"
        )
    if args.output is not None:
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()
