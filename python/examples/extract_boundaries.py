from __future__ import annotations

import argparse
from pathlib import Path

import rt2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract propagation boundaries for a 2D RT scene.")
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
    args = parser.parse_args()

    payload = rt2d.extract_scene_boundaries(
        args.scene_id,
        tx_ids=args.tx_ids,
        epsilon=args.epsilon,
        output_path=args.output,
    )

    print(f"scene_id: {payload['scene_id']}")
    print(f"boundary_count: {payload['boundary_count']}")
    if args.output is not None:
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()

