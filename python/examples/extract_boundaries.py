from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

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
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=1,
        help="Maximum environment interactions to expand. Supported: 0, 1, 2.",
    )
    args = parser.parse_args()

    payload = rt2d.extract_scene_boundaries(
        args.scene_id,
        tx_ids=args.tx_ids,
        max_interactions=args.max_interactions,
        epsilon=args.epsilon,
        output_path=args.output,
    )

    print(f"scene_id: {payload['scene_id']}")
    print(f"boundary_count: {payload['boundary_count']}")
    if args.output is not None:
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()
