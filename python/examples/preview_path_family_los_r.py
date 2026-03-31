from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d


PARTITION_COLORS = {
    "L": (22, 163, 74, 160),
    "R": (245, 158, 11, 160),
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


def _overlay_partition(
    base_image_path: Path,
    partition_grid: list[list[str]],
    output_path: Path,
) -> None:
    image = Image.open(base_image_path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")

    height = len(partition_grid)
    width = len(partition_grid[0]) if height else 0
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image, "RGBA")

    for row, labels in enumerate(partition_grid):
        for col, label in enumerate(labels):
            color = PARTITION_COLORS.get(label)
            if color is None:
                continue
            draw.point((col, row), fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview path-family LoS + R partition over building PNG.")
    parser.add_argument("map_id", type=int, help="RadioMapSeer map id.")
    parser.add_argument("--tx-id", type=int, default=0, help="TX index.")
    parser.add_argument(
        "--radiomapseer-root",
        type=Path,
        default=Path(r"D:\TheSonder\0.8 Resources\data\RadioMapSeer"),
        help="RadioMapSeer root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: build/path_family_map_<id>_tx_<tx>.png",
    )
    parser.add_argument(
        "--acceleration-backend",
        choices=("cpu", "auto", "torch"),
        default="cpu",
        help="Backend used by the underlying runtime.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Optional torch device, for example 'cuda' or 'cpu'.",
    )
    args = parser.parse_args()

    scene = _load_scene(args.radiomapseer_root, args.map_id)
    runtime = rt2d.build_path_family_runtime(
        scene,
        tx_id=args.tx_id,
        bounds=(0.0, 0.0, 255.0, 255.0),
        acceleration_backend=args.acceleration_backend,
        torch_device=args.torch_device,
    )
    payload = rt2d.compute_path_family_partition_runtime(runtime)

    base_image_path = args.radiomapseer_root / "png" / "buildings_complete" / f"{args.map_id}.png"
    output_path = args.output or Path("build") / f"path_family_map_{args.map_id}_tx_{args.tx_id}.png"
    _overlay_partition(base_image_path, payload["partition_grid"], output_path)

    print(f"scene_id: {payload['scene_id']}")
    print(f"tx_id: {payload['tx_id']}")
    print(f"counts: {payload['counts']}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
