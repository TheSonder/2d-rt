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

VISIBILITY_COLOR = (11, 110, 79, 255)
LOS_SHADOW_COLOR = (185, 28, 28, 255)
BUILDING_FILL = (0, 0, 0, 255)
BACKGROUND = (255, 255, 255, 255)
CANVAS_SIZE = 256


def _load_scene(radiomapseer_root: Path, map_id: int) -> dict[str, object]:
    antenna_path = radiomapseer_root / "antenna" / f"{map_id}.json"
    polygon_path = radiomapseer_root / "polygon" / "buildings_and_cars" / f"{map_id}.json"

    return {
        "scene_id": str(map_id),
        "root_dir": str(radiomapseer_root),
        "antenna_path": str(antenna_path),
        "polygon_path": str(polygon_path),
        "antenna": json.loads(antenna_path.read_text(encoding="utf-8")),
        "polygons": json.loads(polygon_path.read_text(encoding="utf-8")),
    }


def _draw_scene(image: Image.Image, scene: dict[str, object], boundaries: list[dict[str, object]]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")

    for polygon in scene["polygons"]:
        points = [tuple(point) for point in polygon]
        draw.polygon(points, fill=BUILDING_FILL, outline=BUILDING_FILL)

    for boundary in boundaries:
        role = str(boundary["role"])
        if role == "visibility":
            color = VISIBILITY_COLOR
        elif role == "los_shadow":
            color = LOS_SHADOW_COLOR
        else:
            continue

        p0 = tuple(boundary["p0"])
        p1 = tuple(boundary["p1"])
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=1)


def _resolve_tx_ids(
    gain_dir: Path,
    antenna_png_dir: Path,
    map_id: int,
) -> list[int]:
    gain_ids = {
        int(path.stem.split("_")[1])
        for path in gain_dir.glob(f"{map_id}_*.png")
    }
    antenna_ids = {
        int(path.stem.split("_")[1])
        for path in antenna_png_dir.glob(f"{map_id}_*.png")
    }
    return sorted(gain_ids & antenna_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RadioMapSeer LoS + LoS-shadow label images.")
    parser.add_argument(
        "--radiomapseer-root",
        type=Path,
        default=Path(r"D:\TheSonder\0.8 Resources\data\RadioMapSeer"),
        help="RadioMapSeer root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\TheSonder\0.8 Resources\data\RadioMapSeer\DPM"),
        help="Output directory for generated label PNG files.",
    )
    parser.add_argument("--map-start", type=int, default=12, help="First map id to generate.")
    parser.add_argument("--map-end", type=int, default=20, help="Last map id to generate.")
    args = parser.parse_args()

    gain_dir = args.radiomapseer_root / "gain" / "DPM"
    antenna_png_dir = args.radiomapseer_root / "png" / "antennas"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for map_id in range(args.map_start, args.map_end + 1):
        scene = _load_scene(args.radiomapseer_root, map_id)
        tx_ids = _resolve_tx_ids(gain_dir, antenna_png_dir, map_id)
        if not tx_ids:
            print(f"skip map {map_id}: no matching gain/antenna PNG names found")
            continue

        for tx_id in tx_ids:
            payload = rt2d.extract_scene_boundaries(scene, tx_ids=[tx_id], max_interactions=0)
            image = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND)
            _draw_scene(image, scene, payload["boundaries"])

            output_path = args.output_dir / f"{map_id}_{tx_id}.png"
            image.save(output_path)
            generated += 1

        print(f"map {map_id}: generated {len(tx_ids)} files")

    print(f"generated_total: {generated}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
