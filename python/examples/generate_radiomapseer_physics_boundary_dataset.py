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
REFLECTION_FACE_COLOR = (200, 76, 9, 255)
REFLECTION_SHADOW_COLOR = (109, 40, 217, 255)


def _to_image_point(point: list[float] | tuple[float, float], image_height: int) -> tuple[int, int]:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    return (x, image_height - 1 - y)


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


def _resolve_tx_ids(gain_dir: Path, antenna_png_dir: Path, map_id: int) -> list[int]:
    gain_ids = {
        int(path.stem.split("_")[1])
        for path in gain_dir.glob(f"{map_id}_*.png")
    }
    antenna_ids = {
        int(path.stem.split("_")[1])
        for path in antenna_png_dir.glob(f"{map_id}_*.png")
    }
    return sorted(gain_ids & antenna_ids)


def _overlay_boundaries(
    base_image_path: Path,
    boundaries: list[dict[str, object]],
    output_path: Path,
) -> None:
    image = Image.open(base_image_path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    image_height = image.height

    for boundary in boundaries:
        role = str(boundary["role"])
        if role == "visibility":
            color = VISIBILITY_COLOR
        elif role == "los_shadow":
            color = LOS_SHADOW_COLOR
        elif role == "reflection_face":
            color = REFLECTION_FACE_COLOR
        elif role == "reflection_shadow":
            color = REFLECTION_SHADOW_COLOR
        else:
            continue

        p0 = _to_image_point(boundary["p0"], image_height)
        p1 = _to_image_point(boundary["p1"], image_height)
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RadioMapSeer physics boundary images with LoS and LoS-shadow overlays."
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
        default=Path(r"D:\TheSonder\0.8 Resources\data\RadioMapSeer\physics_boundary\DPM"),
        help="Output directory for generated PNG files.",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=0,
        help="Boundary interaction depth. Use 0 for LoS only, 1 for first-order reflections.",
    )
    parser.add_argument("--map-start", type=int, default=0, help="First map id to generate.")
    parser.add_argument("--map-end", type=int, default=19, help="Last map id to generate.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the total number of generated files. Useful for sampling.",
    )
    args = parser.parse_args()

    gain_dir = args.radiomapseer_root / "gain" / "DPM"
    antenna_png_dir = args.radiomapseer_root / "png" / "antennas"
    building_png_dir = args.radiomapseer_root / "png" / "buildings_complete"

    generated = 0
    for map_id in range(args.map_start, args.map_end + 1):
        scene = _load_scene(args.radiomapseer_root, map_id)
        tx_ids = _resolve_tx_ids(gain_dir, antenna_png_dir, map_id)
        if not tx_ids:
            print(f"skip map {map_id}: no matching gain/antenna PNG names found")
            continue

        base_image_path = building_png_dir / f"{map_id}.png"
        if not base_image_path.is_file():
            print(f"skip map {map_id}: missing building PNG {base_image_path}")
            continue

        for tx_id in tx_ids:
            payload = rt2d.extract_scene_boundaries(
                scene,
                tx_ids=[tx_id],
                max_interactions=args.max_interactions,
            )
            output_path = args.output_dir / f"{map_id}_{tx_id}.png"
            _overlay_boundaries(base_image_path, payload["boundaries"], output_path)
            generated += 1

            if args.max_files is not None and generated >= args.max_files:
                print(f"generated_total: {generated}")
                print(f"output_dir: {args.output_dir}")
                return

        print(f"map {map_id}: generated {len(tx_ids)} files")

    print(f"generated_total: {generated}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
