from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d

LABEL_TO_INDEX = {
    -2: 0,
    -1: 1,
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
}

INDEX_TO_STYLE = {
    0: {"color": "#111827", "label": "Blocked"},
    1: {"color": "#F8FAFC", "label": "Unreachable"},
    2: {"color": "#0B6E4F", "label": "LoS"},
    3: {"color": "#C84C09", "label": "Order 1"},
    4: {"color": "#2563EB", "label": "Order 2"},
    5: {"color": "#DC2626", "label": "Order 3"},
    6: {"color": "#7C3AED", "label": "Order 4"},
}


def _grid_to_index_grid(grid: list[list[int]]) -> list[list[int]]:
    return [
        [LABEL_TO_INDEX.get(int(value), LABEL_TO_INDEX[-1]) for value in row]
        for row in grid
    ]


def _style_rgba(index: int) -> tuple[int, int, int, int]:
    color = matplotlib.colors.to_rgba(INDEX_TO_STYLE[index]["color"], 1.0)
    return (
        int(round(color[0] * 255.0)),
        int(round(color[1] * 255.0)),
        int(round(color[2] * 255.0)),
        int(round(color[3] * 255.0)),
    )


def _resolve_tx_ids(payload: dict[str, Any], requested_tx_ids: list[int] | None) -> list[int]:
    if requested_tx_ids:
        return list(dict.fromkeys(int(tx_id) for tx_id in requested_tx_ids))
    return [int(item["tx_id"]) for item in payload["tx_results"]]


def _resolve_output_path(base_output: Path, scene_id: str, tx_id: int, multiple: bool) -> Path:
    if multiple:
        stem = base_output.stem if base_output.suffix else base_output.name
        suffix = base_output.suffix or ".png"
        return base_output.with_name(f"{stem}_tx{tx_id}{suffix}")
    if base_output.suffix:
        return base_output
    return base_output / f"scene_{scene_id}_tx{tx_id}_rx_visibility.png"


def _tx_point(scene: dict[str, Any], tx_id: int) -> tuple[float, float]:
    point = scene["antenna"][tx_id]
    return (float(point[0]), float(point[1]))


def _build_extent(grid: dict[str, Any]) -> tuple[float, float, float, float]:
    step = float(grid["step"])
    min_x = float(grid["min_x"])
    min_y = float(grid["min_y"])
    max_x = float(grid["max_x"])
    max_y = float(grid["max_y"])
    return (
        min_x - step * 0.5,
        max_x + step * 0.5,
        min_y - step * 0.5,
        max_y + step * 0.5,
    )


def _plot_scene(ax: plt.Axes, scene: dict[str, Any]) -> None:
    for polygon in scene["polygons"]:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        ax.fill(xs, ys, facecolor="#111827", edgecolor="#111827", linewidth=0.8, alpha=0.95, zorder=3)


def _build_legend(ax: plt.Axes, counts: dict[str, int]) -> None:
    ordered = [
        ("blocked", 0),
        ("unreachable", 1),
        ("los", 2),
        ("order1", 3),
        ("order2", 4),
        ("order3", 5),
        ("order4", 6),
    ]
    handles = []
    labels = []
    for count_key, index in ordered:
        count = int(counts.get(count_key, 0))
        if count == 0:
            continue
        handle = plt.Line2D([0], [0], marker="s", linestyle="None", markersize=10, color=INDEX_TO_STYLE[index]["color"])
        handles.append(handle)
        labels.append(f"{INDEX_TO_STYLE[index]['label']} ({count})")

    if handles:
        ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.96)


def _render_pretty(
    scene: dict[str, Any],
    tx_id: int,
    tx_result: dict[str, Any],
    grid: dict[str, Any],
    output_path: Path,
) -> None:
    extent = _build_extent(grid)
    colors = [INDEX_TO_STYLE[index]["color"] for index in sorted(INDEX_TO_STYLE)]
    cmap = ListedColormap(colors)
    data = _grid_to_index_grid(tx_result["visibility_order_grid"])

    fig, ax = plt.subplots(figsize=(10.0, 10.0), constrained_layout=True)
    fig.patch.set_facecolor("#F3F4F6")
    ax.set_facecolor("#F3F4F6")

    ax.imshow(data, cmap=cmap, interpolation="nearest", extent=extent, origin="upper", zorder=0)
    _plot_scene(ax, scene)

    tx_x, tx_y = _tx_point(scene, tx_id)
    ax.scatter(tx_x, tx_y, s=80, c="#F59E0B", edgecolors="#111827", linewidths=0.9, marker="*", zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"Scene {scene['scene_id']} TX {tx_id} RX Visibility", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.55)
    _build_legend(ax, tx_result["counts"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, pad_inches=0.05)
    plt.close(fig)


def _render_aligned(
    scene: dict[str, Any],
    tx_id: int,
    tx_result: dict[str, Any],
    grid: dict[str, Any],
    output_path: Path,
    scale: int,
) -> None:
    width = int(grid["width"])
    height = int(grid["height"])
    image = Image.new("RGBA", (width, height), _style_rgba(1))
    pixels = image.load()
    data = _grid_to_index_grid(tx_result["visibility_order_grid"])
    for y in range(height):
        for x in range(width):
            pixels[x, y] = _style_rgba(data[y][x])

    draw = ImageDraw.Draw(image, "RGBA")
    min_x = float(grid["min_x"])
    max_y = float(grid["max_y"])
    step = float(grid["step"])

    def world_to_pixel(point: list[float] | tuple[float, float]) -> tuple[float, float]:
        px = (float(point[0]) - min_x) / step
        py = (max_y - float(point[1])) / step
        return (px, py)

    for polygon in scene["polygons"]:
        points = [world_to_pixel(point) for point in polygon]
        draw.polygon(points, fill=_style_rgba(0), outline=_style_rgba(0))

    tx_x, tx_y = world_to_pixel(_tx_point(scene, tx_id))
    radius = 2
    draw.ellipse((tx_x - radius, tx_y - radius, tx_x + radius, tx_y + radius), fill=(245, 158, 11, 255), outline=(17, 24, 39, 255))

    if scale > 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RX visibility order grids.")
    parser.add_argument("scene_id", help="Scene id under data/environment")
    parser.add_argument(
        "--tx-id",
        type=int,
        action="append",
        dest="tx_ids",
        help="Optional TX index. Repeat to render selected TX only. Default: all TX.",
    )
    parser.add_argument(
        "--rx-json",
        type=Path,
        default=None,
        help="Optional precomputed RX visibility JSON. If omitted, compute on the fly.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: build/scene_<scene_id>_rx_visibility.png",
    )
    parser.add_argument(
        "--mode",
        choices=("pretty", "aligned"),
        default="pretty",
        help="Visualization mode. 'aligned' writes the RX grid directly as a pixel map.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Pixel upsample factor for aligned mode.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=1.0,
        help="RX grid step in world units when computing on the fly.",
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
        help="Disable reflection states when computing on the fly.",
    )
    parser.add_argument(
        "--disable-diffraction",
        action="store_true",
        help="Disable diffraction states when computing on the fly.",
    )
    args = parser.parse_args()

    output = args.output or Path("build") / f"scene_{args.scene_id}_rx_visibility.png"
    scene = rt2d.load_scene(args.scene_id)

    if args.rx_json is None:
        payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=args.tx_ids,
            max_interactions=args.max_interactions,
            grid_step=args.grid_step,
            bounds=tuple(args.bounds) if args.bounds is not None else None,
            enable_reflection=not args.disable_reflection,
            enable_diffraction=not args.disable_diffraction,
        )
    else:
        payload = json.loads(args.rx_json.read_text(encoding="utf-8"))

    tx_ids = _resolve_tx_ids(payload, args.tx_ids)
    tx_results = {int(item["tx_id"]): item for item in payload["tx_results"]}
    multiple_outputs = len(tx_ids) > 1

    for tx_id in tx_ids:
        tx_result = tx_results[tx_id]
        tx_output = _resolve_output_path(output, str(payload["scene_id"]), tx_id, multiple_outputs)
        if args.mode == "aligned":
            _render_aligned(scene, tx_id, tx_result, payload["grid"], tx_output, max(args.scale, 1))
        else:
            _render_pretty(scene, tx_id, tx_result, payload["grid"], tx_output)

        print(f"scene_id: {payload['scene_id']}")
        print(f"tx_id: {tx_id}")
        print(f"mode: {args.mode}")
        print(f"output: {tx_output}")


if __name__ == "__main__":
    main()
