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
from rt2d.coverage import SequenceCostConfig, build_energy_pruned_sequence_render_grid

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

SPECIAL_LABEL_STYLES = {
    "blocked": {"color": "#111827", "label": "Blocked"},
    "unreachable": {"color": "#F8FAFC", "label": "Unreachable"},
    "L": {"color": "#0B6E4F", "label": "LoS"},
}


def _default_bounds(scene_id: str, requested_bounds: list[float] | None) -> tuple[float, float, float, float] | None:
    if requested_bounds is not None:
        return tuple(float(value) for value in requested_bounds)
    if str(scene_id).isdigit():
        return (0.0, 0.0, 255.0, 255.0)
    return None


def _grid_to_index_grid(grid: list[list[int]]) -> list[list[int]]:
    return [
        [LABEL_TO_INDEX.get(int(value), LABEL_TO_INDEX[-1]) for value in row]
        for row in grid
    ]


def _sequence_render_styles(labels: list[str]) -> dict[str, dict[str, str]]:
    styles = {label: SPECIAL_LABEL_STYLES[label] for label in SPECIAL_LABEL_STYLES if label in labels}
    dynamic = [label for label in labels if label not in styles]
    if not dynamic:
        return styles

    cmap = matplotlib.colormaps["tab20"]
    total = max(len(dynamic), 1)
    for index, label in enumerate(sorted(dynamic, key=lambda value: (len(value), value))):
        color = matplotlib.colors.to_hex(cmap(index / total))
        styles[label] = {"color": color, "label": label}
    return styles


def _grid_to_label_index_grid(
    grid: list[list[str]],
    labels: list[str],
) -> tuple[list[list[int]], dict[str, int]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    data = [[label_to_index[label] for label in row] for row in grid]
    return data, label_to_index


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


def _build_legend(ax: plt.Axes, counts: dict[str, int], styles: dict[str, dict[str, str]]) -> None:
    ordered = [
        "blocked",
        "unreachable",
        "L",
        "los",
        "order1",
        "order2",
        "order3",
        "order4",
    ]
    handles = []
    labels = []
    remaining_keys = [key for key in counts if key not in ordered and int(counts[key]) > 0]
    for count_key in ordered + remaining_keys:
        count = int(counts.get(count_key, 0))
        if count == 0:
            continue
        style = styles.get(count_key)
        if style is None and count_key in {"los", "order1", "order2", "order3", "order4"}:
            fallback_index = {"los": 2, "order1": 3, "order2": 4, "order3": 5, "order4": 6}[count_key]
            style = INDEX_TO_STYLE[fallback_index]
        if style is None:
            continue
        handle = plt.Line2D([0], [0], marker="s", linestyle="None", markersize=10, color=style["color"])
        handles.append(handle)
        labels.append(f"{style['label']} ({count})")

    if handles:
        ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.96)


def _render_pretty(
    scene: dict[str, Any],
    tx_id: int,
    tx_result: dict[str, Any],
    grid: dict[str, Any],
    output_path: Path,
    *,
    render_logic: str,
) -> None:
    extent = _build_extent(grid)
    if render_logic == "layered-sequence":
        label_grid = tx_result["layered_sequence_grid"]
        labels = sorted({label for row in label_grid for label in row}, key=lambda value: (value not in SPECIAL_LABEL_STYLES, len(value), value))
        styles = _sequence_render_styles(labels)
        data, label_to_index = _grid_to_label_index_grid(label_grid, labels)
        colors = [styles[label]["color"] for label, _index in sorted(label_to_index.items(), key=lambda item: item[1])]
        counts = tx_result["layered_sequence_counts"]
    else:
        colors = [INDEX_TO_STYLE[index]["color"] for index in sorted(INDEX_TO_STYLE)]
        data = _grid_to_index_grid(tx_result["visibility_order_grid"])
        styles = {
            "blocked": INDEX_TO_STYLE[0],
            "unreachable": INDEX_TO_STYLE[1],
            "los": INDEX_TO_STYLE[2],
            "order1": INDEX_TO_STYLE[3],
            "order2": INDEX_TO_STYLE[4],
            "order3": INDEX_TO_STYLE[5],
            "order4": INDEX_TO_STYLE[6],
        }
        counts = tx_result["counts"]
    cmap = ListedColormap(colors)

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
    title_suffix = "Layered Sequence" if render_logic == "layered-sequence" else "Minimal Order"
    ax.set_title(f"Scene {scene['scene_id']} TX {tx_id} RX Visibility ({title_suffix})", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.55)
    _build_legend(ax, counts, styles)

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
    *,
    render_logic: str,
) -> None:
    width = int(grid["width"])
    height = int(grid["height"])
    if render_logic == "layered-sequence":
        label_grid = tx_result["layered_sequence_grid"]
        labels = sorted({label for row in label_grid for label in row}, key=lambda value: (value not in SPECIAL_LABEL_STYLES, len(value), value))
        styles = _sequence_render_styles(labels)

        def color_at(row: int, col: int) -> tuple[int, int, int, int]:
            color = matplotlib.colors.to_rgba(styles[label_grid[row][col]]["color"], 1.0)
            return (
                int(round(color[0] * 255.0)),
                int(round(color[1] * 255.0)),
                int(round(color[2] * 255.0)),
                int(round(color[3] * 255.0)),
            )
    else:
        data = _grid_to_index_grid(tx_result["visibility_order_grid"])

        def color_at(row: int, col: int) -> tuple[int, int, int, int]:
            return _style_rgba(data[row][col])

    image = Image.new("RGBA", (width, height), _style_rgba(1))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = color_at(y, x)

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
        "--render-logic",
        choices=("minimal-order", "layered-sequence", "energy-pruned-sequence"),
        default="minimal-order",
        help="Choose between the original minimal-order rendering and the new layered sequence rendering.",
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
    bounds = _default_bounds(args.scene_id, args.bounds)
    needs_sequence_payload = args.render_logic in {"layered-sequence", "energy-pruned-sequence"}

    output = args.output or Path("build") / f"scene_{args.scene_id}_rx_visibility.png"
    scene = rt2d.load_scene(args.scene_id)

    if args.rx_json is None:
        payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=args.tx_ids,
            max_interactions=args.max_interactions,
            grid_step=args.grid_step,
            bounds=bounds,
            enable_reflection=not args.disable_reflection,
            enable_diffraction=not args.disable_diffraction,
            include_sequence_render_grid=needs_sequence_payload,
            include_sequence_hit_grids=args.render_logic == "energy-pruned-sequence",
        )
    else:
        payload = json.loads(args.rx_json.read_text(encoding="utf-8"))

    tx_ids = _resolve_tx_ids(payload, args.tx_ids)
    tx_results = {int(item["tx_id"]): item for item in payload["tx_results"]}
    multiple_outputs = len(tx_ids) > 1

    for tx_id in tx_ids:
        tx_result = tx_results[tx_id]
        if args.render_logic == "layered-sequence" and "layered_sequence_grid" not in tx_result:
            raise ValueError(
                "Selected render logic requires layered_sequence_grid in the payload. "
                "Recompute on the fly without --rx-json or export a payload that includes layered sequence data."
            )
        if args.render_logic == "energy-pruned-sequence":
            if "sequence_hit_grids" not in tx_result:
                raise ValueError(
                    "Selected render logic requires sequence_hit_grids in the payload. "
                    "Recompute on the fly without --rx-json or export a payload that includes sequence hit grids."
                )
            pruned_grid, pruned_counts = build_energy_pruned_sequence_render_grid(
                tx_result["sequence_hit_grids"],
                [[label != "blocked" for label in row] for row in tx_result["layered_sequence_grid"]],
                _tx_point(scene, tx_id),
                payload["grid"],
                SequenceCostConfig(),
            )
            tx_result = dict(tx_result)
            tx_result["layered_sequence_grid"] = pruned_grid
            tx_result["layered_sequence_counts"] = pruned_counts
        tx_output = _resolve_output_path(output, str(payload["scene_id"]), tx_id, multiple_outputs)
        if args.mode == "aligned":
            _render_aligned(
                scene,
                tx_id,
                tx_result,
                payload["grid"],
                tx_output,
                max(args.scale, 1),
                render_logic="layered-sequence" if args.render_logic == "energy-pruned-sequence" else args.render_logic,
            )
        else:
            _render_pretty(
                scene,
                tx_id,
                tx_result,
                payload["grid"],
                tx_output,
                render_logic="layered-sequence" if args.render_logic == "energy-pruned-sequence" else args.render_logic,
            )

        print(f"scene_id: {payload['scene_id']}")
        print(f"tx_id: {tx_id}")
        print(f"mode: {args.mode}")
        print(f"render_logic: {args.render_logic}")
        print(f"output: {tx_output}")


if __name__ == "__main__":
    main()
