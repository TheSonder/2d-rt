from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d

TYPE_STYLES = {
    "los": {"color": "#0B6E4F", "linewidth": 1.2, "alpha": 0.85, "label": "LoS"},
    "reflection": {"color": "#C84C09", "linewidth": 1.0, "alpha": 0.75, "label": "Reflection"},
    "diffraction": {"color": "#1D4ED8", "linewidth": 0.0, "alpha": 0.95, "label": "Diffraction"},
    "mixed": {"color": "#7C3AED", "linewidth": 1.4, "alpha": 0.9, "label": "Mixed"},
}

ROLE_STYLE_OVERRIDES = {
    "reflection_face": {"color": "#C84C09", "linewidth": 1.0, "alpha": 0.8, "label": "Reflection Face"},
    "reflection_shadow": {"color": "#6D28D9", "linewidth": 1.0, "alpha": 0.95, "label": "Reflection Shadow"},
}


def _plot_scene(ax: plt.Axes, scene: dict[str, Any], *, aligned: bool) -> None:
    for polygon in scene["polygons"]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        if aligned:
            ax.fill(xs, ys, facecolor=(0.90, 0.90, 0.90, 0.65), edgecolor="#4B5563", linewidth=0.6, zorder=1)
        else:
            ax.fill(xs, ys, facecolor="#E5E7EB", edgecolor="#4B5563", linewidth=0.8, zorder=1)


def _plot_boundaries(ax: plt.Axes, boundaries: list[dict[str, Any]], highlight_tx_ids: set[int]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for boundary in boundaries:
        tx_id = int(boundary["tx_id"])
        if highlight_tx_ids and tx_id not in highlight_tx_ids:
            continue

        boundary_type = str(boundary["type"])
        role = str(boundary.get("role", ""))
        style = ROLE_STYLE_OVERRIDES.get(role, TYPE_STYLES.get(boundary_type))
        if style is None:
            continue

        legend_key = role if role in ROLE_STYLE_OVERRIDES else boundary_type
        counts[legend_key] = counts.get(legend_key, 0) + 1
        p0 = boundary["p0"]
        p1 = boundary["p1"]

        if abs(p0[0] - p1[0]) < 1.0e-9 and abs(p0[1] - p1[1]) < 1.0e-9:
            ax.scatter(
                p0[0],
                p0[1],
                s=18,
                c=style["color"],
                alpha=style["alpha"],
                zorder=6,
            )
            continue

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=3 if boundary_type == "los" else 4,
        )

    return counts


def _build_legend(ax: plt.Axes, counts: dict[str, int]) -> None:
    handles = []
    labels = []
    ordered_styles = {
        "los": TYPE_STYLES["los"],
        "reflection_face": ROLE_STYLE_OVERRIDES["reflection_face"],
        "reflection_shadow": ROLE_STYLE_OVERRIDES["reflection_shadow"],
        "diffraction": TYPE_STYLES["diffraction"],
        "mixed": TYPE_STYLES["mixed"],
    }
    for boundary_type, style in ordered_styles.items():
        count = counts.get(boundary_type, 0)
        if count == 0:
            continue

        handle = plt.Line2D(
            [0],
            [0],
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )
        handles.append(handle)
        labels.append(f"{style['label']} ({count})")

    if handles:
        ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.95)


def _infer_canvas_extent(
    scene: dict[str, Any],
    boundaries: list[dict[str, Any]],
    canvas_size: int,
    *,
    aligned: bool,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    for point in scene["antenna"]:
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    for polygon in scene["polygons"]:
        for point in polygon:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    if not aligned:
        for boundary in boundaries:
            xs.extend([float(boundary["p0"][0]), float(boundary["p1"][0])])
            ys.extend([float(boundary["p0"][1]), float(boundary["p1"][1])])

    if not xs or not ys:
        return 0.0, float(canvas_size), 0.0, float(canvas_size)

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # Aligned mode must preserve the source image grid. Do not let out-of-bounds
    # boundary extensions rescale the scene.
    if aligned and min_x >= -1.0 and max_x <= canvas_size + 1.0 and min_y >= -1.0 and max_y <= canvas_size + 1.0:
        return 0.0, float(canvas_size), 0.0, float(canvas_size)

    # RadioMapSeer scenes are typically authored on a 256x256 image grid. If the
    # scene already fits that grid, keep the full grid instead of tight-cropping.
    if min_x >= -1.0 and max_x <= canvas_size + 1.0 and min_y >= -1.0 and max_y <= canvas_size + 1.0:
        return 0.0, float(canvas_size), 0.0, float(canvas_size)

    return min_x, max_x, min_y, max_y


def _draw_radiomap_background(ax: plt.Axes, image_path: Path, extent: tuple[float, float, float, float], aligned: bool) -> None:
    image = plt.imread(image_path)
    x_min, x_max, y_min, y_max = extent
    if aligned:
        image_extent = (x_min, x_max, y_max, y_min)
    else:
        image_extent = (x_min, x_max, y_min, y_max)
    ax.imshow(image, extent=image_extent, interpolation="nearest", zorder=0)


def _style_to_rgba(style: dict[str, Any]) -> tuple[int, int, int, int]:
    r, g, b, a = matplotlib.colors.to_rgba(style["color"], style.get("alpha", 1.0))
    return (
        int(round(r * 255.0)),
        int(round(g * 255.0)),
        int(round(b * 255.0)),
        int(round(a * 255.0)),
    )


def _world_to_raster(
    point: list[float] | tuple[float, float],
    extent: tuple[float, float, float, float],
    canvas_size: int,
) -> tuple[float, float]:
    x_min, x_max, y_min, y_max = extent
    width = max(x_max - x_min, 1.0e-9)
    height = max(y_max - y_min, 1.0e-9)
    x = ((float(point[0]) - x_min) / width) * canvas_size
    y = ((y_max - float(point[1])) / height) * canvas_size
    x = min(max(x, 0.0), float(canvas_size - 1))
    y = min(max(y, 0.0), float(canvas_size - 1))
    return x, y


def _render_aligned_image(
    scene: dict[str, Any],
    boundaries: list[dict[str, Any]],
    highlight_tx_ids: set[int],
    extent: tuple[float, float, float, float],
    canvas_size: int,
    radiomap_png: Path | None,
    *,
    geometry_only: bool,
) -> Image.Image:
    if radiomap_png is not None and not geometry_only:
        image = Image.open(radiomap_png).convert("RGBA")
        if image.size != (canvas_size, canvas_size):
            image = image.resize((canvas_size, canvas_size), Image.Resampling.NEAREST)
    else:
        image = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))

    draw = ImageDraw.Draw(image, "RGBA")

    if radiomap_png is None or geometry_only:
        for polygon in scene["polygons"]:
            points = [_world_to_raster(point, extent, canvas_size) for point in polygon]
            fill = (0, 0, 0, 255) if geometry_only else (32, 32, 32, 220)
            outline = (0, 0, 0, 255) if geometry_only else (24, 24, 27, 255)
            draw.polygon(points, fill=fill, outline=outline)

    for boundary in boundaries:
        tx_id = int(boundary["tx_id"])
        if highlight_tx_ids and tx_id not in highlight_tx_ids:
            continue

        boundary_type = str(boundary["type"])
        role = str(boundary.get("role", ""))
        style = ROLE_STYLE_OVERRIDES.get(role, TYPE_STYLES.get(boundary_type))
        if style is None:
            continue

        p0 = _world_to_raster(boundary["p0"], extent, canvas_size)
        p1 = _world_to_raster(boundary["p1"], extent, canvas_size)
        color = _style_to_rgba(style)
        width = max(1, int(round(float(style.get("linewidth", 1.0)))))

        if abs(p0[0] - p1[0]) < 1.0e-9 and abs(p0[1] - p1[1]) < 1.0e-9:
            radius = 1 if boundary_type == "diffraction" else 2
            draw.ellipse((p0[0] - radius, p0[1] - radius, p0[0] + radius, p0[1] + radius), fill=color)
            continue

        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)

    return image


def _create_figure(mode: str, figsize: tuple[float, float], canvas_size: int) -> tuple[plt.Figure, plt.Axes, int]:
    if mode == "aligned":
        dpi = 100
        side_inch = canvas_size / dpi
        fig = plt.figure(figsize=(side_inch, side_inch), dpi=dpi, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        return fig, ax, dpi

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax, 220


def _configure_axes(
    ax: plt.Axes,
    mode: str,
    extent: tuple[float, float, float, float],
    scene_id: str,
) -> None:
    x_min, x_max, y_min, y_max = extent
    ax.set_aspect("equal", adjustable="box")

    if mode == "aligned":
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.axis("off")
        return

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Scene {scene_id} Boundary Visualization", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#CBD5E1", linewidth=0.5, alpha=0.6)


def _resolve_tx_ids(boundaries: list[dict[str, Any]], requested_tx_ids: list[int] | None) -> list[int]:
    if requested_tx_ids:
        return list(dict.fromkeys(int(tx_id) for tx_id in requested_tx_ids))
    return sorted({int(boundary["tx_id"]) for boundary in boundaries})


def _resolve_output_path(base_output: Path, scene_id: str, tx_id: int, multiple: bool) -> Path:
    if multiple:
        stem = base_output.stem if base_output.suffix else base_output.name
        suffix = base_output.suffix or ".png"
        return base_output.with_name(f"{stem}_tx{tx_id}{suffix}")
    if base_output.suffix:
        return base_output
    return base_output / f"scene_{scene_id}_tx{tx_id}_boundaries.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 2D propagation boundaries.")
    parser.add_argument("scene_id", help="Scene id under data/environment")
    parser.add_argument(
        "--tx-id",
        type=int,
        action="append",
        dest="tx_ids",
        help="Optional TX index. Repeat to plot selected TX only. Default: all TX in payload.",
    )
    parser.add_argument(
        "--boundary-json",
        type=Path,
        default=None,
        help="Optional precomputed boundary JSON. If omitted, boundaries are extracted on the fly.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: build/scene_<scene_id>_boundaries.png",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=(10.0, 10.0),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=1,
        help="Maximum environment interactions to expand when extracting on the fly. Supported: 0, 1, 2.",
    )
    parser.add_argument(
        "--mode",
        choices=("pretty", "aligned", "aligned-geometry"),
        default="pretty",
        help="Visualization mode. 'aligned' overlays on the RadioMapSeer-style image grid. 'aligned-geometry' exports a geometry-only aligned image with buildings and boundary lines.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=256,
        help="Aligned mode canvas size in pixels. Default: 256.",
    )
    parser.add_argument(
        "--radiomap-png",
        type=Path,
        default=None,
        help="Optional RadioMapSeer PNG used as image-aligned background.",
    )
    parser.add_argument(
        "--with-diffraction",
        action="store_true",
        help="Enable diffraction expansion when extracting boundaries on the fly. Disabled by default.",
    )
    args = parser.parse_args()

    output = args.output or Path("build") / f"scene_{args.scene_id}_boundaries.png"

    scene = rt2d.load_scene(args.scene_id)
    if args.boundary_json is None:
        payload = rt2d.extract_scene_boundaries(
            scene,
            tx_ids=args.tx_ids,
            max_interactions=args.max_interactions,
            include_diffraction=args.with_diffraction,
        )
    else:
        payload = json.loads(args.boundary_json.read_text(encoding="utf-8"))
    tx_ids = _resolve_tx_ids(payload["boundaries"], args.tx_ids)
    multiple_outputs = len(tx_ids) > 1
    aligned_mode = args.mode in {"aligned", "aligned-geometry"}
    geometry_only = args.mode == "aligned-geometry"

    extent = _infer_canvas_extent(
        scene,
        payload["boundaries"],
        args.canvas_size,
        aligned=aligned_mode,
    )
    written_outputs: list[Path] = []

    for tx_id in tx_ids:
        highlight_tx_ids = {tx_id}
        tx_output = _resolve_output_path(output, str(payload["scene_id"]), tx_id, multiple_outputs)

        if aligned_mode:
            image = _render_aligned_image(
                scene,
                payload["boundaries"],
                highlight_tx_ids,
                extent,
                args.canvas_size,
                args.radiomap_png,
                geometry_only=geometry_only,
            )
            tx_output.parent.mkdir(parents=True, exist_ok=True)
            image.save(tx_output)
        else:
            fig, ax, dpi = _create_figure(args.mode, tuple(args.figsize), args.canvas_size)
            fig.patch.set_facecolor("#F8FAFC")
            ax.set_facecolor("#F8FAFC")
            if args.radiomap_png is not None:
                _draw_radiomap_background(ax, args.radiomap_png, extent, aligned=False)

            _plot_scene(ax, scene, aligned=False)
            counts = _plot_boundaries(ax, payload["boundaries"], highlight_tx_ids)
            _build_legend(ax, counts)
            _configure_axes(ax, args.mode, extent, str(payload["scene_id"]))

            tx_output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(tx_output, dpi=dpi, pad_inches=0)
            plt.close(fig)

        written_outputs.append(tx_output)

    print(f"scene_id: {payload['scene_id']}")
    print(f"boundary_count: {payload['boundary_count']}")
    print(f"mode: {args.mode}")
    print(f"tx_ids: {tx_ids}")
    for written_output in written_outputs:
        print(f"output: {written_output}")


if __name__ == "__main__":
    main()
