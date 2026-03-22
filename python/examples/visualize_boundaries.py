from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import rt2d

TYPE_STYLES = {
    "los": {"color": "#0B6E4F", "linewidth": 1.2, "alpha": 0.85, "label": "LoS"},
    "reflection": {"color": "#C84C09", "linewidth": 1.0, "alpha": 0.75, "label": "Reflection"},
    "diffraction": {"color": "#1D4ED8", "linewidth": 0.0, "alpha": 0.95, "label": "Diffraction"},
    "mixed": {"color": "#7C3AED", "linewidth": 1.4, "alpha": 0.9, "label": "Mixed"},
}


def _load_payload(
    scene_id: str,
    tx_ids: list[int] | None,
    boundary_json: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = rt2d.load_scene(scene_id)
    if boundary_json is None:
        payload = rt2d.extract_scene_boundaries(scene, tx_ids=tx_ids)
        return scene, payload

    payload = json.loads(boundary_json.read_text(encoding="utf-8"))
    return scene, payload


def _plot_scene(ax: plt.Axes, scene: dict[str, Any], highlight_tx_ids: set[int]) -> None:
    for polygon in scene["polygons"]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        ax.fill(xs, ys, facecolor="#E5E7EB", edgecolor="#4B5563", linewidth=0.8, zorder=1)

    for tx_id, point in enumerate(scene["antenna"]):
        is_active = tx_id in highlight_tx_ids
        ax.scatter(
            point[0],
            point[1],
            s=28 if is_active else 18,
            c="#D90429" if is_active else "#6B7280",
            marker="*",
            zorder=5,
        )
        if is_active:
            ax.text(point[0] + 1.5, point[1] + 1.5, f"TX {tx_id}", fontsize=8, color="#991B1B")


def _plot_boundaries(ax: plt.Axes, boundaries: list[dict[str, Any]], highlight_tx_ids: set[int]) -> dict[str, int]:
    counts = {key: 0 for key in TYPE_STYLES}

    for boundary in boundaries:
        tx_id = int(boundary["tx_id"])
        if highlight_tx_ids and tx_id not in highlight_tx_ids:
            continue

        boundary_type = str(boundary["type"])
        style = TYPE_STYLES.get(boundary_type)
        if style is None:
            continue

        counts[boundary_type] += 1
        p0 = boundary["p0"]
        p1 = boundary["p1"]

        if boundary_type == "diffraction" or (
            abs(p0[0] - p1[0]) < 1.0e-9 and abs(p0[1] - p1[1]) < 1.0e-9
        ):
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
    for boundary_type, style in TYPE_STYLES.items():
        count = counts[boundary_type]
        if count == 0:
            continue

        if boundary_type == "diffraction":
            handle = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=style["color"],
                markersize=6,
                linewidth=0,
            )
        else:
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
    args = parser.parse_args()

    output = args.output or Path("build") / f"scene_{args.scene_id}_boundaries.png"
    highlight_tx_ids = set(args.tx_ids or [])

    scene, payload = _load_payload(args.scene_id, args.tx_ids, args.boundary_json)
    if not highlight_tx_ids:
        highlight_tx_ids = {int(boundary["tx_id"]) for boundary in payload["boundaries"]}

    fig, ax = plt.subplots(figsize=tuple(args.figsize), constrained_layout=True)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    _plot_scene(ax, scene, highlight_tx_ids)
    counts = _plot_boundaries(ax, payload["boundaries"], highlight_tx_ids)
    _build_legend(ax, counts)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Scene {payload['scene_id']} Boundary Visualization", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#CBD5E1", linewidth=0.5, alpha=0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    print(f"scene_id: {payload['scene_id']}")
    print(f"boundary_count: {payload['boundary_count']}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
