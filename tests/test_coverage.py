from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import rt2d
from rt2d.path_family import build_path_family_runtime, compute_rx_partition_runtime
from rt2d.coverage import SequenceCostConfig, build_energy_pruned_sequence_render_grid, build_layered_sequence_render_grid


def _label_at(payload: dict[str, object], tx_index: int, x: float, y: float) -> int:
    grid = payload["grid"]
    tx_result = payload["tx_results"][tx_index]

    step = float(grid["step"])
    min_x = float(grid["min_x"])
    max_y = float(grid["max_y"])
    col = int(round((x - min_x) / step))
    row = int(round((max_y - y) / step))
    return int(tx_result["visibility_order_grid"][row][col])


class RxCoverageTests(unittest.TestCase):
    def test_runtime_prefers_building_mask_png_when_provided(self) -> None:
        scene = {
            "scene_id": "runtime-mask",
            "antenna": [[0.0, 0.0]],
            "polygons": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "building_mask.png"
            image = Image.new("L", (3, 3), 0)
            image.putpixel((1, 1), 255)
            image.save(mask_path)
            scene["building_mask_path"] = str(mask_path)

            runtime = rt2d.build_rx_visibility_runtime(
                scene,
                bounds=(0.0, 0.0, 2.0, 2.0),
                acceleration_backend="cpu",
            )

            self.assertEqual(runtime.outdoor_mask_source, "building_png")
            self.assertFalse(runtime.outdoor_mask[1][1])
            self.assertTrue(runtime.outdoor_mask[0][0])

    def test_path_family_los_r_d_partition_runs_on_small_scene(self) -> None:
        scene = {
            "scene_id": "path-family-preview",
            "antenna": [[-2.0, 1.0]],
            "polygons": [
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
            ],
        }

        runtime = build_path_family_runtime(
            scene,
            tx_id=0,
            bounds=(-3.0, -1.0, 6.0, 6.0),
            acceleration_backend="cpu",
        )
        payload = compute_rx_partition_runtime(runtime)

        labels = {label for row in payload["partition_grid"] for label in row}
        self.assertIn("L", labels)
        self.assertIn("D", labels)
        self.assertTrue(labels.issubset({"blocked", "unreachable", "L", "R", "D"}))

    def test_path_family_runtime_uses_radio_map_building_png_when_available(self) -> None:
        root = Path(r"D:\TheSonder\0.8 Resources\data\RadioMapSeer")
        if not (root / "png" / "buildings_complete" / "0.png").is_file():
            self.skipTest("RadioMapSeer buildings_complete png not available")

        scene = {
            "scene_id": "0",
            "root_dir": str(root),
            "antenna_path": str(root / "antenna" / "0.json"),
            "polygon_path": str(root / "polygon" / "buildings_complete" / "0.json"),
            "antenna": [[108.0, 68.0]],
            "polygons": [],
            "building_mask_path": str(root / "png" / "buildings_complete" / "0.png"),
        }
        runtime = build_path_family_runtime(
            scene,
            tx_id=0,
            bounds=(0.0, 0.0, 255.0, 255.0),
            acceleration_backend="cpu",
        )
        self.assertEqual(runtime.rx_runtime.outdoor_mask_source, "building_png")

    def test_runtime_builder_matches_direct_compute_on_small_scene(self) -> None:
        scene = {
            "scene_id": "runtime-compare",
            "antenna": [[-2.0, 2.0]],
            "polygons": [
                [[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
                [[4.0, -1.0], [6.0, -1.0], [6.0, 5.0], [4.0, 5.0], [4.0, -1.0]],
            ],
        }

        direct_payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=[0],
            max_interactions=2,
            bounds=(-3.0, -2.0, 8.0, 7.0),
            include_sequence_render_grid=True,
            include_sequence_hit_grids=True,
            acceleration_backend="cpu",
        )

        runtime = rt2d.build_rx_visibility_runtime(
            scene,
            bounds=(-3.0, -2.0, 8.0, 7.0),
            acceleration_backend="cpu",
        )
        runtime_payload = rt2d.compute_rx_visibility_runtime(
            runtime,
            tx_ids=[0],
            max_interactions=2,
            include_sequence_render_grid=True,
            include_sequence_hit_grids=True,
        )

        self.assertEqual(direct_payload["grid"], runtime_payload["grid"])
        self.assertEqual(direct_payload["tx_results"], runtime_payload["tx_results"])

    def test_runtime_warm_keeps_results_consistent(self) -> None:
        scene = {
            "scene_id": "runtime-warm",
            "antenna": [[-2.0, 2.0]],
            "polygons": [
                [[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
                [[4.0, -1.0], [6.0, -1.0], [6.0, 5.0], [4.0, 5.0], [4.0, -1.0]],
            ],
        }

        runtime = rt2d.build_rx_visibility_runtime(
            scene,
            bounds=(-3.0, -2.0, 8.0, 7.0),
            acceleration_backend="cpu",
        )
        rt2d.warm_rx_visibility_runtime(
            runtime,
            tx_ids=[0],
            max_interactions=2,
        )
        payload = rt2d.compute_rx_visibility_runtime(
            runtime,
            tx_ids=[0],
            max_interactions=2,
            include_sequence_render_grid=True,
            include_sequence_hit_grids=True,
        )
        self.assertEqual(_label_at(payload, 0, 8.0, 5.0), 2)

    def test_torch_backend_matches_cpu_on_small_scene(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")

        scene = {
            "scene_id": "torch-compare",
            "antenna": [[-2.0, 2.0]],
            "polygons": [
                [[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
                [[4.0, -1.0], [6.0, -1.0], [6.0, 5.0], [4.0, 5.0], [4.0, -1.0]],
            ],
        }

        cpu_payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=[0],
            max_interactions=2,
            bounds=(-3.0, -2.0, 8.0, 7.0),
            include_sequence_render_grid=True,
            include_sequence_hit_grids=True,
            acceleration_backend="cpu",
        )

        torch_payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=[0],
            max_interactions=2,
            bounds=(-3.0, -2.0, 8.0, 7.0),
            include_sequence_render_grid=True,
            include_sequence_hit_grids=True,
            acceleration_backend="torch",
            torch_device="cpu",
            torch_state_chunk_size=4,
            torch_point_chunk_size=64,
            torch_edge_chunk_size=8,
        )

        self.assertEqual(cpu_payload["grid"], torch_payload["grid"])
        self.assertEqual(cpu_payload["tx_results"][0]["counts"], torch_payload["tx_results"][0]["counts"])
        self.assertEqual(
            cpu_payload["tx_results"][0]["visibility_order_grid"],
            torch_payload["tx_results"][0]["visibility_order_grid"],
        )
        self.assertEqual(
            cpu_payload["tx_results"][0]["layered_sequence_grid"],
            torch_payload["tx_results"][0]["layered_sequence_grid"],
        )
        self.assertEqual(
            cpu_payload["tx_results"][0]["sequence_hit_grids"],
            torch_payload["tx_results"][0]["sequence_hit_grids"],
        )

    def test_layered_sequence_grid_prefers_first_order_reflection_over_first_order_diffraction(self) -> None:
        outdoor_mask = [[True]]
        sequence_hit_grids = {
            "L": [[0]],
            "D": [[1]],
            "R": [[1]],
        }

        grid, counts = build_layered_sequence_render_grid(sequence_hit_grids, outdoor_mask)

        self.assertEqual(grid[0][0], "R")
        self.assertEqual(counts["R"], 1)

    def test_layered_sequence_grid_allows_third_order_suffix_r_to_override_dd(self) -> None:
        outdoor_mask = [[True]]
        sequence_hit_grids = {
            "L": [[0]],
            "DD": [[1]],
            "DDR": [[1]],
        }

        grid, counts = build_layered_sequence_render_grid(sequence_hit_grids, outdoor_mask)

        self.assertEqual(grid[0][0], "DDR")
        self.assertEqual(counts["DDR"], 1)

    def test_layered_sequence_grid_keeps_dd_when_only_third_order_suffix_d_exists(self) -> None:
        outdoor_mask = [[True]]
        sequence_hit_grids = {
            "L": [[0]],
            "DD": [[1]],
            "DDD": [[1]],
        }

        grid, counts = build_layered_sequence_render_grid(sequence_hit_grids, outdoor_mask)

        self.assertEqual(grid[0][0], "DD")
        self.assertEqual(counts["DD"], 1)

    def test_energy_pruned_sequence_grid_can_drop_high_cost_dd(self) -> None:
        outdoor_mask = [[True]]
        sequence_hit_grids = {
            "L": [[0]],
            "DD": [[1]],
        }
        grid_meta = {
            "min_x": 0.0,
            "max_y": 200.0,
            "step": 1.0,
        }

        grid, counts = build_energy_pruned_sequence_render_grid(
            sequence_hit_grids,
            outdoor_mask,
            (0.0, 0.0),
            grid_meta,
            SequenceCostConfig(max_cost=50.0),
        )

        self.assertEqual(grid[0][0], "unreachable")
        self.assertEqual(counts["unreachable"], 1)

    def test_empty_scene_marks_all_grid_points_as_los(self) -> None:
        scene = {
            "scene_id": "empty-grid",
            "antenna": [[0.0, 0.0]],
            "polygons": [],
        }

        payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=[0],
            max_interactions=0,
            bounds=(-1.0, -1.0, 1.0, 1.0),
        )

        counts = payload["tx_results"][0]["counts"]
        self.assertEqual(counts["los"], 9)
        self.assertEqual(counts["blocked"], 0)
        self.assertEqual(counts["unreachable"], 0)
        self.assertEqual(_label_at(payload, 0, 1.0, 1.0), 0)

    def test_first_order_reflection_marks_shadow_points(self) -> None:
        scene = {
            "scene_id": "left-wall",
            "antenna": [[-2.0, 2.0]],
            "polygons": [[[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0], [0.0, 0.0]]],
        }

        payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=[0],
            max_interactions=1,
            bounds=(-3.0, -1.0, 6.0, 6.0),
            enable_diffraction=False,
        )

        counts = payload["tx_results"][0]["counts"]
        self.assertGreater(counts["order1"], 0)
        self.assertEqual(_label_at(payload, 0, 6.0, 6.0), 1)
        self.assertEqual(_label_at(payload, 0, 2.0, 2.0), -2)

    def test_first_order_diffraction_marks_corner_shadow_points(self) -> None:
        scene = {
            "scene_id": "diff-rect",
            "antenna": [[-2.0, 1.0]],
            "polygons": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
        }

        payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=[0],
            max_interactions=1,
            bounds=(-3.0, -1.0, 4.0, 4.0),
            enable_reflection=False,
        )

        counts = payload["tx_results"][0]["counts"]
        self.assertGreater(counts["order1"], 0)
        self.assertEqual(_label_at(payload, 0, 4.0, 4.0), 1)
        self.assertEqual(_label_at(payload, 0, 4.0, 0.0), -1)

    def test_second_order_reflection_only_marks_after_first_order(self) -> None:
        scene = {
            "scene_id": "left-wall-2",
            "antenna": [[-2.0, 2.0]],
            "polygons": [
                [[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
                [[4.0, -1.0], [6.0, -1.0], [6.0, 5.0], [4.0, 5.0], [4.0, -1.0]],
            ],
        }

        payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=[0],
            max_interactions=2,
            bounds=(-3.0, -2.0, 8.0, 7.0),
            enable_diffraction=False,
        )

        counts = payload["tx_results"][0]["counts"]
        self.assertGreater(counts["order1"], 0)
        self.assertGreater(counts["order2"], 0)
        self.assertEqual(_label_at(payload, 0, 3.0, 3.0), 1)
        self.assertEqual(_label_at(payload, 0, 8.0, 5.0), 2)

    def test_max_interactions_four_is_supported_for_rx_visibility(self) -> None:
        scene = {
            "scene_id": "left-wall-2",
            "antenna": [[-2.0, 2.0]],
            "polygons": [
                [[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
                [[4.0, -1.0], [6.0, -1.0], [6.0, 5.0], [4.0, 5.0], [4.0, -1.0]],
            ],
        }

        payload = rt2d.compute_rx_visibility(
            scene,
            tx_ids=[0],
            max_interactions=4,
            bounds=(-3.0, -2.0, 8.0, 7.0),
        )

        state_counts = payload["tx_results"][0]["state_counts"]
        self.assertEqual(set(state_counts.keys()), {"order1", "order2", "order3", "order4"})
        labels = {
            value
            for row in payload["tx_results"][0]["visibility_order_grid"]
            for value in row
        }
        self.assertTrue(labels.issubset({-2, -1, 0, 1, 2, 3, 4}))


if __name__ == "__main__":
    unittest.main()
