from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import rt2d
from rt2d.coverage import build_layered_sequence_render_grid


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
    def test_layered_sequence_grid_prefers_reflection_over_first_order_diffraction(self) -> None:
        outdoor_mask = [[True]]
        sequence_hit_grids = {
            "L": [[0]],
            "D": [[1]],
            "RR": [[1]],
        }

        grid, counts = build_layered_sequence_render_grid(sequence_hit_grids, outdoor_mask)

        self.assertEqual(grid[0][0], "RR")
        self.assertEqual(counts["RR"], 1)

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
