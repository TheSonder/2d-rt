from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import rt2d
from rt2d.boundary import build_geometry, compute_visible_subsegments


def _approx_point(a: list[float], b: tuple[float, float], places: int = 4) -> bool:
    return round(a[0] - b[0], places) == 0 and round(a[1] - b[1], places) == 0


class BoundaryExtractionTests(unittest.TestCase):
    def test_single_rectangle_outputs_los_and_reflection_boundaries(self) -> None:
        scene = {
            "scene_id": "unit-rect",
            "antenna": [[2.0, -2.0]],
            "polygons": [[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
        }

        payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=1)
        los = [item for item in payload["boundaries"] if item["type"] == "los"]
        reflection = [item for item in payload["boundaries"] if item["type"] == "reflection"]

        self.assertGreaterEqual(len(los), 2)
        self.assertGreaterEqual(len(reflection), 2)
        self.assertTrue(any(_approx_point(item["p0"], (0.0, 0.0)) for item in los))
        self.assertTrue(any(_approx_point(item["p0"], (4.0, 0.0)) for item in los))
        self.assertTrue(any(_approx_point(item["p0"], (0.0, 0.0)) for item in reflection))
        self.assertTrue(any(_approx_point(item["p0"], (4.0, 0.0)) for item in reflection))
        self.assertTrue(all(item["sequence"] == "L" for item in los))
        self.assertTrue(all(item["sequence"] == "R" for item in reflection))


    def test_partial_occlusion_clips_visible_subsegment(self) -> None:
        scene = {
            "scene_id": "partial-occlusion",
            "antenna": [[0.0, -1.0]],
            "polygons": [
                [[6.0, -2.0], [8.0, -2.0], [8.0, 2.0], [6.0, 2.0], [6.0, -2.0]],
                [[3.0, 0.0], [4.0, 0.0], [4.0, 3.0], [3.0, 3.0], [3.0, 0.0]],
            ],
        }

        geom = build_geometry(scene)
        tx = geom.antennas[0]
        target_edge = next(
            edge
            for edge in geom.edges
            if edge.poly_id == 0 and math.isclose(edge.a[0], 6.0) and math.isclose(edge.b[0], 6.0)
        )

        visible = compute_visible_subsegments(tx, target_edge, geom)
        self.assertEqual(len(visible), 1)
        self.assertGreater(visible[0][0], 0.3)
        self.assertLess(visible[0][0], 0.5)
        self.assertAlmostEqual(visible[0][1], 1.0, places=4)

        payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=1)
        reflection = [item for item in payload["boundaries"] if item["type"] == "reflection"]
        self.assertTrue(any(abs(item["p0"][1] - 0.5) < 0.15 for item in reflection))

    def test_los_boundary_stops_at_next_building(self) -> None:
        scene = {
            "scene_id": "los-stop",
            "antenna": [[1.0, -2.0]],
            "polygons": [
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
                [[3.5, 3.5], [4.5, 3.5], [4.5, 4.5], [3.5, 4.5], [3.5, 3.5]],
            ],
        }

        payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=1)
        los = [item for item in payload["boundaries"] if item["type"] == "los"]
        target = next(item for item in los if _approx_point(item["p0"], (2.0, 0.0)))

        self.assertGreaterEqual(target["p1"][0], 3.5)
        self.assertLessEqual(target["p1"][0], 4.5)
        self.assertAlmostEqual(target["p1"][1], 3.5, places=4)

    def test_reflection_boundary_stops_at_next_building(self) -> None:
        scene = {
            "scene_id": "reflection-stop",
            "antenna": [[2.0, -2.0]],
            "polygons": [
                [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
                [[5.0, -1.0], [6.0, -1.0], [6.0, 2.0], [5.0, 2.0], [5.0, -1.0]],
            ],
        }

        payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=1)
        reflection = [item for item in payload["boundaries"] if item["type"] == "reflection"]
        target = next(item for item in reflection if _approx_point(item["p0"], (4.0, 0.0)))

        self.assertAlmostEqual(target["p1"][0], 5.0, places=4)
        self.assertAlmostEqual(target["p1"][1], -1.0, places=4)

    def test_reflection_tube_outputs_shadow_boundaries_on_occluder(self) -> None:
        scene = {
            "scene_id": "r-shadow",
            "antenna": [[2.0, 4.0]],
            "polygons": [
                [[0.0, 12.0], [12.0, 12.0], [12.0, 15.0], [0.0, 15.0], [0.0, 12.0]],
                [[10.0, 6.0], [14.0, 6.0], [14.0, 8.0], [10.0, 8.0], [10.0, 6.0]],
            ],
        }

        payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=1)
        reflection = [item for item in payload["boundaries"] if item["sequence"] == "R"]
        occluder_boundaries = [item for item in reflection if item["source"]["poly_id"] == 1]

        self.assertGreaterEqual(len(occluder_boundaries), 2)
        self.assertTrue(any(_approx_point(item["p0"], (10.0, 6.0)) for item in occluder_boundaries))
        self.assertTrue(any(_approx_point(item["p0"], (14.0, 6.0)) for item in occluder_boundaries))

    def test_interior_departure_vertex_is_filtered(self) -> None:
        scene = {
            "scene_id": "interior-departure",
            "antenna": [[-2.0, 4.0]],
            "polygons": [
                [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
            ],
        }

        payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=1)
        los = [item for item in payload["boundaries"] if item["type"] == "los"]
        self.assertFalse(any(_approx_point(item["p0"], (0.0, 2.0)) for item in los))

    def test_max_interactions_two_only_adds_rr(self) -> None:
        scene = {
            "scene_id": "two-hop",
            "antenna": [[2.0, -2.0]],
            "polygons": [
                [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
                [[5.0, -1.0], [6.0, -1.0], [6.0, 3.0], [5.0, 3.0], [5.0, -1.0]],
                [[8.0, 0.0], [10.0, 0.0], [10.0, 3.0], [8.0, 3.0], [8.0, 0.0]],
            ],
        }

        payload_one = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=1)
        seq_one = {item["sequence"] for item in payload_one["boundaries"]}
        self.assertEqual(seq_one, {"L", "R"})

        payload_two = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=2)
        seq_two = {item["sequence"] for item in payload_two["boundaries"]}
        self.assertTrue({"L", "R", "RR"}.issubset(seq_two))
        self.assertFalse(any("D" in sequence for sequence in seq_two))

    def test_max_interactions_four_accepts_higher_order_reflection_expansion(self) -> None:
        payload = rt2d.extract_scene_boundaries("0", tx_ids=[0], max_interactions=4)
        sequences = {item["sequence"] for item in payload["boundaries"]}

        self.assertIn("RR", sequences)
        self.assertTrue(all(set(sequence) <= {"L", "R"} for sequence in sequences))
        self.assertLessEqual(max(len(sequence) for sequence in sequences), 4)

    def test_invalid_tx_id_raises_value_error(self) -> None:
        scene = {
            "scene_id": "invalid-tx",
            "antenna": [[2.0, -2.0]],
            "polygons": [[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
        }

        with self.assertRaises(ValueError):
            rt2d.extract_scene_boundaries(scene, tx_ids=[-1], max_interactions=1)

    def test_scene_zero_and_one_export_json(self) -> None:
        for scene_id in ("0", "1"):
            payload = rt2d.extract_scene_boundaries(scene_id, tx_ids=[0], max_interactions=2)
            self.assertEqual(payload["scene_id"], scene_id)
            self.assertGreater(payload["boundary_count"], 0)
            sample = payload["boundaries"][0]
            self.assertEqual(
                set(sample.keys()),
                {"type", "p0", "p1", "source", "mechanism", "sequence", "role", "scene_id", "tx_id"},
            )


if __name__ == "__main__":
    unittest.main()



