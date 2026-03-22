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

        payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0])
        los = [item for item in payload["boundaries"] if item["type"] == "los"]
        reflection = [item for item in payload["boundaries"] if item["type"] == "reflection"]
        diffraction = [item for item in payload["boundaries"] if item["type"] == "diffraction"]

        self.assertGreaterEqual(len(los), 2)
        self.assertGreaterEqual(len(reflection), 2)
        self.assertGreaterEqual(len(diffraction), 2)

        self.assertTrue(any(_approx_point(item["p0"], (0.0, 0.0)) for item in los))
        self.assertTrue(any(_approx_point(item["p0"], (4.0, 0.0)) for item in los))
        self.assertTrue(any(_approx_point(item["p0"], (0.0, 0.0)) for item in reflection))
        self.assertTrue(any(_approx_point(item["p0"], (4.0, 0.0)) for item in reflection))

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

        payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0])
        reflection = [item for item in payload["boundaries"] if item["type"] == "reflection"]
        self.assertTrue(any(abs(item["p0"][1] - 0.5) < 0.15 for item in reflection))

    def test_scene_zero_and_one_export_json(self) -> None:
        for scene_id in ("0", "1"):
            payload = rt2d.extract_scene_boundaries(scene_id, tx_ids=[0])
            self.assertEqual(payload["scene_id"], scene_id)
            self.assertGreater(payload["boundary_count"], 0)
            sample = payload["boundaries"][0]
            self.assertEqual(
                set(sample.keys()),
                {"type", "p0", "p1", "source", "mechanism", "scene_id", "tx_id"},
            )


if __name__ == "__main__":
    unittest.main()



