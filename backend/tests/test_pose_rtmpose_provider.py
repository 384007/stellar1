"""Unit tests for RTMPose COCO→stellar mapping (no MMPose runtime required)."""

from __future__ import annotations

import unittest

import numpy as np

from services.providers.pose_rtmpose_provider import _parse_inferencer_batch, coco17_to_stellar_joints


class TestPoseRtmposeProvider(unittest.TestCase):
    def test_coco17_maps_golf_names_and_wrist(self) -> None:
        k = np.zeros((17, 2), dtype=np.float64)
        k[9] = [100.0, 200.0]
        s = np.ones(17, dtype=np.float64) * 0.9
        joints, _conn, angles = coco17_to_stellar_joints(k, s, 640, 480)
        names = [j["name"] for j in joints]
        self.assertEqual(names[0], "head")
        lw = next(j for j in joints if j["name"] == "left_wrist")
        self.assertAlmostEqual(lw["x"], 100.0)
        self.assertAlmostEqual(lw["normalized"]["x"], 100.0 / 640.0, places=4)
        self.assertIn("left_elbow", angles)

    def test_parse_predictions_nested_list(self) -> None:
        kpt = np.random.default_rng(0).random((17, 2)) * 200.0
        out = {"predictions": [[{"keypoints": kpt, "keypoint_scores": np.ones(17, dtype=np.float64) * 0.8}]]}
        p = _parse_inferencer_batch(out)
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p[0].shape, (17, 2))
        self.assertEqual(len(p[1]), 17)

    def test_parse_empty_returns_none(self) -> None:
        self.assertIsNone(_parse_inferencer_batch({"predictions": [[]]}))


if __name__ == "__main__":
    unittest.main()
