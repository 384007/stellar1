"""Sweet spot window median robustness (synthetic)."""
import sys
import unittest
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.shot_predictor import estimate_sweet_spot_robust


def _pose(wrist_x: float, wrist_y: float, vis: float = 0.9) -> dict:
    return {
        "frame_index": 0,
        "timestamp": 0.0,
        "joints": [
            {"name": "left_shoulder", "visibility": 0.9, "normalized": {"x": 0.42, "y": 0.32}},
            {"name": "right_shoulder", "visibility": 0.9, "normalized": {"x": 0.58, "y": 0.32}},
            {"name": "left_elbow", "visibility": 0.9, "normalized": {"x": 0.44, "y": 0.42}},
            {"name": "right_elbow", "visibility": 0.9, "normalized": {"x": 0.56, "y": 0.42}},
            {"name": "left_wrist", "visibility": vis, "normalized": {"x": wrist_x - 0.01, "y": wrist_y}},
            {"name": "right_wrist", "visibility": vis, "normalized": {"x": wrist_x + 0.01, "y": wrist_y}},
            {"name": "left_hip", "visibility": 0.9, "normalized": {"x": 0.45, "y": 0.62}},
            {"name": "right_hip", "visibility": 0.9, "normalized": {"x": 0.55, "y": 0.62}},
        ],
    }


class TestSweetSpotWindowRobustness(unittest.TestCase):
    def test_sweet_spot_window_robustness(self):
        base_x, base_y = 0.5, 0.48
        poses = []
        for _ in range(5):
            poses.append(_pose(base_x, base_y))
        impact = 2
        poses[impact] = _pose(0.95, 0.95, vis=0.9)
        r = estimate_sweet_spot_robust(poses, impact, window=2, hand="R")
        mx = r["sweet_spot"]["nx"]
        self.assertLess(abs(mx - base_x), 0.12)
        self.assertGreaterEqual(r["sweet_spot_valid_frames"], 2)

    def test_default_wider_window_median_resists_outlier(self):
        """Default SWEET_SPOT_WINDOW=4: more frames → median stable vs one bad impact frame."""
        base_x, base_y = 0.5, 0.48
        n = 15
        impact = 7
        poses = [_pose(base_x, base_y) for _ in range(n)]
        poses[impact] = _pose(0.95, 0.95, vis=0.9)
        r = estimate_sweet_spot_robust(poses, impact, hand="R")
        self.assertEqual(r["sweet_spot_window_size"], 9)
        mx = r["sweet_spot"]["nx"]
        self.assertLess(abs(mx - base_x), 0.12)
        self.assertGreaterEqual(r["sweet_spot_valid_frames"], 4)

    def test_single_valid_frame_soft_reason_not_unstable(self):
        poses = [_pose(0.52, 0.47, vis=0.9)]
        r = estimate_sweet_spot_robust(poses, 0, window=0, hand="R")
        self.assertEqual(r["sweet_spot_valid_frames"], 1)
        self.assertFalse(r["sweet_spot_unstable"])
        self.assertIn("SWEET_SPOT_LOW_VALID_FRAMES", r["sweet_spot_reasons"])
        self.assertLess(r["sweet_spot_confidence"], 0.35)


if __name__ == "__main__":
    unittest.main()
