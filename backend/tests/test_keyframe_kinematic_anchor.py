"""
Regression: kinematic top/impact anchoring and monotonic phase maps.

Uses synthetic pose streams (no video). Validates that refine_phase_keyframes_top_impact
keeps impact after top and that get_phase_keyframes returns ordered indices.

Run: python3 -m unittest backend.tests.test_keyframe_kinematic_anchor -v
"""

import unittest
import numpy as np

from services.swing_flow_utils import (
    get_phase_keyframes,
    refine_phase_keyframes_top_impact,
    detect_swing_phases,
)


def _joint(name: str, x: float, y: float, vis: float = 0.92) -> dict:
    return {
        "name": name,
        "visibility": vis,
        "normalized": {"x": x, "y": y, "z": 0.0},
    }


def _make_swing_poses(n: int = 48) -> list[dict]:
    """Synthetic side-view-ish wrist path: setup high, backswing up (low y), downswing to impact."""
    poses = []
    t = np.linspace(0.0, 1.6, n)
    for i in range(n):
        frac = i / max(n - 1, 1)
        # Wrist Y: address ~0.55, top ~0.22 (hands high in frame), impact ~0.48, finish ~0.35
        if frac < 0.12:
            wy = 0.55 + 0.02 * np.sin(frac * 40)
        elif frac < 0.48:
            wy = 0.55 - 0.55 * ((frac - 0.12) / 0.36)
        elif frac < 0.62:
            wy = 0.20 + 0.35 * ((frac - 0.48) / 0.14)
        else:
            wy = 0.55 - 0.12 * ((frac - 0.62) / 0.38)

        wx = 0.48 + 0.08 * np.sin(frac * np.pi * 2.1)
        xf = 15.0 + 55.0 * min(frac / 0.5, 1.0) - 20.0 * max(0.0, (frac - 0.55) / 0.45)
        sr = 8.0 + 40.0 * min(frac / 0.52, 1.0)

        joints = [
            _joint("left_shoulder", 0.42, 0.32),
            _joint("right_shoulder", 0.58, 0.32),
            _joint("left_hip", 0.45, 0.62),
            _joint("right_hip", 0.55, 0.62),
            _joint("left_wrist", wx - 0.01, wy),
            _joint("right_wrist", wx + 0.01, wy),
            _joint("left_knee", 0.46, 0.78),
            _joint("right_knee", 0.54, 0.78),
        ]
        poses.append({
            "frame_index": i * 2,
            "timestamp": float(t[i]),
            "joints": joints,
            "angles": {
                "x_factor": float(np.clip(xf, 5.0, 55.0)),
                "shoulder_rotation": float(sr),
                "spine_tilt": 9.0,
                "left_elbow": 155.0,
                "right_elbow": 135.0,
                "left_knee": 165.0,
                "right_knee": 168.0,
            },
        })
    return poses


class TestKeyframeKinematicAnchor(unittest.TestCase):
    def test_refine_phase_keyframes_keeps_impact_after_top(self):
        poses = _make_swing_poses(52)
        pk = {p: i * 3 for i, p in enumerate(
            ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]
        )}
        refine_phase_keyframes_top_impact(poses, pk)
        self.assertLess(pk["top"], pk["impact"])
        self.assertGreaterEqual(pk["top"], 0)
        self.assertLess(pk["top"], len(poses) - 3)
        self.assertLessEqual(pk["impact"], len(poses) - 1)

    def test_get_phase_keyframes_monotonic_and_top_before_impact(self):
        poses = _make_swing_poses(50)
        phases = detect_swing_phases(poses)
        result = get_phase_keyframes(phases, poses)
        order = [
            "address", "takeaway", "backswing", "top",
            "downswing", "impact", "follow_through", "finish",
        ]
        prev = -1
        for pid in order:
            idx = result[pid]
            self.assertGreater(idx, prev, f"phase {pid} idx={idx} prev={prev}")
            prev = idx
        self.assertGreater(result["impact"], result["top"])

    def test_refine_pulls_toward_kinematic_window_when_map_is_stale(self):
        poses = _make_swing_poses(56)
        pk = {
            "address": 2,
            "takeaway": 8,
            "backswing": 14,
            "top": 10,
            "downswing": 30,
            "impact": 40,
            "follow_through": 48,
            "finish": 54,
        }
        refine_phase_keyframes_top_impact(poses, pk)
        self.assertGreater(pk["impact"], pk["top"])
        self.assertGreater(pk["top"], 10)


if __name__ == "__main__":
    unittest.main()
