"""Unit tests for phase + MMAction2 fusion helpers."""

from __future__ import annotations

import unittest

from services.phase_action_fusion import (
    build_boundary_confidence,
    build_per_frame_phase_logits,
    fuse_kinematic_phases_with_action_priors,
    median_smooth_phase_labels,
    phase_confidence_summary,
    temporal_prior_strength,
)


class TestPhaseActionFusion(unittest.TestCase):
    def test_median_smooth_stable(self) -> None:
        raw = ["address", "address", "takeaway", "takeaway", "backswing", "backswing"]
        out = median_smooth_phase_labels(raw, k=3)
        self.assertEqual(len(out), len(raw))

    def test_logits_row_sums_reasonable(self) -> None:
        pfp = ["address", "impact", "finish"]
        rows = build_per_frame_phase_logits(
            pfp,
            action_clip_confidence=0.5,
            window_peak_confidence=0.4,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(rows[1]), 8)
        self.assertGreater(rows[1][5], rows[1][0])  # impact index 5

    def test_temporal_prior_strength_windows(self) -> None:
        w = [{"confidence": 0.8}, {"confidence": 0.6}]
        t = temporal_prior_strength(0.3, w)
        self.assertGreater(t, temporal_prior_strength(0.3, []))

    def test_boundary_confidence_boost(self) -> None:
        bds = [{"phase_id": "a", "start_idx": 0, "end_idx": 2}]
        low = build_boundary_confidence(bds, temporal_boost=0.0)
        high = build_boundary_confidence(bds, temporal_boost=0.15)
        self.assertGreater(high[0]["confidence"], low[0]["confidence"])

    def test_phase_confidence_summary(self) -> None:
        s = phase_confidence_summary(
            action_clip_confidence=0.5,
            window_predictions=[{"confidence": 0.7}],
            boundary_confidences=[{"confidence": 0.8}],
        )
        self.assertIn("global_segmentation_confidence", s)
        self.assertEqual(s["window_count"], 1)

    def test_fuse_windows_overrides_range(self) -> None:
        kin = ["address"] * 10
        windows = [
            {
                "label": "my_downswing_class",
                "confidence": 0.9,
                "pose_start_idx": 3,
                "pose_end_idx": 6,
            },
        ]
        out = fuse_kinematic_phases_with_action_priors(
            kin,
            window_predictions=windows,
            label_map={"my_downswing_class": "downswing"},
        )
        self.assertEqual(out[:3], ["address", "address", "address"])
        self.assertEqual(out[3:7], ["downswing", "downswing", "downswing", "downswing"])
        self.assertEqual(out[7:], ["address", "address", "address"])

    def test_fuse_without_map_unchanged(self) -> None:
        kin = ["takeaway", "backswing", "top"]
        out = fuse_kinematic_phases_with_action_priors(
            kin,
            window_predictions=[{"label": "kinetics_xyz", "confidence": 0.99, "pose_start_idx": 0, "pose_end_idx": 2}],
            label_map=None,
        )
        self.assertEqual(out, kin)

    def test_fuse_clip_span_when_no_windows(self) -> None:
        kin = ["address"] * 20
        out = fuse_kinematic_phases_with_action_priors(
            kin,
            window_predictions=[],
            action_label="clip_top",
            action_confidence=0.5,
            label_map={"clip_top": "top"},
            min_clip_confidence=0.45,
        )
        self.assertIn("top", out)
        self.assertIn("address", out)


if __name__ == "__main__":
    unittest.main()
