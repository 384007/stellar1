import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _ensure_mediapipe_stub_if_missing() -> None:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        if "mediapipe" in sys.modules:
            return
        m = mock.MagicMock()
        sol = mock.MagicMock()
        sol.pose = mock.MagicMock()
        sol.drawing_utils = mock.MagicMock()
        m.solutions = sol
        sys.modules["mediapipe"] = m


_ensure_mediapipe_stub_if_missing()


def _ensure_cv2_stub_if_missing() -> None:
    try:
        import cv2  # noqa: F401
    except Exception:
        if "cv2" in sys.modules:
            return
        sys.modules["cv2"] = mock.MagicMock()


_ensure_cv2_stub_if_missing()

from services.pose_service import get_render_joints, pose_for_skeleton_render

import services.keyframe_service as keyframe_service
import services.swing_flow_utils as swing_flow_utils


class TestPoseJointSpaceConsistency(unittest.TestCase):
    def test_render_joints_prefer_analysis_space(self):
        pose = {
            "joints": [{"name": "left_wrist", "x": 10, "y": 20, "normalized": {"x": 0.1, "y": 0.2}}],
            "raw_detection_joints": [{"name": "left_wrist", "x": 111, "y": 222, "normalized": {"x": 0.9, "y": 0.9}}],
            "detection": {"joints": [{"name": "left_wrist", "x": 333, "y": 444}]},
        }
        joints = get_render_joints(pose)
        self.assertEqual(joints[0]["x"], 10)
        self.assertEqual(joints[0]["y"], 20)

    def test_pose_for_skeleton_render_exposes_joint_sources(self):
        pose = {
            "joints": [{"name": "left_wrist", "x": 8, "y": 18, "normalized": {"x": 0.08, "y": 0.18}}],
            "angles": {"x_factor": 12.3},
            "raw_detection_joints": [{"name": "left_wrist", "x": 80, "y": 180, "normalized": {"x": 0.8, "y": 0.8}}],
            "detection": {"angles": {"x_factor": 40.0}},
        }
        out = pose_for_skeleton_render(pose)
        self.assertEqual(out["joint_space"], "analysis_frame")
        self.assertEqual(out["joints"][0]["x"], 8)
        self.assertEqual(out["analysis_joints"][0]["x"], 8)
        self.assertEqual(out["raw_detection_joints"][0]["x"], 80)


class TestPhaseKeyframeSelectionConsistency(unittest.TestCase):
    def test_get_phase_keyframes_prefers_semantically_stronger_candidate(self):
        poses = [{"frame_index": i, "timestamp": i / 30.0, "joints": [], "angles": {}} for i in range(40)]
        bucket_map = {p: i * 4 for i, p in enumerate(swing_flow_utils._PHASE_IDS)}
        anchor_map = dict(bucket_map)
        anchor_map["top"] = anchor_map["impact"] + 1  # intentionally semantically worse

        def _fake_validate(kf_map, _poses, source="unknown"):
            if source == "anchor":
                return {"passed": False, "issues": ["impact_before_top"], "min_gap": 0}
            return {"passed": True, "issues": [], "min_gap": 4}

        with mock.patch.object(swing_flow_utils, "_get_phase_keyframes_bucket_driven", return_value=dict(bucket_map)), \
             mock.patch.object(swing_flow_utils, "build_phase_keyframes_from_top_impact_anchors", return_value=(dict(anchor_map), True, {})), \
             mock.patch.object(swing_flow_utils, "refine_phase_keyframes_top_impact", side_effect=lambda _p, _k: None), \
             mock.patch.object(swing_flow_utils, "validate_phase_keyframes", side_effect=_fake_validate):
            got = swing_flow_utils.get_phase_keyframes([], poses)

        self.assertEqual(got, bucket_map)


class TestEnsureKeyframeRepairPath(unittest.TestCase):
    def test_gate_fail_then_anchor_repair_returns_smart_repaired(self):
        poses = [{"frame_index": i * 3, "timestamp": i / 30.0, "joints": [], "angles": {}} for i in range(24)]
        keyframes = [
            {
                "phase": ph,
                "source_pose_idx": i,
                "source_frame_index": i * 3,
                "frame_index": i * 3,
                "timestamp": i / 30.0,
            }
            for i, ph in enumerate(keyframe_service.PHASE_ORDER)
        ]
        phase_map = {ph: i for i, ph in enumerate(keyframe_service.PHASE_ORDER)}
        kfv = {"details": []}
        first_gate = {
            "pass": False,
            "final_keyframe_order_ok": False,
            "final_keyframe_time_order_ok": False,
            "final_phase_keyframes_sync_ok": False,
            "negative_time_gap_in_details": False,
        }
        repaired_gate = {
            "pass": True,
            "final_keyframe_order_ok": True,
            "final_keyframe_time_order_ok": True,
            "final_phase_keyframes_sync_ok": True,
            "negative_time_gap_in_details": False,
        }
        rebuilt = {ph: i + 1 for i, ph in enumerate(keyframe_service.PHASE_ORDER)}
        rebuilt_kf = [dict(k, source_pose_idx=rebuilt[k["phase"]], source_frame_index=rebuilt[k["phase"]] * 3) for k in keyframes]
        rebuilt_det = [{"phase": k["phase"], "validation_passed": True} for k in rebuilt_kf]

        with mock.patch.object(keyframe_service, "validate_final_keyframes_for_ai", side_effect=[first_gate, repaired_gate]), \
             mock.patch.object(keyframe_service, "rebuild_phase_map_from_event_anchors", return_value={"rebuild_ok": True, "phase_keyframes_rebuilt": rebuilt, "top_reselected": True, "impact_reselected": True}), \
             mock.patch.object(keyframe_service, "_rebind_keyframes_from_rebuilt_map", return_value=(rebuilt_kf, rebuilt_det)), \
             mock.patch.object(keyframe_service.cv2, "VideoCapture") as cap_mock, \
             mock.patch.object(keyframe_service, "get_video_rotation", return_value=0):
            cap_mock.return_value.get.return_value = 30.0
            out_kf, merged, out_phase, final_src = keyframe_service.ensure_keyframes_ordered_for_ai(
                "dummy.mp4", poses, [], dict(phase_map), list(keyframes), dict(kfv), dict(phase_map), 320
            )

        self.assertEqual(final_src, "smart_repaired")
        self.assertTrue(merged["final_keyframe_gate_pass"])
        self.assertEqual(out_phase, rebuilt)
        self.assertEqual(out_kf, rebuilt_kf)


class TestNoFakeRepairFlags(unittest.TestCase):
    def test_failed_repair_clears_reselected_flags(self):
        merged = keyframe_service._merge_keyframe_validation_with_repairs(
            {"strict_contract_ok": False, "semantic_strip_ok": True, "relabel_count": 0},
            {"top_reselected": True, "impact_reselected": True, "reselected_top": True, "reselected_impact": True},
        )
        self.assertFalse(merged["pass"])
        self.assertFalse(merged["top_reselected"])
        self.assertFalse(merged["impact_reselected"])
        self.assertFalse(merged["reselected_top"])
        self.assertFalse(merged["reselected_impact"])


class TestImpactSemanticHardConstraint(unittest.TestCase):
    def test_impact_requires_unwinding_and_strike_zone(self):
        kin = {
            "kinematic_fail_codes": [],
            "n": 40,
            "valid": np.array([True] * 40),
            "speed_s": np.array([0.2] * 20 + [1.2] * 20),
            "xf_d": np.array([0.1] * 40),  # no unwinding (needs negative)
            "hand_hip": np.array([0.1] * 40),
        }
        ok, checks = swing_flow_utils.validate_impact_semantic_at_index(28, 12, 10, kin)
        self.assertFalse(ok)
        self.assertFalse(checks["unwinding"])


class TestMiddlePhaseGapContract(unittest.TestCase):
    def test_middle_phase_cluster_fails_source_frame_gap_contract(self):
        kfs = []
        for i, ph in enumerate(keyframe_service.PHASE_ORDER):
            fi = 100 + i  # intentionally too close
            kfs.append(
                {
                    "phase": ph,
                    "source_pose_idx": i,
                    "source_frame_index": fi,
                    "frame_index": fi,
                    "timestamp": i / 30.0,
                }
            )
        pk = {ph: i for i, ph in enumerate(keyframe_service.PHASE_ORDER)}
        g = keyframe_service.validate_final_keyframes_for_ai(kfs, pk, [], poses=None, fps=30.0)
        self.assertFalse(g["strict_contract_checks"]["source_frame_gaps_ok"])
        self.assertTrue(any(str(r).startswith("MIN_GAP_VIOLATION:TAKEAWAY_BACKSWING") for r in g["source_frame_gap_reasons"]))


if __name__ == "__main__":
    unittest.main()
