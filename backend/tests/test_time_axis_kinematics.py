"""Time axis monotonicity, safe gradients, and stable semantic failure reasons."""
import math
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.keyframe_service import PHASE_ORDER, SWING_PHASE_META, verify_phase_strip_semantics
from services.phase_analysis_gate import should_run_phase_analysis_strict
from services.swing_flow_utils import (
    build_semantic_phase_report,
    detect_phase_events_agnostic,
    safe_gradient,
)


def _golf_pose(i: int, timestamp: float) -> dict:
    """Minimal full-body pose for kinematics (duplicate ts allowed)."""
    fi = float(i)
    return {
        "frame_index": i,
        "timestamp": timestamp,
        "frame_size": {"width": 640, "height": 360},
        "joints": [
            {"name": "left_shoulder", "visibility": 0.9, "normalized": {"x": 0.42, "y": 0.32}},
            {"name": "right_shoulder", "visibility": 0.9, "normalized": {"x": 0.58, "y": 0.32}},
            {"name": "left_hip", "visibility": 0.9, "normalized": {"x": 0.45, "y": 0.62}},
            {"name": "right_hip", "visibility": 0.9, "normalized": {"x": 0.55, "y": 0.62}},
            {"name": "left_wrist", "visibility": 0.85, "normalized": {"x": 0.48 + fi * 0.001, "y": 0.48}},
            {"name": "right_wrist", "visibility": 0.85, "normalized": {"x": 0.52 + fi * 0.001, "y": 0.48}},
            {"name": "left_knee", "visibility": 0.85, "normalized": {"x": 0.46, "y": 0.72}},
            {"name": "right_knee", "visibility": 0.85, "normalized": {"x": 0.54, "y": 0.72}},
        ],
        "angles": {"x_factor": 15.0 + i * 0.3, "shoulder_rotation": 8.0 + i * 0.2},
    }


def _keyframes_ok(pk: dict) -> list[dict]:
    out = []
    for ph in PHASE_ORDER:
        m = SWING_PHASE_META[ph]
        out.append(
            {
                "phase": ph,
                "label_en": m["label_en"],
                "label_zh": m["label_zh"],
                "frame_index": int(pk[ph]) * 2,
                "timestamp": int(pk[ph]) * 0.033,
                "source_pose_idx": int(pk[ph]),
                "image_base64": "qq==",
            }
        )
    return out


class TestTimeAxisAndGradient(unittest.TestCase):
    def test_strict_increasing_time_axis_no_divide_warnings(self):
        poses = [_golf_pose(i, 0.02 + i * 0.033) for i in range(36)]
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            ev = detect_phase_events_agnostic(poses)
        bad_div = [
            w
            for w in wrec
            if issubclass(w.category, RuntimeWarning)
            and ("divide" in str(w.message).lower() or "invalid" in str(w.message).lower())
        ]
        self.assertEqual(len(bad_div), 0)
        for name in ("top_candidate_debug", "impact_candidate_debug"):
            d = ev.get(name) or {}
            if d.get("reason") == "kinematics_unavailable":
                continue
            for kk, vv in d.items():
                if kk in ("signals", "window", "reason"):
                    continue
                if isinstance(vv, (int, float)):
                    self.assertTrue(math.isfinite(float(vv)), msg=f"{name}.{kk}={vv}")
        pk = {ph: i * 3 for i, ph in enumerate(PHASE_ORDER)}
        kfs = _keyframes_ok(pk)
        rep = build_semantic_phase_report(
            poses, pk, {"passed": True}, keyframes=kfs, final_keyframe_validation={"rebuild_used": True},
        )
        self.assertNotIn(rep.get("fail_code"), ("DT_AXIS_INVALID", "NON_FINITE_KINEMATICS"))

    def test_dt_duplicate_timestamps(self):
        dup_ts = 0.99
        poses = [_golf_pose(i, dup_ts) for i in range(20)]
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            ev = detect_phase_events_agnostic(poses)
        bad_runtime = [
            w
            for w in wrec
            if issubclass(w.category, RuntimeWarning) and ("divide" in str(w.message).lower() or "invalid" in str(w.message).lower())
        ]
        self.assertEqual(len(bad_runtime), 0)
        self.assertTrue(ev["time_axis_debug"].get("dt_axis_invalid"))
        self.assertIn("DT_AXIS_INVALID", ev["kinematic_fail_codes"])

        pk = {ph: i * 2 for i, ph in enumerate(PHASE_ORDER)}
        rep = build_semantic_phase_report(poses, pk, {"passed": True}, keyframes=None)
        self.assertTrue(rep.get("dt_axis_invalid"))
        strip_reasons = list(rep.get("phase_strip_semantic_reasons") or [])
        self.assertIn("DT_AXIS_INVALID", strip_reasons)

        def _kf_val():
            return {
                "final_keyframe_gate_pass": True,
                "final_keyframe_source": "smart",
                "final_keyframe_validation": {
                    "strict_contract_ok": True,
                    "semantic_strip_ok": False,
                    "semantic_strip_reasons": strip_reasons,
                },
            }

        dec = should_run_phase_analysis_strict(
            pose_quality_bundle={
                "pose_quality_report": {"coverage_ratio": 0.9, "frame_count": 20},
                "pose_quality_report_post": {
                    "coverage_ratio": 0.9,
                    "frame_count": 20,
                    "wrist_jump_clamped_count": 0,
                    "smoothing_lag_score": 0.01,
                },
                "pose_reliability_level": "high",
                "reliability_reason_codes": [],
            },
            sem_report=rep,
            kf_validation=_kf_val(),
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle={"sweet_spot_confidence": 0.95, "sweet_spot_unstable": False},
        )
        self.assertIn("DT_AXIS_INVALID", dec["reasons"])

    def test_safe_gradient_no_nonfinite(self):
        t = np.array([0.0, 0.04, 0.08, 0.12, 0.16], dtype=np.float64)
        s = np.array([1.0, 2.0, 1.5, 3.0, 2.2], dtype=np.float64)
        g, had_nf = safe_gradient(s, t)
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertFalse(had_nf)
        g2, _ = safe_gradient(s, np.array([0.0, 0.0, 0.1, 0.2, 0.3]))
        self.assertTrue(np.all(np.isfinite(g2)))


class TestPhaseStripReasonsStable(unittest.TestCase):
    def test_phase_strip_semantic_fail_reasons_stable(self):
        n = 24
        poses = [_golf_pose(i, i * 0.033) for i in range(n)]
        pk = {ph: i * 2 for i, ph in enumerate(PHASE_ORDER)}
        kfs = _keyframes_ok(pk)
        # List order stays standard; pose indices are not increasing with phase order.
        kfs[0]["source_pose_idx"] = 40
        kfs[1]["source_pose_idx"] = 2

        a = verify_phase_strip_semantics(kfs, poses, pk)
        b = verify_phase_strip_semantics(kfs, poses, pk)
        self.assertEqual(sorted(a["reasons"]), sorted(b["reasons"]))
        self.assertIn("PHASE_STRIP_PHASE_ORDER_MISMATCH_AFTER_POSE_SORT", a["reasons"])


if __name__ == "__main__":
    unittest.main()
