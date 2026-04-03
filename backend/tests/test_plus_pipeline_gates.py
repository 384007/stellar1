"""Plus route hard gates, keyframe validation, and skeleton render contract."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _ensure_mediapipe_stub_if_missing() -> None:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        if "mediapipe" in sys.modules:
            return
        m = MagicMock()
        sol = MagicMock()
        sol.pose = MagicMock()
        sol.drawing_utils = MagicMock()
        m.solutions = sol
        sys.modules["mediapipe"] = m


_ensure_mediapipe_stub_if_missing()

from services.hud_service import generate_hud_data
from services.keyframe_service import PHASE_ORDER, SWING_PHASE_META, validate_final_keyframes_for_ai
from services.phase_analysis_gate import build_phase_alignment_fail_detail, collect_plus_route_hard_reasons, should_run_phase_analysis_strict


def _pose_bundle_high() -> dict:
    post = {
        "coverage_ratio": 0.9,
        "frame_count": 10,
        "wrist_jump_clamped_count": 0,
        "smoothing_lag_score": 0.01,
    }
    return {
        "pose_quality_report": {"coverage_ratio": 0.9, "frame_count": 10},
        "pose_quality_report_post": post,
        "pose_reliability_level": "high",
        "reliability_reason_codes": [],
    }


_SWEET_STABLE = {
    "sweet_spot_confidence": 0.95,
    "sweet_spot_unstable": False,
    "sweet_spot_valid_frames": 5,
}


def _sem_ok_strict() -> dict:
    return {
        "final_phase_semantic_ok_strict": True,
        "align_tol": 6,
        "top_abs_err": 1,
        "impact_abs_err": 1,
        "keyframe_semantic_ok": True,
        "phase_reselection_failed": False,
        "semantic_validation": {"phase_validation_passed": True},
        "final_phase_semantic_ok_strict_reasons": [],
    }


def _kf(ph: str, fi: int, spi: int) -> dict:
    m = SWING_PHASE_META[ph]
    return {
        "phase": ph,
        "label_en": m["label_en"],
        "label_zh": m["label_zh"],
        "frame_index": fi,
        "timestamp": float(fi) * 0.033,
        "source_pose_idx": spi,
        "source_frame_index": fi,
    }


class TestKeyframeGateDuplicates(unittest.TestCase):
    def test_gate_near_duplicates_forces_fail(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        kfs = [_kf(p, 10 + i * 15, i) for i, p in enumerate(PHASE_ORDER)]
        details = [{"phase": p, "is_near_duplicate": True, "validation_passed": False} for p in PHASE_ORDER]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertFalse(g["pass"])
        self.assertFalse(g["strict_contract_ok"])
        self.assertIn("NEAR_DUPLICATE_PRESENT", g["strict_contract_fail_reasons"])
        self.assertNotIn("DETAIL_VALIDATION_FAILED", g["strict_contract_fail_reasons"])
        self.assertFalse(g["strict_contract_checks"]["near_duplicates_ok"])

    def test_gate_time_too_close_forces_fail(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        kfs = [_kf(p, 10 + i * 15, i) for i, p in enumerate(PHASE_ORDER)]
        details = [{"phase": p, "time_too_close": True, "validation_passed": False} for p in PHASE_ORDER]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertFalse(g["pass"])
        self.assertIn("TIME_TOO_CLOSE_PRESENT", g["strict_contract_fail_reasons"])
        self.assertNotIn("DETAIL_VALIDATION_FAILED", g["strict_contract_fail_reasons"])
        self.assertFalse(g["strict_contract_checks"]["time_too_close_ok"])


class TestDetailValidationFailedReason(unittest.TestCase):
    def test_detail_validation_failed_code(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        kfs = [_kf(p, 10 + i * 15, i) for i, p in enumerate(PHASE_ORDER)]
        details = [
            {
                "phase": p,
                "is_near_duplicate": False,
                "time_too_close": False,
                "validation_passed": (p != "impact"),
            }
            for p in PHASE_ORDER
        ]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertFalse(g["strict_contract_ok"])
        self.assertIn("DETAIL_VALIDATION_FAILED", g["strict_contract_fail_reasons"])
        self.assertFalse(g["strict_contract_checks"]["details_all_passed"])


class TestStrictContractGatePassthrough(unittest.TestCase):
    def test_phase_analysis_gate_trace_and_detail_show_subreasons(self):
        fv = {
            "strict_contract_ok": False,
            "semantic_strip_ok": True,
            "semantic_strip_reasons": [],
            "near_duplicates": 2,
            "time_too_close_count": 1,
            "strip_detail_any_failed": True,
            "source_frame_gaps_ok": True,
            "source_frame_gap_reasons": [],
            "adjacent_strip_hard_dup_reasons": [],
            "strict_contract_checks": {
                "near_duplicates_ok": False,
                "phase_count_ok": True,
                "phase_sequence_ok": True,
                "unique_source_pose_ok": True,
                "final_keyframe_order_ok": True,
                "final_keyframe_time_order_ok": True,
                "final_phase_keyframes_sync_ok": True,
                "negative_time_gap_in_details": False,
                "no_negative_time_gap": True,
                "time_too_close_ok": True,
                "details_all_passed": False,
                "adjacent_strip_ok": True,
                "source_frame_gaps_ok": True,
            },
            "strict_contract_fail_reasons": ["NEAR_DUPLICATE_PRESENT", "DETAIL_VALIDATION_FAILED"],
        }
        kfv = {
            "final_keyframe_gate_pass": False,
            "final_keyframe_source": "smart",
            "final_keyframe_validation": fv,
        }
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=_pose_bundle_high(),
            sem_report=_sem_ok_strict(),
            kf_validation=kfv,
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle=_SWEET_STABLE,
        )
        self.assertFalse(d["pass"])
        self.assertIn("KEYFRAME_STRICT_CONTRACT_FAIL", d["reasons"])
        self.assertIn("NEAR_DUPLICATE_PRESENT", d["reasons"])
        tr = d["gate_decision_trace"]
        self.assertEqual(tr["strict_contract_fail_reasons"], fv["strict_contract_fail_reasons"])
        self.assertEqual(tr["strict_contract_checks"]["near_duplicates_ok"], False)
        self.assertEqual(tr["near_duplicates"], 2)
        self.assertEqual(tr["time_too_close_count"], 1)
        self.assertTrue(tr["strip_detail_any_failed"])
        det = build_phase_alignment_fail_detail(d)
        self.assertEqual(det["strict_contract_fail_reasons"], fv["strict_contract_fail_reasons"])
        self.assertEqual(det["strict_contract_checks"]["near_duplicates_ok"], False)
        self.assertEqual(det["near_duplicates"], 2)
        self.assertEqual(det["time_too_close_count"], 1)
        self.assertTrue(det["strip_detail_any_failed"])


class TestPlusDegradedReason(unittest.TestCase):
    def test_plus_422_on_degraded_phase_source_reason(self):
        r = collect_plus_route_hard_reasons(
            gate_pass_kf=True,
            phase_source="kinematic_degraded",
            pv_pass=True,
            sem_ok=True,
            phase_evaluations_reliable=True,
        )
        self.assertIn("PHASE_SOURCE_DEGRADED", r)


class TestRawFrameMinGap(unittest.TestCase):
    def test_raw_frame_min_gap_monotonic(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        base = 100
        step = 15
        kfs = [_kf(p, base + i * step, i) for i, p in enumerate(PHASE_ORDER)]
        details = [{"phase": p, "validation_passed": True, "is_near_duplicate": False, "time_too_close": False} for p in PHASE_ORDER]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertTrue(g["source_frame_gaps_ok"], msg=str(g.get("source_frame_gap_reasons")))

    def test_raw_frame_gap_too_tight_fails(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        kfs = [_kf(p, 10 + i, i) for i, p in enumerate(PHASE_ORDER)]
        details = [{"phase": p, "validation_passed": True, "is_near_duplicate": False, "time_too_close": False} for p in PHASE_ORDER]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertFalse(g["source_frame_gaps_ok"])
        self.assertTrue(any("MIN_GAP_VIOLATION" in x for x in (g.get("source_frame_gap_reasons") or [])))


def _ensure_mediapipe_stub_if_missing() -> None:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        if "mediapipe" in sys.modules:
            return
        m = MagicMock()
        sol = MagicMock()
        sol.pose = MagicMock()
        sol.drawing_utils = MagicMock()
        m.solutions = sol
        sys.modules["mediapipe"] = m


class TestRenderUsesDetection(unittest.TestCase):
    def test_render_uses_detection_joints(self):
        _ensure_mediapipe_stub_if_missing()
        pose = {
            "joints": [{"name": "left_wrist", "x": 1.0, "y": 2.0, "normalized": {"x": 0.1, "y": 0.2}, "visibility": 0.9}],
            "connections": [],
            "angles": {"x_factor": 10.0},
            "frame_size": {"width": 100, "height": 200},
            "detection": {
                "joints": [{"name": "left_wrist", "x": 99.0, "y": 88.0, "normalized": {"x": 0.99, "y": 0.44}, "visibility": 0.9}],
                "angles": {"x_factor": 20.0},
            },
        }
        from services.pose_service import pose_for_skeleton_render

        rpose = pose_for_skeleton_render(pose)
        hud = generate_hud_data(rpose, mode="lite", hand="RIGHT")
        x0 = hud["joints"][0]["x"]
        self.assertAlmostEqual(float(x0), 0.99, places=5)


if __name__ == "__main__":
    unittest.main()
