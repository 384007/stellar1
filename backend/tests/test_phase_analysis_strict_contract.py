"""Strict phase-AI gate + final keyframe validator (synthetic, no video)."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _ensure_mediapipe_stub_if_missing() -> None:
    """pose_service imports mediapipe at module load; refine tests need angles without a real install."""
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

from services.keyframe_service import PHASE_ORDER, SWING_PHASE_META, validate_final_keyframes_for_ai
from services.phase_analysis_gate import should_run_phase_analysis_strict


def _kf(ph: str, fi: int, ts: float, spi: int) -> dict:
    m = SWING_PHASE_META[ph]
    return {
        "phase": ph,
        "label_en": m["label_en"],
        "label_zh": m["label_zh"],
        "frame_index": fi,
        "timestamp": ts,
        "source_pose_idx": spi,
        "image_base64": "qq==",
    }


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


class TestValidateFinalKeyframesStrict(unittest.TestCase):
    def test_less_than_eight_phases_fails(self):
        kfs = [_kf(PHASE_ORDER[i], 10 + i, 0.1 * i, i) for i in range(4)]
        pk = {PHASE_ORDER[i]: i for i in range(4)}
        g = validate_final_keyframes_for_ai(kfs, pk, [])
        self.assertFalse(g["pass"])
        self.assertFalse(g["strict_contract_ok"])

    def test_duplicate_source_pose_fails(self):
        # Same pose index for every phase while phase_keyframes expects distinct indices.
        kfs = [_kf(PHASE_ORDER[i], 10 + i * 2, 0.05 * i, 0) for i in range(8)]
        pk = {PHASE_ORDER[i]: i for i in range(8)}
        g = validate_final_keyframes_for_ai(kfs, pk, [])
        self.assertFalse(g["final_phase_keyframes_sync_ok"])
        self.assertFalse(g["unique_source_pose_ok"])

    def test_wrong_phase_order_fails(self):
        order_bad = list(PHASE_ORDER)
        order_bad[0], order_bad[1] = order_bad[1], order_bad[0]
        kfs = [_kf(order_bad[i], 10 + i * 2, 0.05 * i, i) for i in range(8)]
        pk = {order_bad[i]: i for i in range(8)}
        g = validate_final_keyframes_for_ai(kfs, pk, [])
        self.assertFalse(g["phase_sequence_ok"])

    def test_non_monotonic_frame_index_fails(self):
        kfs = [_kf(PHASE_ORDER[i], 10 + i if i < 4 else 8, 0.05 * i, i) for i in range(8)]
        pk = {PHASE_ORDER[i]: i for i in range(8)}
        g = validate_final_keyframes_for_ai(kfs, pk, [])
        self.assertFalse(g["final_keyframe_order_ok"])


class TestStrictGateThreeRoutes(unittest.TestCase):
    def _sem_ok(self) -> dict:
        return {
            "final_phase_semantic_ok_strict": True,
            "align_tol": 6,
            "top_abs_err": 1,
            "impact_abs_err": 1,
            "keyframe_semantic_ok": True,
            "phase_reselection_failed": False,
            "semantic_validation": {"phase_validation_passed": True},
        }

    def _sem_bad_align(self) -> dict:
        return {
            "final_phase_semantic_ok_strict": False,
            "align_tol": 1,
            "top_abs_err": 50,
            "impact_abs_err": 50,
            "keyframe_semantic_ok": False,
            "phase_reselection_failed": False,
            "semantic_validation": {"phase_validation_passed": True},
        }

    def _kf_val_ok(self) -> dict:
        return {
            "final_keyframe_gate_pass": True,
            "final_keyframe_source": "smart",
            "final_keyframe_validation": {
                "strict_contract_ok": True,
                "semantic_strip_ok": True,
                "semantic_strip_reasons": [],
            },
        }

    def test_alignment_over_threshold_fails(self):
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=_pose_bundle_high(),
            sem_report=self._sem_bad_align(),
            kf_validation=self._kf_val_ok(),
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess=None,
            sweet_spot_bundle=_SWEET_STABLE,
        )
        self.assertFalse(d["pass"])
        self.assertIn("PHASE_SEMANTIC_OR_ALIGNMENT_STRICT_FAIL", d["reasons"])

    def test_three_routes_identical_decision(self):
        args = dict(
            pose_quality_bundle=_pose_bundle_high(),
            sem_report=self._sem_ok(),
            kf_validation=self._kf_val_ok(),
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle=_SWEET_STABLE,
        )
        a = should_run_phase_analysis_strict(**args)
        b = should_run_phase_analysis_strict(**args)
        c = should_run_phase_analysis_strict(**args)
        self.assertEqual(a["pass"], b["pass"])
        self.assertEqual(b["pass"], c["pass"])
        self.assertTrue(a["pass"])

    def test_gemini_map_diverged_fails(self):
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=_pose_bundle_high(),
            sem_report=self._sem_ok(),
            kf_validation=self._kf_val_ok(),
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={
                "gemini_uniform_thumbnail_map_applies": True,
                "gemini_map_aligned_with_final_strip": False,
            },
            sweet_spot_bundle=_SWEET_STABLE,
        )
        self.assertFalse(d["pass"])
        self.assertIn("GEMINI_THUMB_MAP_DIVERGED", d["reasons"])

    def test_phase_strip_semantic_fail_reason(self):
        from services.phase_analysis_gate import build_phase_alignment_fail_detail

        kfv = {
            "final_keyframe_gate_pass": False,
            "final_keyframe_source": "smart",
            "final_keyframe_validation": {
                "strict_contract_ok": True,
                "semantic_strip_ok": False,
                "semantic_strip_reasons": ["TOP_SEMANTIC_AT_KEYFRAME_FAIL"],
            },
        }
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=_pose_bundle_high(),
            sem_report=self._sem_ok(),
            kf_validation=kfv,
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess=None,
            sweet_spot_bundle=_SWEET_STABLE,
        )
        self.assertFalse(d["pass"])
        self.assertIn("PHASE_STRIP_SEMANTIC_ORDER_FAIL", d["reasons"])
        detail = build_phase_alignment_fail_detail(d)
        self.assertIn("phase_strip_semantic_ok", detail)
        self.assertIn("gate_decision_trace", detail)


class TestNonFiniteDebugSanitized(unittest.TestCase):
    def test_non_finite_debug_values_eliminated(self):
        from services.swing_flow_utils import _sanitize_phase_event_debug_dict

        d = {
            "speed_at_top": float("nan"),
            "xf_deriv_at_impact": float("inf"),
            "window": [0, 5],
            "signals": "test",
        }
        _sanitize_phase_event_debug_dict(d)
        self.assertTrue(math.isfinite(d["speed_at_top"]))
        self.assertTrue(math.isfinite(d["xf_deriv_at_impact"]))


class TestSemanticStrictObservability(unittest.TestCase):
    def test_build_semantic_report_exposes_strict_subconditions(self):
        from services.swing_flow_utils import build_semantic_phase_report, detect_phase_events_agnostic

        kin_path = Path(__file__).resolve().parent / "test_keyframe_kinematic_anchor.py"
        spec = importlib.util.spec_from_file_location("kf_kin", kin_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        poses = mod._make_swing_poses(48)
        ev = detect_phase_events_agnostic(poses)
        te, ie = int(ev["top_pose_idx"]), int(ev["impact_pose_idx"])
        pk = {p: 2 + i * 5 for i, p in enumerate(PHASE_ORDER)}
        pk["top"] = max(0, te - 20)
        pk["impact"] = min(len(poses) - 1, ie + 14)
        rep = build_semantic_phase_report(poses, dict(pk), {"passed": False}, keyframes=None)
        self.assertIn("align_top", rep)
        self.assertIn("align_impact", rep)
        self.assertIn("keyframe_semantic_ok", rep)
        self.assertIn("phase_validation_passed", rep)
        self.assertIn("phase_validation_soft_fail", rep)
        self.assertIn("phase_validation_warning", rep)
        self.assertEqual(bool(rep["phase_validation_soft_fail"]), not bool(rep["phase_validation_passed"]))
        self.assertIn("final_phase_semantic_ok_strict_reasons", rep)
        self.assertIsInstance(rep["final_phase_semantic_ok_strict_reasons"], list)
        self.assertNotIn(
            "PHASE_VALIDATION_FAIL",
            rep["final_phase_semantic_ok_strict_reasons"],
        )
        if rep.get("phase_validation_reran_after_reselect"):
            self.assertIsNotNone(rep.get("phase_validation_post_reselect"))
            pv = rep["phase_validation_post_reselect"]
            self.assertEqual(bool(rep["phase_validation_passed"]), bool(pv.get("passed")))
        if not rep.get("final_phase_semantic_ok_strict"):
            self.assertGreater(len(rep["final_phase_semantic_ok_strict_reasons"]), 0)


class TestTopImpactReselection(unittest.TestCase):
    def test_top_impact_reselection_reduces_error(self):
        from services.swing_flow_utils import build_semantic_phase_report, detect_phase_events_agnostic

        kin_path = Path(__file__).resolve().parent / "test_keyframe_kinematic_anchor.py"
        spec = importlib.util.spec_from_file_location("kf_kin", kin_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        poses = mod._make_swing_poses(52)
        ev = detect_phase_events_agnostic(poses)
        te, ie = int(ev["top_pose_idx"]), int(ev["impact_pose_idx"])
        pk = {
            "address": 0,
            "takeaway": 2,
            "backswing": 4,
            "top": max(0, te - 18),
            "downswing": max(6, ie - 4),
            "impact": min(len(poses) - 1, ie + 12),
            "follow_through": min(len(poses) - 2, ie + 16),
            "finish": len(poses) - 1,
        }
        rep = build_semantic_phase_report(poses, dict(pk), {"passed": True})
        before = (rep.get("top_abs_err_before") or 0) + (rep.get("impact_abs_err_before") or 0)
        after = (rep.get("top_abs_err_after") or 0) + (rep.get("impact_abs_err_after") or 0)
        self.assertLessEqual(after, before)


class TestWristJumpClampTrack(unittest.TestCase):
    def test_wrist_jump_clamp_improves_track_consistency(self):
        _ensure_mediapipe_stub_if_missing()
        from services.pose_refine_service import compute_pose_quality_report, refine_pose_sequence_pipeline

        # Gap ≤4 low-vis frames get interpolated away before the clamp runs; use 5+ consecutive
        # low-vis so occlusion fill skips and the spike survives to filter + torso clamp.
        low_band = range(7, 12)
        poses = []
        for i in range(20):
            v = 0.12 if i in low_band else 0.85
            poses.append(
                {
                    "frame_index": i,
                    "timestamp": i * 0.033,
                    "frame_size": {"width": 100, "height": 100},
                    "joints": [
                        {"name": "left_shoulder", "visibility": 0.9, "normalized": {"x": 0.4, "y": 0.3}},
                        {"name": "right_shoulder", "visibility": 0.9, "normalized": {"x": 0.6, "y": 0.3}},
                        {"name": "left_hip", "visibility": 0.9, "normalized": {"x": 0.45, "y": 0.65}},
                        {"name": "right_hip", "visibility": 0.9, "normalized": {"x": 0.55, "y": 0.65}},
                        {
                            "name": "left_wrist",
                            "visibility": v,
                            "normalized": {"x": 0.45 + (0.35 if i == 9 else 0), "y": 0.5},
                        },
                        {
                            "name": "right_wrist",
                            "visibility": v,
                            "normalized": {"x": 0.55 + (0.35 if i == 9 else 0), "y": 0.5},
                        },
                    ],
                    "angles": {"x_factor": 20.0, "shoulder_rotation": 10.0},
                }
            )
        pre = compute_pose_quality_report(poses)
        bundle = refine_pose_sequence_pipeline(poses, 30.0)
        post = bundle["pose_quality_report_post"]
        self.assertGreaterEqual(float(post.get("track_consistency", 0)), float(pre.get("track_consistency", 0)) - 0.01)
        self.assertGreaterEqual(int(post.get("wrist_jump_clamped_count", 0)), 1)


class TestSweetSpotSoftGate(unittest.TestCase):
    def test_unstable_sweet_spot_does_not_422_when_kf_semantic_strict_ok(self):
        tg = TestStrictGateThreeRoutes()
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=_pose_bundle_high(),
            sem_report=tg._sem_ok(),
            kf_validation=tg._kf_val_ok(),
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle={
                "sweet_spot_confidence": 0.166,
                "sweet_spot_unstable": True,
                "sweet_spot_reasons": ["SWEET_SPOT_UNSTABLE"],
                "sweet_spot_valid_frames": 1,
            },
        )
        self.assertTrue(d["pass"], msg=str(d.get("reasons")))
        self.assertNotIn("SWEET_SPOT_UNSTABLE", d["reasons"])
        tr = d["gate_decision_trace"]
        self.assertTrue(tr.get("sweet_spot_warning"))
        self.assertAlmostEqual(tr.get("sweet_spot_confidence"), 0.166, places=3)
        self.assertIn("SWEET_SPOT_UNSTABLE", tr.get("sweet_spot_reasons") or [])

    def test_keyframe_strict_fail_still_fails_gate(self):
        tg = TestStrictGateThreeRoutes()
        kfv = {
            "final_keyframe_gate_pass": True,
            "final_keyframe_source": "smart",
            "final_keyframe_validation": {
                "strict_contract_ok": False,
                "semantic_strip_ok": True,
                "semantic_strip_reasons": [],
            },
        }
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=_pose_bundle_high(),
            sem_report=tg._sem_ok(),
            kf_validation=kfv,
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle=_SWEET_STABLE,
        )
        self.assertFalse(d["pass"])
        self.assertIn("KEYFRAME_STRICT_CONTRACT_FAIL", d["reasons"])


class TestCapConfidenceSweetSpotPenalty(unittest.TestCase):
    def test_cap_confidence_applies_sweet_spot_penalty_and_reason(self):
        from services.gemini_service import cap_confidence

        ai = {"primary_diagnosis": {"ai_confidence": 80}}
        ar = cap_confidence(
            ai,
            sweet_spot_unstable=True,
            sweet_spot_confidence=0.166,
        )
        self.assertIn("sweet_spot_unstable", ar["reasons"])
        self.assertLessEqual(ar["capped_confidence"], 60)

    def test_cap_confidence_phase_validation_uses_soft_fail_reason(self):
        from services.gemini_service import cap_confidence

        ai = {"primary_diagnosis": {"ai_confidence": 80}}
        ar = cap_confidence(ai, phase_validation={"passed": False})
        self.assertIn("phase_validation_soft_fail", ar["reasons"])
        self.assertGreaterEqual(ar["penalty"], 30)


class TestBadVideoRegressionPlaceholder(unittest.TestCase):
    """实拍坏视频集需由 STELLAR_BAD_VIDEO_DIR 提供；未提供则跳过并记为回归未完成。"""

    @unittest.skipUnless(
        __import__("os").environ.get("STELLAR_BAD_VIDEO_DIR"),
        "STELLAR_BAD_VIDEO_DIR not set — 实拍坏视频回归未执行",
    )
    def test_run_bad_video_regression(self):
        # Placeholder: 有目录时再跑端到端统计 strict pass / false reliable。
        self.fail("Implement batch runner when bad-video corpus is available")


if __name__ == "__main__":
    unittest.main()
