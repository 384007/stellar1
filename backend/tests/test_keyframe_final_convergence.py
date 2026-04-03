"""Final convergence: keyframe strict contract, selection gates, sticky skeleton snapshot."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        m = MagicMock()
        sol = MagicMock()
        sol.pose = MagicMock()
        sol.drawing_utils = MagicMock()
        m.solutions = sol
        sys.modules["mediapipe"] = m


_ensure_mediapipe_stub_if_missing()

from services.keyframe_service import (
    PHASE_ORDER,
    SWING_PHASE_META,
    _legalize_phase_pick_for_strip,
    _pose_snapshot_for_keyframe,
    _strip_pick_passes_hard_constraints,
    extract_keyframes_smart,
    recompute_keyframe_details_from_final_strip,
    validate_final_keyframes_for_ai,
)
from services.phase_analysis_gate import should_run_phase_analysis_strict
from services.pose_service import get_render_joints


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


def _joints_min():
    names = [
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_wrist", "right_wrist", "left_knee", "right_knee", "head",
    ]
    return [
        {
            "name": n,
            "x": 100.0,
            "y": 100.0,
            "z": 0.0,
            "visibility": 0.9,
            "normalized": {"x": 0.5, "y": 0.5},
        }
        for n in names
    ]


class TestValidateFinalKeyframesStrictFailures(unittest.TestCase):
    def test_near_duplicate_forces_fail(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        kfs = [_kf(p, 10 + i * 15, i) for i, p in enumerate(PHASE_ORDER)]
        details = [{"phase": p, "is_near_duplicate": True, "validation_passed": False} for p in PHASE_ORDER]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertFalse(g["pass"])
        self.assertIn("NEAR_DUPLICATE_PRESENT", g["strict_contract_fail_reasons"])
        self.assertTrue(g["strict_contract_checks"]["negative_time_gap_in_details"] is False)
        self.assertTrue(g["strict_contract_checks"].get("no_negative_time_gap", True))

    def test_time_too_close_forces_fail(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        kfs = [_kf(p, 10 + i * 15, i) for i, p in enumerate(PHASE_ORDER)]
        details = [{"phase": p, "time_too_close": True, "validation_passed": False} for p in PHASE_ORDER]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertFalse(g["pass"])
        self.assertIn("TIME_TOO_CLOSE_PRESENT", g["strict_contract_fail_reasons"])

    def test_details_validation_forces_fail(self):
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


class TestNegativeTimeGapSemantics(unittest.TestCase):
    def test_negative_time_gap_flag_true_when_problem(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        kfs = [_kf(p, 10 + i * 15, i) for i, p in enumerate(PHASE_ORDER)]
        details = [
            {"phase": p, "validation_passed": True, "time_gap": -0.1, "is_near_duplicate": False, "time_too_close": False}
            if p == "top"
            else {"phase": p, "validation_passed": True, "time_gap": 1.0, "is_near_duplicate": False, "time_too_close": False}
            for p in PHASE_ORDER
        ]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertTrue(g["negative_time_gap_in_details"])
        self.assertTrue(g["strict_contract_checks"]["negative_time_gap_in_details"])
        self.assertFalse(g["strict_contract_checks"]["no_negative_time_gap"])


class TestTopImpactFollowFinishGaps(unittest.TestCase):
    def test_top_impact_follow_finish_min_frame_gaps(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        base = 100
        step = 15
        kfs = [_kf(p, base + i * step, i) for i, p in enumerate(PHASE_ORDER)]
        details = [
            {"phase": p, "validation_passed": True, "is_near_duplicate": False, "time_too_close": False}
            for p in PHASE_ORDER
        ]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertTrue(g["source_frame_gaps_ok"], msg=str(g.get("source_frame_gap_reasons")))
        for i, p in enumerate(PHASE_ORDER):
            self.assertEqual(kfs[i]["source_frame_index"], base + i * step)
            if i > 0:
                self.assertGreater(kfs[i]["source_frame_index"], kfs[i - 1]["source_frame_index"])


class TestRenderUsesDetectionJoints(unittest.TestCase):
    def test_render_uses_detection_joints(self):
        pose = {
            "joints": [{"name": "head", "normalized": {"x": 0.11, "y": 0.22}, "visibility": 1.0}],
            "connections": [],
            "detection": {
                "joints": [{"name": "head", "normalized": {"x": 0.88, "y": 0.77}, "visibility": 1.0}],
            },
        }
        rj = get_render_joints(pose)
        self.assertAlmostEqual(rj[0]["normalized"]["x"], 0.88)
        snap = _pose_snapshot_for_keyframe(pose)
        self.assertAlmostEqual(snap["joints"][0]["nx"], 0.88)


class TestRecomputeDetailsFromFinalStrip(unittest.TestCase):
    def test_recompute_details_matches_final_strip(self):
        _ensure_mediapipe_stub_if_missing()
        poses = []
        for i in range(24):
            poses.append(
                {
                    "frame_index": 10 + i,
                    "timestamp": round((10 + i) / 30.0, 3),
                    "joints": _joints_min(),
                    "angles": {
                        "left_elbow": 90.0 + i * 4.0,
                        "right_elbow": 85.0 + i * 3.0,
                        "left_knee": 90.0,
                        "right_knee": 88.0,
                        "left_shoulder": 10.0 + i,
                        "right_shoulder": 12.0 + i,
                        "x_factor": 5.0 + i * 0.5,
                        "spine_tilt": 5.0 + i * 0.3,
                    },
                    "connections": [],
                }
            )
        kfs = []
        for j, p in enumerate(PHASE_ORDER):
            pi = min(j * 2, len(poses) - 1)
            fi = int(poses[pi]["frame_index"])
            m = SWING_PHASE_META[p]
            kfs.append({
                "phase": p,
                "label_en": m["label_en"],
                "label_zh": m["label_zh"],
                "frame_index": fi,
                "timestamp": float(poses[pi]["timestamp"]),
                "source_pose_idx": pi,
                "source_frame_index": fi,
            })

        def _pix(fi: int) -> np.ndarray:
            rng = np.random.default_rng(seed=int(fi) * 9973 + 42)
            return (rng.random((48, 48, 3)) * 255).astype(np.uint8)

        mock_cap = MagicMock()
        with patch(
            "services.keyframe_service._read_frame_pose_matched",
            side_effect=lambda _c, fi, _r: _pix(fi),
        ):
            with patch(
                "services.keyframe_service._read_frame_with_decode_fallback",
                side_effect=lambda _c, fi, _r: _pix(fi),
            ):
                d_ok = recompute_keyframe_details_from_final_strip(
                    mock_cap, 0, 30.0, poses, kfs, 0.001, min_visual_diff=0.01,
                )
        self.assertEqual(len(d_ok), 8)
        self.assertTrue(all(x["validation_passed"] for x in d_ok))

        kfs_bad = [dict(x) for x in kfs]
        kfs_bad[-1]["source_frame_index"] = kfs_bad[-2]["source_frame_index"]
        kfs_bad[-1]["frame_index"] = kfs_bad[-2]["frame_index"]
        with patch(
            "services.keyframe_service._read_frame_pose_matched",
            side_effect=lambda _c, fi, _r: _pix(fi),
        ):
            with patch(
                "services.keyframe_service._read_frame_with_decode_fallback",
                side_effect=lambda _c, fi, _r: _pix(fi),
            ):
                d_bad = recompute_keyframe_details_from_final_strip(
                    mock_cap, 0, 30.0, poses, kfs_bad, 0.001, min_visual_diff=0.01,
                )
        self.assertFalse(d_bad[-1]["validation_passed"])


class TestCandidateSearchQuality(unittest.TestCase):
    def test_candidate_search_removes_near_duplicate_if_possible(self):
        _ensure_mediapipe_stub_if_missing()
        import cv2

        n = 64
        poses = []
        for i in range(n):
            poses.append(
                {
                    "frame_index": i,
                    "timestamp": round(i / 30.0, 3),
                    "joints": _joints_min(),
                    "angles": {
                        "left_elbow": 90.0,
                        "right_elbow": 90.0,
                        "left_knee": 90.0,
                        "right_knee": 90.0,
                        "left_shoulder": 10.0,
                        "right_shoulder": 10.0,
                        "x_factor": 5.0,
                        "spine_tilt": 5.0,
                    },
                    "connections": [],
                }
            )
        swing_phases = [{"phase_id": PHASE_ORDER[min(i // 8, 7)]} for i in range(n)]
        pk = {p: min(8 + j * 7, n - 1) for j, p in enumerate(PHASE_ORDER)}

        def _fake_frame(fi: int) -> np.ndarray:
            v = int(fi) % 256
            return np.full((96, 96, 3), (v, (3 * v) % 256, (5 * v) % 256), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.return_value.isOpened.return_value = True
        mock_cap.return_value.get.side_effect = lambda k: (
            30.0 if k == cv2.CAP_PROP_FPS else float(n) if k == cv2.CAP_PROP_FRAME_COUNT else 0
        )

        with patch("services.keyframe_service.cv2.VideoCapture", mock_cap):
            with patch(
                "services.keyframe_service._read_frame_pose_matched",
                side_effect=lambda _cap, fi, _rot: _fake_frame(fi),
            ):
                with patch(
                    "services.keyframe_service._read_frame_with_decode_fallback",
                    side_effect=lambda _cap, fi, _rot: _fake_frame(fi),
                ):
                    kfs, summary = extract_keyframes_smart(
                        "/tmp/_dummy_convergence.mp4",
                        poses,
                        swing_phases,
                        dict(pk),
                        keyframe_width=200,
                    )

        self.assertEqual(summary.get("near_duplicates"), 0)
        self.assertEqual(summary.get("time_too_close"), 0)
        self.assertGreaterEqual(len(kfs), 1)

    def test_candidate_search_removes_time_too_close_if_possible(self):
        _ensure_mediapipe_stub_if_missing()
        n = 40
        poses = []
        ang = {
            "left_elbow": 90.0,
            "right_elbow": 90.0,
            "left_knee": 90.0,
            "right_knee": 90.0,
            "left_shoulder": 10.0,
            "right_shoulder": 10.0,
            "x_factor": 5.0,
            "spine_tilt": 5.0,
        }
        for i in range(n):
            poses.append(
                {
                    "frame_index": i,
                    "timestamp": round(i * 0.02, 3),
                    "joints": _joints_min(),
                    "angles": ang,
                    "connections": [],
                }
            )

        def _fr(fi: int) -> np.ndarray:
            return np.full((40, 40, 3), int(fi) * 13 % 255, dtype=np.uint8)

        cap = MagicMock()
        with patch(
            "services.keyframe_service._read_frame_pose_matched",
            side_effect=lambda _c, fi, _r: _fr(fi),
        ):
            with patch(
                "services.keyframe_service._read_frame_with_decode_fallback",
                side_effect=lambda _c, fi, _r: _fr(fi),
            ):
                best = {
                    "pose_idx": 5,
                    "frame": _fr(5),
                    "confidence": 0.9,
                    "fallback_used": False,
                    "selection_reason": "t",
                }
                out = _legalize_phase_pick_for_strip(
                    cap,
                    0,
                    poses,
                    "takeaway",
                    best,
                    list(range(n)),
                    5,
                    [6, 7, 8],
                    {},
                    set(),
                    1,
                    strip_prev_fi=3,
                    strip_prev_ts=0.08,
                    min_time_gap_early=0.12,
                    strip_acc_frames=[_fr(3)],
                    strip_acc_poses=[poses[3]],
                    kin_ctx=None,
                    exc_ctx=0,
                    phase_keyframes={},
                )
        self.assertIsNotNone(out)
        self.assertGreaterEqual(poses[out["pose_idx"]]["timestamp"], 0.18)


class TestNoLegalCandidateContract(unittest.TestCase):
    def test_no_legal_candidate_keeps_fail(self):
        pk = {p: i for i, p in enumerate(PHASE_ORDER)}
        kfs = [_kf(p, 10 + i * 15, i) for i, p in enumerate(PHASE_ORDER)]
        details = [
            {
                "phase": p,
                "is_near_duplicate": True,
                "time_too_close": False,
                "validation_passed": False,
                "fail_code": "NO_LEGAL_CANDIDATE",
            }
            for p in PHASE_ORDER
        ]
        g = validate_final_keyframes_for_ai(kfs, pk, details, poses=[], fps=30.0)
        self.assertFalse(g["pass"])
        self.assertFalse(g["strict_contract_ok"])
        self.assertIn("NEAR_DUPLICATE_PRESENT", g["strict_contract_fail_reasons"])


class TestStrictFailReasonExact(unittest.TestCase):
    def test_strict_fail_reason_exact(self):
        _ensure_mediapipe_stub_if_missing()
        post = {
            "coverage_ratio": 0.9,
            "frame_count": 10,
            "wrist_jump_clamped_count": 0,
            "smoothing_lag_score": 0.01,
        }
        pose_bundle = {
            "pose_quality_report": {"coverage_ratio": 0.9},
            "pose_quality_report_post": post,
            "pose_reliability_level": "high",
            "reliability_reason_codes": [],
        }
        sem = {
            "final_phase_semantic_ok_strict": True,
            "align_tol": 6,
            "top_abs_err": 1,
            "impact_abs_err": 1,
            "keyframe_semantic_ok": True,
            "phase_reselection_failed": False,
            "semantic_validation": {"phase_validation_passed": True},
            "final_phase_semantic_ok_strict_reasons": [],
        }
        kfv = {
            "final_keyframe_gate_pass": False,
            "final_keyframe_source": "smart",
            "keyframe_quality_repair_attempted": True,
            "keyframe_quality_repair_success": False,
            "final_keyframe_validation": {
                "strict_contract_ok": False,
                "semantic_strip_ok": True,
                "semantic_strip_reasons": [],
                "pass": False,
                "strict_contract_checks": {
                    "near_duplicates_ok": False,
                    "time_too_close_ok": False,
                    "details_all_passed": False,
                },
                "strict_contract_fail_reasons": [
                    "NEAR_DUPLICATE_PRESENT",
                    "TIME_TOO_CLOSE_PRESENT",
                    "DETAIL_VALIDATION_FAILED",
                ],
            },
        }
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=pose_bundle,
            sem_report=sem,
            kf_validation=kfv,
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle={
                "sweet_spot_confidence": 0.95,
                "sweet_spot_unstable": False,
                "sweet_spot_valid_frames": 5,
            },
        )
        self.assertFalse(d["pass"])
        r = d["reasons"]
        for code in (
            "KEYFRAME_STRICT_CONTRACT_FAIL",
            "NEAR_DUPLICATE_PRESENT",
            "TIME_TOO_CLOSE_PRESENT",
            "DETAIL_VALIDATION_FAILED",
        ):
            self.assertIn(code, r)
        tr = d["gate_decision_trace"]
        self.assertTrue(tr.get("keyframe_quality_repair_attempted"))
        self.assertFalse(tr.get("keyframe_quality_repair_success"))


class TestStripHardGate(unittest.TestCase):
    def test_strip_rejects_near_duplicate_against_accumulator(self):
        _ensure_mediapipe_stub_if_missing()
        f0 = np.full((32, 32, 3), 120, dtype=np.uint8)
        f1_diff = np.full((32, 32, 3), 40, dtype=np.uint8)
        pose0 = {"frame_index": 1, "timestamp": 0.1, "joints": _joints_min(), "angles": {}}
        pose1 = {"frame_index": 1, "timestamp": 0.5, "joints": _joints_min(), "angles": {}}
        ok, _viol = _strip_pick_passes_hard_constraints(
            f1_diff, pose1, 0, -999.0, 0.05, [f0], [pose0],
        )
        self.assertTrue(ok)
        ok2, viol2 = _strip_pick_passes_hard_constraints(
            f1_diff, pose1, 1, 0.1, 0.05, [f0], [pose0],
        )
        self.assertFalse(ok2)
        self.assertEqual(viol2, "SOURCE_FRAME_NOT_INCREASING")


if __name__ == "__main__":
    unittest.main()
