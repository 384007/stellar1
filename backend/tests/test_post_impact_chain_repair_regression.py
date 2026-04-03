import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from services.keyframe_service import (
    PHASE_ORDER,
    _reselect_strip_quality_failures,
    ensure_keyframes_ordered_for_ai,
    validate_final_keyframes_for_ai,
    joint_rebuild_phase_map_for_monotonic_strip,
    joint_rebuild_phase_map_for_quality_spacing,
)
from services.swing_flow_utils import propose_post_impact_chain_indices, propose_quality_spacing_post_top_chain


def _mk_pose(i: int) -> dict:
    return {
        "frame_index": i * 5,
        "timestamp": round(i * 0.05, 3),
        "joints": [
            {"name": "left_shoulder", "visibility": 0.9, "normalized": {"x": 0.4, "y": 0.3}},
            {"name": "right_shoulder", "visibility": 0.9, "normalized": {"x": 0.6, "y": 0.3}},
            {"name": "left_hip", "visibility": 0.9, "normalized": {"x": 0.45, "y": 0.6}},
            {"name": "right_hip", "visibility": 0.9, "normalized": {"x": 0.55, "y": 0.6}},
            {"name": "left_wrist", "visibility": 0.9, "normalized": {"x": 0.43 + i * 0.001, "y": 0.52}},
            {"name": "right_wrist", "visibility": 0.9, "normalized": {"x": 0.57 + i * 0.001, "y": 0.52}},
            {"name": "left_knee", "visibility": 0.9, "normalized": {"x": 0.46, "y": 0.78}},
            {"name": "right_knee", "visibility": 0.9, "normalized": {"x": 0.54, "y": 0.78}},
        ],
        "angles": {
            "left_elbow": 140.0,
            "right_elbow": 130.0,
            "left_knee": 165.0,
            "right_knee": 167.0,
            "left_shoulder": 20.0,
            "right_shoulder": 20.0,
            "x_factor": max(5.0, 58.0 - abs(78 - i) * 1.5),
            "spine_tilt": 10.0,
        },
        "connections": [],
    }


def _mk_kf(phase: str, spi: int, poses: list[dict]) -> dict:
    fi = poses[spi]["frame_index"]
    ts = poses[spi]["timestamp"]
    return {
        "phase": phase,
        "label_en": phase,
        "label_zh": phase,
        "frame_index": fi,
        "source_frame_index": fi,
        "source_pose_idx": spi,
        "timestamp": ts,
        "image_base64": "x",
        "pose_snapshot": {"joints": [], "connections": []},
        "selection_reason": "seed",
    }


class TestPostImpactChainRepairRegression(unittest.TestCase):
    def test_post_impact_joint_chain_separates_impact_follow_finish(self):
        n = 120
        poses = [_mk_pose(i) for i in range(n)]
        speed = np.concatenate([
            np.linspace(0.05, 0.3, 60),
            np.linspace(0.35, 1.0, 18),
            np.linspace(0.92, 0.35, 20),
            np.linspace(0.3, 0.08, 22),
        ])
        speed = speed[:n]
        valid = np.ones(n, dtype=bool)
        q = np.full(n, 0.92, dtype=float)
        kin = {"n": n, "speed_s": speed, "valid": valid, "q": q, "xf_d": -np.abs(np.gradient(speed)), "hand_hip": np.linspace(0.8, 0.2, n)}
        seed = {
            "top": 66,
            "downswing": 74,
            "impact": 78,
            "follow_through": 80,
            "finish": 82,
        }

        with patch("services.swing_flow_utils._build_view_agnostic_kinematics", return_value=kin), \
             patch("services.swing_flow_utils.validate_impact_semantic_at_index", side_effect=lambda i, *_: (i >= 76, {"ok": i >= 76})):
            out = propose_post_impact_chain_indices(poses, seed)

        self.assertTrue(out)
        self.assertGreater(out["impact"], out["downswing"])
        self.assertGreater(out["follow_through"], out["impact"])
        self.assertGreater(out["finish"], out["follow_through"])
        self.assertGreaterEqual(out["follow_through"] - out["impact"], 2)
        self.assertGreaterEqual(out["finish"] - out["follow_through"], 2)
        self.assertLess(speed[out["finish"]], speed[out["impact"]])

    def test_strip_quality_repair_changes_indices_and_reduces_failures(self):
        poses = [_mk_pose(i) for i in range(120)]
        bad_idxs = [10, 27, 34, 66, 74, 78, 80, 82]
        keyframes = [_mk_kf(p, bad_idxs[i], poses) for i, p in enumerate(PHASE_ORDER)]
        phase_map = {p: bad_idxs[i] for i, p in enumerate(PHASE_ORDER)}

        def _fake_frame(_cap, fi, _rot):
            v = int(fi) % 255
            return np.full((64, 64, 3), (v, (v * 7) % 255, (v * 13) % 255), dtype=np.uint8)

        repaired_chain = {"downswing": 70, "impact": 76, "follow_through": 86, "finish": 97}

        with patch("services.keyframe_service._read_frame_pose_matched", side_effect=_fake_frame), \
             patch("services.keyframe_service._read_frame_with_decode_fallback", side_effect=_fake_frame), \
             patch("services.keyframe_service._pose_snapshot_for_keyframe", return_value={"joints": [], "connections": []}), \
             patch("services.swing_flow_utils.propose_post_impact_chain_indices", return_value=repaired_chain):
            out_kf, out_details = _reselect_strip_quality_failures(
                cap=MagicMock(),
                rotation=0,
                fps=30.0,
                poses=poses,
                keyframes=[dict(k) for k in keyframes],
                phase_keyframes=dict(phase_map),
                keyframe_width=160,
                min_time_gap=0.05,
                semantic_fail_reasons=["IMPACT_SEMANTIC_AT_KEYFRAME_FAIL", "POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"],
            )

        out_idxs = [int(k["source_pose_idx"]) for k in out_kf]
        self.assertNotEqual(out_idxs, bad_idxs)
        self.assertEqual(out_idxs[5], 76)
        self.assertEqual(out_idxs[6], 86)
        self.assertEqual(out_idxs[7], 97)
        self.assertFalse(any(d.get("is_near_duplicate") for d in out_details[1:]))

    def test_ensure_noop_repair_forces_real_fallback(self):
        poses = [_mk_pose(i) for i in range(32)]
        idxs = [2, 5, 8, 11, 14, 17, 20, 23]
        keyframes = [_mk_kf(p, idxs[i], poses) for i, p in enumerate(PHASE_ORDER)]
        phase_map = {p: idxs[i] for i, p in enumerate(PHASE_ORDER)}
        bad_gate = {
            "pass": False,
            "strict_contract_ok": False,
            "semantic_strip_ok": False,
            "strict_contract_fail_reasons": ["NEAR_DUPLICATE_PRESENT"],
            "semantic_strip_reasons": ["IMPACT_SEMANTIC_AT_KEYFRAME_FAIL"],
            "final_keyframe_order_ok": True,
            "final_keyframe_time_order_ok": True,
            "final_phase_keyframes_sync_ok": True,
            "negative_time_gap_in_details": False,
        }
        ok_gate = dict(bad_gate)
        ok_gate.update({"pass": True, "strict_contract_ok": True, "semantic_strip_ok": True, "strict_contract_fail_reasons": [], "semantic_strip_reasons": []})

        fallback_kf = [_mk_kf(p, i + 3, poses) for i, p in enumerate(PHASE_ORDER)]
        fallback_summary = {"details": [], "final_phase_keyframes": {p: i + 3 for i, p in enumerate(PHASE_ORDER)}}

        cap_mock = MagicMock()
        cap_mock.return_value.get.return_value = 30.0

        with patch("services.keyframe_service.cv2.VideoCapture", cap_mock), \
             patch("services.keyframe_service.validate_final_keyframes_for_ai", side_effect=[bad_gate, ok_gate]), \
             patch("services.keyframe_service._reselect_strip_quality_failures", return_value=([dict(k) for k in keyframes], [])), \
             patch("services.keyframe_service.joint_rebuild_phase_map_for_quality_spacing", return_value=(dict(phase_map), False)), \
             patch("services.phase_chain_solver_service.solve_post_impact_phase_chain", return_value={"phase_keyframes": dict(phase_map), "chain": {}, "material_change": False, "reasons": []}), \
             patch("services.keyframe_service.build_semantic_oriented_phase_map", return_value={"ok": False}), \
             patch("services.keyframe_service.extract_keyframes_ordered_fallback", return_value=(fallback_kf, fallback_summary)), \
             patch("services.keyframe_service.get_video_rotation", return_value=0):
            out_kf, merged, out_phase, src = ensure_keyframes_ordered_for_ai(
                "dummy.mp4", poses, [], dict(phase_map), list(keyframes), {"details": []}, dict(phase_map), 160
            )

        self.assertEqual(src, "ordered_fallback_repaired")
        self.assertTrue(merged["final_keyframe_gate_pass"])
        self.assertEqual(out_phase["impact"], 8)
        self.assertEqual(out_kf[0]["source_pose_idx"], 3)

    def test_ensure_noop_triggers_forced_joint_rebuild_best_candidate(self):
        poses = [_mk_pose(i) for i in range(40)]
        idxs = [4, 8, 12, 16, 20, 22, 24, 26]
        keyframes = [_mk_kf(p, idxs[i], poses) for i, p in enumerate(PHASE_ORDER)]
        phase_map = {p: idxs[i] for i, p in enumerate(PHASE_ORDER)}
        base_gate = {
            "pass": False,
            "strict_contract_ok": False,
            "semantic_strip_ok": False,
            "strict_contract_fail_reasons": ["NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"],
            "semantic_strip_reasons": ["IMPACT_SEMANTIC_AT_KEYFRAME_FAIL", "POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"],
            "final_keyframe_order_ok": True,
            "final_keyframe_time_order_ok": True,
            "final_phase_keyframes_sync_ok": True,
            "negative_time_gap_in_details": False,
        }
        improved_gate = dict(base_gate)
        improved_gate["strict_contract_fail_reasons"] = ["TIME_TOO_CLOSE_PRESENT"]
        improved_gate["semantic_strip_reasons"] = ["POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"]
        improved_gate["pass"] = False
        improved_map = dict(phase_map)
        improved_map.update({"downswing": 21, "impact": 25, "follow_through": 30, "finish": 35})
        improved_kf = [_mk_kf(p, improved_map[p], poses) for p in PHASE_ORDER]
        for k in improved_kf:
            k["image_base64"] = "ok"

        cap_mock = MagicMock()
        cap_mock.return_value.isOpened.return_value = True
        cap_mock.return_value.get.return_value = 30.0

        # validate call order: initial gate → forced joint ch_gate (must beat base for best_candidate) → ordered fallback of_gate (must fail)
        with patch("services.keyframe_service.cv2.VideoCapture", cap_mock), \
             patch("services.keyframe_service.validate_final_keyframes_for_ai", side_effect=[base_gate, improved_gate, base_gate]), \
             patch("services.keyframe_service._reselect_strip_quality_failures", return_value=([dict(k) for k in keyframes], [])), \
             patch("services.keyframe_service.joint_rebuild_phase_map_for_quality_spacing", return_value=(dict(phase_map), False)), \
             patch("services.keyframe_service.build_semantic_oriented_phase_map", return_value={"ok": False}), \
             patch("services.keyframe_service.rebuild_phase_map_from_event_anchors", return_value={"rebuild_ok": False}), \
             patch("services.phase_chain_solver_service.solve_post_impact_phase_chain", return_value={"phase_keyframes": improved_map, "chain": {"impact": 25}, "material_change": True, "reasons": []}), \
             patch("services.keyframe_service._rebind_keyframes_from_rebuilt_map", return_value=(improved_kf, [])), \
             patch("services.keyframe_service.extract_keyframes_ordered_fallback", return_value=(keyframes, {"details": [], "final_phase_keyframes": phase_map})), \
             patch("services.keyframe_service.get_video_rotation", return_value=0):
            out_kf, merged, _out_phase, src = ensure_keyframes_ordered_for_ai(
                "dummy.mp4", poses, [], dict(phase_map), list(keyframes), {"details": []}, dict(phase_map), 160
            )

        self.assertEqual(src, "forced_joint_rebuild_best_effort")
        self.assertEqual(out_kf[5]["source_pose_idx"], 25)
        self.assertNotEqual(merged.get("final_keyframe_source"), "smart_gate_failed")

    def test_quality_spacing_chain_spreads_online_tight_strip(self):
        """Regression (Modal): monotonic+semantic OK but NEAR_DUPLICATE/TIME_TOO_CLOSE on tight post-top SPI."""
        n = 80
        poses = [_mk_pose(i) for i in range(n)]
        tight = {
            "address": 10,
            "takeaway": 27,
            "backswing": 34,
            "top": 66,
            "downswing": 67,
            "impact": 70,
            "follow_through": 72,
            "finish": 78,
        }
        spi_seq = [tight[p] for p in PHASE_ORDER]
        self.assertEqual(spi_seq, [10, 27, 34, 66, 67, 70, 72, 78])
        for a, b in zip(spi_seq, spi_seq[1:]):
            self.assertLess(a, b)

        speed = np.concatenate([
            np.linspace(0.05, 0.32, 62),
            np.linspace(0.38, 1.0, 10),
            np.linspace(0.92, 0.30, 8),
        ])
        speed = np.concatenate([speed, np.linspace(0.28, 0.05, n - len(speed))])[:n]
        valid = np.ones(n, dtype=bool)
        q = np.full(n, 0.9, dtype=float)
        kin = {
            "n": n,
            "speed_s": speed,
            "valid": valid,
            "q": q,
            "xf_d": -np.abs(np.gradient(speed)),
            "hand_hip": np.linspace(0.88, 0.2, n),
        }
        with patch("services.swing_flow_utils._build_view_agnostic_kinematics", return_value=kin), \
             patch("services.swing_flow_utils.detect_phase_events_agnostic", return_value={"excursion_apex_idx": 60}), \
             patch("services.swing_flow_utils.validate_impact_semantic_at_index", side_effect=lambda i, *_: (True, {})), \
             patch("services.swing_flow_utils.validate_follow_through_semantic_at_index", side_effect=lambda *a, **k: (True, {})), \
             patch("services.swing_flow_utils.validate_finish_semantic_at_index", side_effect=lambda *a, **k: (True, {})):
            new_m, material = joint_rebuild_phase_map_for_quality_spacing(poses, tight, fps=30.0)
        self.assertTrue(material, "quality-spacing rebuild must change the phase map vs tight strip")
        self.assertTrue(
            any(int(new_m[pid]) != int(tight[pid]) for pid in ("downswing", "impact", "follow_through", "finish")),
            "at least one post-top phase index must move to widen the strip",
        )
        after_spi = [int(new_m[p]) for p in PHASE_ORDER]
        for a, b in zip(after_spi, after_spi[1:]):
            self.assertLess(a, b)
        old_tail_gap = tight["finish"] - tight["top"]
        new_tail_gap = new_m["finish"] - new_m["top"]
        self.assertGreaterEqual(new_tail_gap, old_tail_gap)

    def test_quality_spacing_propose_prefers_wider_chain_over_anchor_tie(self):
        """``propose_post_impact_chain_indices`` can no-op on duplicate/time-close strips; quality search should not."""
        n = 90
        poses = [_mk_pose(i) for i in range(n)]
        speed = np.concatenate([
            np.linspace(0.05, 0.35, 55),
            np.linspace(0.4, 1.0, 14),
            np.linspace(0.95, 0.32, 12),
            np.linspace(0.28, 0.06, 9),
        ])
        speed = speed[:n]
        valid = np.ones(n, dtype=bool)
        q = np.full(n, 0.9, dtype=float)
        kin = {"n": n, "speed_s": speed, "valid": valid, "q": q, "xf_d": -np.abs(np.gradient(speed)), "hand_hip": np.linspace(0.85, 0.18, n)}
        tight = {
            "top": 52,
            "downswing": 56,
            "impact": 59,
            "follow_through": 61,
            "finish": 64,
        }
        phase_seed = {
            "address": 5,
            "takeaway": 14,
            "backswing": 28,
            "top": tight["top"],
            "downswing": tight["downswing"],
            "impact": tight["impact"],
            "follow_through": tight["follow_through"],
            "finish": tight["finish"],
        }
        with patch("services.swing_flow_utils._build_view_agnostic_kinematics", return_value=kin), \
             patch("services.swing_flow_utils.detect_phase_events_agnostic", return_value={"excursion_apex_idx": 48}), \
             patch("services.swing_flow_utils.validate_impact_semantic_at_index", side_effect=lambda i, *_: (True, {})), \
             patch("services.swing_flow_utils.validate_follow_through_semantic_at_index", side_effect=lambda *a, **k: (True, {})), \
             patch("services.swing_flow_utils.validate_finish_semantic_at_index", side_effect=lambda *a, **k: (True, {})):
            classic = propose_post_impact_chain_indices(poses, phase_seed)
            quality = propose_quality_spacing_post_top_chain(poses, phase_seed, fps=30.0, spacing_boost=1.0)
        self.assertNotEqual(quality, {})
        seed_span = tight["finish"] - tight["downswing"]
        qual_span = quality["finish"] - quality["downswing"]
        self.assertGreater(
            qual_span,
            seed_span,
            "quality-spacing search must widen downswing→finish pose span vs the tight seed",
        )
        if classic:
            self.assertGreaterEqual(
                qual_span,
                classic["finish"] - classic["downswing"],
                "quality-spacing should not produce a shorter post-top chain than the classic proposal",
            )

    def test_ensure_strip_quality_noop_triggers_quality_spacing_without_semantic_fail(self):
        """strip_quality_noop + duplicate/time_close only + semantic_strip_ok → quality_spacing_repaired, not smart_gate_failed."""
        n = 80
        poses = [_mk_pose(i) for i in range(n)]
        bad_idxs = [10, 27, 34, 66, 67, 70, 72, 78]
        keyframes = [_mk_kf(p, bad_idxs[i], poses) for i, p in enumerate(PHASE_ORDER)]
        phase_map = {p: bad_idxs[i] for i, p in enumerate(PHASE_ORDER)}
        fail_gate = {
            "pass": False,
            "strict_contract_ok": False,
            "semantic_strip_ok": True,
            "strict_contract_fail_reasons": ["NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"],
            "semantic_strip_reasons": [],
            "final_keyframe_order_ok": True,
            "final_keyframe_time_order_ok": True,
            "final_phase_keyframes_sync_ok": True,
            "negative_time_gap_in_details": False,
        }
        ok_gate = {
            **fail_gate,
            "pass": True,
            "strict_contract_ok": True,
            "strict_contract_fail_reasons": [],
            "semantic_strip_reasons": [],
        }
        spread_map = dict(phase_map)
        spread_map.update({"downswing": 68, "impact": 71, "follow_through": 75, "finish": 78})
        better_kf = [_mk_kf(p, spread_map[p], poses) for p in PHASE_ORDER]

        cap_mock = MagicMock()
        cap_mock.return_value.isOpened.return_value = True
        cap_mock.return_value.get.return_value = 30.0

        with patch("services.keyframe_service.cv2.VideoCapture", cap_mock), \
             patch("services.keyframe_service.validate_final_keyframes_for_ai", side_effect=[fail_gate, ok_gate]), \
             patch("services.keyframe_service._reselect_strip_quality_failures", return_value=([dict(k) for k in keyframes], [])), \
             patch(
                 "services.keyframe_service.joint_rebuild_phase_map_for_quality_spacing",
                 return_value=(spread_map, True),
             ), \
             patch("services.keyframe_service._rebind_keyframes_from_rebuilt_map", return_value=(better_kf, [])), \
             patch("services.keyframe_service.rebuild_phase_map_from_event_anchors", return_value={"rebuild_ok": False}), \
             patch("services.keyframe_service.extract_keyframes_ordered_fallback", return_value=(keyframes, {"details": [], "final_phase_keyframes": phase_map})), \
             patch("services.keyframe_service.get_video_rotation", return_value=0):
            out_kf, merged, _out_phase, src = ensure_keyframes_ordered_for_ai(
                "dummy.mp4", poses, [], dict(phase_map), list(keyframes), {"details": []}, dict(phase_map), 160
            )

        self.assertEqual(src, "quality_spacing_repaired")
        self.assertTrue(merged["final_keyframe_gate_pass"])
        self.assertEqual(merged.get("final_keyframe_source"), "quality_spacing_repaired")
        self.assertEqual(out_kf[5]["source_pose_idx"], 71)

    def test_joint_rebuild_changes_modal_log_corrupt_strip(self):
        """Regression: Modal log strip had monotonic pose tail broken (impact pose < downswing pose)."""
        n = 120
        poses = [_mk_pose(i) for i in range(n)]
        corrupt = {
            "address": 10,
            "takeaway": 27,
            "backswing": 34,
            "top": 66,
            "downswing": 82,
            "impact": 74,
            "follow_through": 78,
            "finish": 81,
        }
        before_spi = [corrupt[p] for p in PHASE_ORDER]
        before_sfi = [int(poses[corrupt[p]]["frame_index"]) for p in PHASE_ORDER]
        self.assertGreater(before_spi[4], before_spi[5])  # downswing pose > impact pose (bug)
        self.assertGreater(before_sfi[4], before_sfi[5])  # frame order also inverted vs labels

        new_m, material = joint_rebuild_phase_map_for_monotonic_strip(poses, corrupt, fps=30.0)
        self.assertTrue(material)
        after_spi = [new_m[p] for p in PHASE_ORDER]
        after_sfi = [int(poses[new_m[p]]["frame_index"]) for p in PHASE_ORDER]
        self.assertNotEqual(after_spi, before_spi)
        for a, b in zip(after_spi, after_spi[1:]):
            self.assertLess(a, b)
        for a, b in zip(after_sfi, after_sfi[1:]):
            self.assertLess(a, b)
        self.assertLess(new_m["downswing"], new_m["impact"])
        self.assertLess(new_m["impact"], new_m["follow_through"])
        self.assertLess(new_m["follow_through"], new_m["finish"])

    def test_online_glued_spi_report_material_change_from_chain_solver(self):
        """User-reported glued post-impact SPI; primary chain may no-op — spacing fallback must alter map."""
        from unittest.mock import patch

        from services.phase_chain_solver_service import solve_post_impact_phase_chain

        n = 96
        poses = [_mk_pose(i) for i in range(n)]
        tight = {p: v for p, v in zip(PHASE_ORDER, [10, 27, 34, 66, 74, 78, 80, 82])}
        wider = {"downswing": 70, "impact": 76, "follow_through": 86, "finish": 93}
        with patch("services.swing_flow_utils.propose_post_impact_chain_indices", return_value={}), \
             patch("services.swing_flow_utils.propose_quality_spacing_post_top_chain", return_value=wider):
            out = solve_post_impact_phase_chain(poses, tight, tracks=None)
        self.assertTrue(out.get("material_change"), msg=str(out.get("reasons")))
        self.assertIn("CHAIN_FROM_QUALITY_SPACING_FALLBACK", out.get("reasons", []))
        for k, v in wider.items():
            self.assertEqual(out["phase_keyframes"][k], v)


if __name__ == "__main__":
    unittest.main()
