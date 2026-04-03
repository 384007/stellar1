"""Regression: authoritative phase chain handoff into keyframe ensure pipeline."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.keyframe_service import PHASE_ORDER, try_recover_keyframes_with_authoritative_phase_map


class TestAuthoritativeKeyframeRecovery(unittest.TestCase):
    def test_recovery_returns_ordered_strip_when_gate_passes(self):
        poses = [{"frame_index": i, "timestamp": i / 30.0} for i in range(80)]
        swing_phases: list[dict] = []
        auth_map = {p: 5 + i * 8 for i, p in enumerate(PHASE_ORDER)}
        fake_kf = [{"phase": p, "source_pose_idx": auth_map[p]} for p in PHASE_ORDER]
        fake_gate = {"pass": True, "final_keyframe_order_ok": True, "final_keyframe_time_order_ok": True, "final_phase_keyframes_sync_ok": True, "negative_time_gap_in_details": False}

        def _fake_ordered(*_a, **_k):
            return fake_kf, {
                "details": [],
                "final_phase_keyframes": dict(auth_map),
                "near_duplicates": 0,
                "time_too_close": 0,
                "all_passed": True,
                "source": "ordered_fallback",
            }

        merged_base = {"final_keyframe_source": "smart_gate_failed", "details": []}
        with patch("services.keyframe_service.extract_keyframes_ordered_fallback", side_effect=_fake_ordered), patch(
            "services.keyframe_service.validate_final_keyframes_for_ai", return_value=fake_gate,
        ):
            out = try_recover_keyframes_with_authoritative_phase_map(
                "dummy.mp4",
                poses,
                swing_phases,
                320,
                30.0,
                {},
                dict(auth_map),
                True,
                merged_base,
            )
        self.assertIsNotNone(out)
        kfs, merged, phase, src = out
        self.assertEqual(src, "ordered_fallback_authoritative_chain")
        self.assertTrue(merged.get("final_keyframe_gate_pass"))
        self.assertEqual(merged.get("authoritative_chain_recovery_seed"), "authoritative_pre_extract_raw")
        self.assertEqual(len(kfs), 8)
        self.assertEqual(merged.get("final_keyframe_source"), "ordered_fallback_authoritative_chain")

    def test_recovery_tries_respaced_seed_when_raw_fails_gate(self):
        poses = [{"frame_index": i, "timestamp": i / 30.0} for i in range(80)]
        swing_phases: list[dict] = []
        auth_map = {p: 5 + i * 8 for i, p in enumerate(PHASE_ORDER)}
        fake_kf = [{"phase": p, "source_pose_idx": auth_map[p]} for p in PHASE_ORDER]
        pass_gate = {"pass": True, "final_keyframe_order_ok": True, "final_keyframe_time_order_ok": True, "final_phase_keyframes_sync_ok": True, "negative_time_gap_in_details": False}
        fail_gate = {"pass": False, "final_keyframe_order_ok": False, "final_keyframe_time_order_ok": False, "final_phase_keyframes_sync_ok": False, "negative_time_gap_in_details": False}

        calls: list[str] = []

        def _fake_ordered(_vp, _ps, _sw, seed, **_k):
            label = "raw" if seed == auth_map else "respaced"
            calls.append(label)
            return fake_kf, {
                "details": [],
                "final_phase_keyframes": dict(seed),
                "source": "ordered_fallback",
            }

        def _fake_validate(_kf, _ph, _det, **_k):
            return pass_gate if len(calls) >= 2 else fail_gate

        merged_base: dict = {}
        with patch("services.keyframe_service.extract_keyframes_ordered_fallback", side_effect=_fake_ordered), patch(
            "services.keyframe_service.validate_final_keyframes_for_ai", side_effect=_fake_validate,
        ):
            out = try_recover_keyframes_with_authoritative_phase_map(
                "dummy.mp4",
                poses,
                swing_phases,
                320,
                30.0,
                {},
                dict(auth_map),
                True,
                merged_base,
            )
        self.assertIsNotNone(out)
        self.assertEqual(out[3], "ordered_fallback_authoritative_chain")
        self.assertEqual(out[1].get("authoritative_chain_recovery_seed"), "authoritative_pre_extract_respaced")
        self.assertEqual(calls, ["raw", "respaced"])

    def test_recovery_skips_when_authoritative_not_ok(self):
        poses = [{"frame_index": i, "timestamp": i / 30.0} for i in range(80)]
        auth_map = {p: i for i, p in enumerate(PHASE_ORDER)}
        out = try_recover_keyframes_with_authoritative_phase_map(
            "dummy.mp4",
            poses,
            [],
            320,
            30.0,
            {},
            dict(auth_map),
            False,
            {},
        )
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
