"""Contract tests: phase evaluation reliability gate + keyframe/pose index sync (no video I/O)."""
import sys
import unittest
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.swing_flow_utils import (
    assess_gemini_uniform_map_vs_final_phase_strip,
    build_phase_boundary_flags,
    compute_phase_evaluations_reliable,
)
from services.keyframe_service import (
    PHASE_ORDER,
    SWING_PHASE_META,
    sync_keyframes_phase_map_and_pose_fields,
    validate_final_keyframes_for_ai,
)


class TestPhaseEvalReliable(unittest.TestCase):
    def test_har_contract_all_conditions(self):
        self.assertTrue(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=True,
                phase_validation_passed=True,
                final_keyframe_source="smart",
                final_keyframe_gate_pass=True,
            )
        )
        self.assertFalse(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=True,
                phase_validation_passed=True,
                final_keyframe_source="ordered_fallback",
                final_keyframe_gate_pass=True,
            )
        )
        self.assertFalse(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=True,
                phase_validation_passed=True,
                final_keyframe_source="ordered_fallback_empty",
                final_keyframe_gate_pass=True,
            )
        )
        self.assertFalse(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=False,
                phase_validation_passed=True,
                final_keyframe_source="smart",
                final_keyframe_gate_pass=True,
            )
        )
        self.assertFalse(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=True,
                phase_validation_passed=False,
                final_keyframe_source="smart",
                final_keyframe_gate_pass=True,
            )
        )
        self.assertFalse(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=True,
                phase_validation_passed=True,
                final_keyframe_source="smart",
                final_keyframe_gate_pass=False,
            )
        )
        self.assertFalse(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=True,
                phase_validation_passed=True,
                final_keyframe_source="smart",
                final_keyframe_gate_pass=True,
                ai_vision_frame_count=5,
                keyframe_strip_frame_count=8,
            )
        )
        self.assertFalse(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=True,
                phase_validation_passed=True,
                final_keyframe_source="smart",
                final_keyframe_gate_pass=True,
                gemini_uniform_map_applies=True,
                gemini_map_aligned_with_final_strip=False,
            )
        )
        self.assertTrue(
            compute_phase_evaluations_reliable(
                final_phase_semantic_ok=True,
                phase_validation_passed=True,
                final_keyframe_source="smart",
                final_keyframe_gate_pass=True,
                ai_vision_frame_count=8,
                keyframe_strip_frame_count=8,
                gemini_uniform_map_applies=True,
                gemini_map_aligned_with_final_strip=True,
            )
        )

    def test_gemini_assess_alignment(self):
        g = {p: i * 2 for i, p in enumerate([
            "address", "takeaway", "backswing", "top",
            "downswing", "impact", "follow_through", "finish",
        ])}
        f = dict(g)
        a = assess_gemini_uniform_map_vs_final_phase_strip("gemini", g, f)
        self.assertTrue(a["gemini_uniform_thumbnail_map_applies"])
        self.assertTrue(a["gemini_map_aligned_with_final_strip"])
        self.assertIs(a["aligned"], True)
        f2 = dict(f)
        f2["top"] = f["top"] + 10
        a2 = assess_gemini_uniform_map_vs_final_phase_strip("gemini", g, f2)
        self.assertFalse(a2["gemini_map_aligned_with_final_strip"])
        self.assertIs(a2["aligned"], False)
        a3 = assess_gemini_uniform_map_vs_final_phase_strip("kinematic", g, f2)
        self.assertIsNone(a3["gemini_map_aligned_with_final_strip"])
        self.assertIsNone(a3["aligned"])

    def test_plus_like_gemini_smart_drift_makes_reliable_false(self):
        """Integration-shaped: gemini* source + final strip moved vs gemini_map => not reliable."""
        g = {p: i for i, p in enumerate([
            "address", "takeaway", "backswing", "top",
            "downswing", "impact", "follow_through", "finish",
        ])}
        final = dict(g)
        final["top"] = g["top"] + 5
        assess = assess_gemini_uniform_map_vs_final_phase_strip("gemini", g, final)
        self.assertFalse(assess["aligned"])
        rel = compute_phase_evaluations_reliable(
            final_phase_semantic_ok=True,
            phase_validation_passed=True,
            final_keyframe_source="smart",
            final_keyframe_gate_pass=True,
            ai_vision_frame_count=8,
            keyframe_strip_frame_count=8,
            gemini_uniform_map_applies=True,
            gemini_map_aligned_with_final_strip=assess["gemini_map_aligned_with_final_strip"],
        )
        self.assertFalse(rel)
        warn = "semantic_validation_failed;gemini_uniform_map_diverged_from_strip"
        self.assertIn("gemini_uniform_map_diverged_from_strip", warn)

    def test_build_phase_boundary_ordered_fallback_label(self):
        assess_empty = assess_gemini_uniform_map_vs_final_phase_strip("kinematic", None, None)
        pb = build_phase_boundary_flags(
            final_keyframe_source="ordered_fallback",
            keyframe_strip_frame_count=8,
            ai_vision_frame_count=8,
            gemini_strip_assessment=assess_empty,
            analysis_route="plus",
            plus_grade_phase_evaluations=True,
        )
        self.assertTrue(pb["phase_strip_is_monotonic_fallback_only"])
        self.assertEqual(pb["phase_keyframe_extraction_label"], "monotonic_pose_fallback")
        self.assertTrue(pb["plus_grade_phase_evaluations"])


class TestKeyframeSync(unittest.TestCase):
    def test_sync_aligns_frame_index_and_phase_map(self):
        poses = [
            {"frame_index": 10 + i * 10, "timestamp": 0.1 + i * 0.05}
            for i in range(24)
        ]
        phase_keyframes = {ph: i for i, ph in enumerate(PHASE_ORDER)}
        keyframes = []
        for i, ph in enumerate(PHASE_ORDER):
            meta = SWING_PHASE_META[ph]
            keyframes.append(
                {
                    "phase": ph,
                    "label_en": meta["label_en"],
                    "label_zh": meta["label_zh"],
                    "source_pose_idx": i,
                    "frame_index": 999,
                    "timestamp": None,
                    "image_base64": "qq==",
                }
            )
        sync_keyframes_phase_map_and_pose_fields(poses, phase_keyframes, keyframes)
        self.assertEqual(keyframes[0]["frame_index"], 10)
        self.assertEqual(keyframes[0]["source_frame_index"], 10)
        self.assertEqual(keyframes[1]["source_pose_idx"], 1)
        self.assertEqual(keyframes[1]["frame_index"], 20)
        self.assertEqual(phase_keyframes["top"], 3)
        gate = validate_final_keyframes_for_ai(keyframes, phase_keyframes, [], poses=None)
        self.assertTrue(gate["final_phase_keyframes_sync_ok"])
        self.assertTrue(gate["strict_contract_ok"])
        self.assertFalse(gate.get("semantic_strip_ok", False))
        self.assertFalse(gate["pass"])


if __name__ == "__main__":
    unittest.main()
