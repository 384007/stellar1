"""Phase C: temporal segmentation context fed into Plus Gemini prompt."""
from __future__ import annotations

import unittest

from services.phase_segment_service import compact_phase_c_context_for_plus_prompt


class TestPhaseCPlusPrompt(unittest.TestCase):
    def test_compact_shape(self) -> None:
        seg = {
            "temporal_prior_strength": 0.12,
            "phase_boundaries": [{"phase_id": "address", "start_idx": 0, "end_idx": 2}],
            "phase_confidence": {
                "clip_action_confidence": 0.4,
                "window_count": 2,
                "window_confidence_mean": 0.2,
                "boundary_confidence_mean": 0.8,
                "global_segmentation_confidence": 0.62,
            },
            "action_provider_meta": {"status": "ok", "provider_name": "mmaction2_stub"},
        }
        out = compact_phase_c_context_for_plus_prompt(seg)
        self.assertEqual(out["phase_c_version"], "1")
        self.assertEqual(out["temporal_prior_strength"], 0.12)
        self.assertEqual(out["phase_boundary_segment_count"], 1)
        self.assertEqual(out["action_backend"]["status"], "ok")
        self.assertEqual(out["action_backend"]["name"], "mmaction2_stub")
        self.assertAlmostEqual(out["phase_confidence"]["global_segmentation_confidence"], 0.62)

    def test_empty_seg_stable(self) -> None:
        out = compact_phase_c_context_for_plus_prompt({})
        self.assertEqual(out["phase_c_version"], "1")
        self.assertEqual(out["phase_boundary_segment_count"], 0)
        self.assertEqual(out["action_backend"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
