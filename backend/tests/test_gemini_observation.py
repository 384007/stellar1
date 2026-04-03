import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services import gemini_service


class TestGeminiObservationPayload(unittest.TestCase):
    def test_observation_payload_present_when_frames_available(self):
        async def _run():
            with patch.object(
                gemini_service,
                "_call_vision_ai",
                return_value=(
                    '{"summary_zh":"观察到上肢抬升","summary_en":"Observed arm lift",'
                    '"bullets_zh":["第4张可见帧里手部更高"],'
                    '"bullets_en":["Visible frame 4 shows higher hands"],'
                    '"frame_notes":[{"index":1,"note_zh":"第1张可见帧稳定","note_en":"Visible frame 1 stable"}]}',
                    "gemini",
                    1,
                ),
            ):
                out = await gemini_service.analyze_plus_visual_observation(
                    ["img1", "img2", "img3", "img4"],
                    frame_labels=["address", "takeaway", "top", "impact"],
                    phase_labels_trusted=False,
                    source="degraded_display_keyframes",
                    issues=["TIME_TOO_CLOSE_PRESENT"],
                )
                self.assertTrue(out["available"])
                self.assertEqual(out["mode"], "observation_only")
                self.assertFalse(out["used_as_authoritative_source"])
                self.assertEqual(out["source"], "degraded_display_keyframes")
                self.assertTrue(len(out["summary_zh"]) > 0)
                self.assertTrue(len(out["bullets_zh"]) > 0)
                self.assertEqual(len(out["frame_notes"]), 4)

        asyncio.run(_run())

    def test_fallback_wording_uses_visible_frame_language(self):
        async def _run():
            with patch.object(gemini_service, "_call_vision_ai", side_effect=RuntimeError("boom")):
                out = await gemini_service.analyze_plus_visual_observation(
                    ["img1", "img2"],
                    frame_labels=[None, None],
                    phase_labels_trusted=False,
                    source="uniform_frames",
                )
                self.assertTrue(out["available"])
                self.assertEqual(out["mode"], "observation_only")
                self.assertIn("Visible frame 1", out["frame_notes"][0]["note_en"])
                self.assertIn("第1张可见帧", out["frame_notes"][0]["note_zh"])

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()

