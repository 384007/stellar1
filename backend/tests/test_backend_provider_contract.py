import unittest
from unittest.mock import patch

from services.api_pack_service import pack_plus_response
from services.detection_provider_service import get_video_detections
from services.motion_3d_service import lift_motion_3d
from services.pose_backend_service import extract_pose_stream


class TestBackendProviderContract(unittest.TestCase):
    def test_pose_backend_switch_preserves_schema(self):
        fake = {
            "poses": [{"frame_index": 1, "timestamp": 0.1, "joints": []}],
            "pose_quality_bundle": {"ok": True},
            "landmarks_2d": [[]],
            "landmarks_3d": [[]],
            "visibility": [[]],
            "timestamps": [0.1],
            "frame_indices": [1],
            "world_landmarks": [[]],
            "backend_profile": "high_precision",
        }
        with patch("services.pose_backend_service._pose_mode", return_value="high_precision"), \
             patch(
                 "services.providers.pose_rtmpose_provider.run",
                 return_value={"status": "dependency_missing", "payload": {"poses": [], "pose_quality_bundle": {}}},
             ), \
             patch("services.pose_backend_service.pose_mediapipe_provider.run", return_value={
                 "provider_name": "mediapipe",
                 "payload": {"poses": fake["poses"], "pose_quality_bundle": fake["pose_quality_bundle"]},
             }):
            out = extract_pose_stream("demo.mp4", max_frames=3)
        self.assertIn("poses", out)
        self.assertIn("landmarks_2d", out)
        self.assertIn("timestamps", out)
        self.assertIn("frame_indices", out)

    def test_detection_and_motion_disabled_do_not_break(self):
        with patch("os.getenv", side_effect=lambda k, d=None: "disabled" if "MODE" in k else (d or "")):
            det = get_video_detections("demo.mp4", poses=[])
            mot = lift_motion_3d([])
        self.assertFalse(det["enabled"])
        self.assertTrue(det.get("yolo11_degraded"))
        self.assertFalse(mot["enabled"])

    def test_response_contract_hides_technical_terms(self):
        raw = {
            "analysis_id": "a1",
            "type": "plus",
            "analysis_mode": "pose_plus_partial",
            "keyframes": [],
            "phase_keyframes": {},
            "prediction": {"note": "use provider-x"},
            "trajectory": {},
            "scores": {},
            "issues": ["ultralytics was enabled"],
            "training": {},
            "analysis_reliability": {},
            "phase_source": "kinematic_degraded",
            "final_keyframe_gate_pass": False,
            "partial_mode": True,
            "keyframe_display_mode": "degraded_failed",
        }
        out = pack_plus_response(raw)
        self.assertIn("result_partial", out)
        self.assertIn("image_missing", out)
        self.assertIn("final_keyframe_gate_pass", out)
        self.assertIn("phase_source", out)
        self.assertEqual(out["issues"], [""])
        self.assertTrue(out["result_partial"])

    def test_pack_sets_image_missing_when_any_keyframe_image_empty(self):
        raw = {
            "analysis_id": "a2",
            "type": "plus",
            "keyframes": [{"phase": "finish", "image_base64": ""}],
            "phase_keyframes": {"finish": 10},
            "prediction": {},
            "trajectory": {},
            "scores": {},
            "issues": [],
            "training": {},
            "analysis_reliability": {},
            "final_keyframe_gate_pass": False,
        }
        out = pack_plus_response(raw)
        self.assertTrue(out["image_missing"])
        self.assertTrue(out["result_partial"])

    def test_pack_hides_phase_debug_and_yolo_flags(self):
        raw = {
            "analysis_id": "a3b",
            "type": "plus",
            "keyframes": [],
            "phase_keyframes": {},
            "prediction": {},
            "trajectory": {},
            "scores": {},
            "issues": [],
            "training": {},
            "analysis_reliability": {},
            "final_keyframe_gate_pass": False,
            "phase_debug": {"gemini_raw": {"x": 1}},
            "yolo11_status": "ok",
            "yolo11_degraded": False,
        }
        out = pack_plus_response(raw)
        self.assertNotIn("phase_debug", out)
        self.assertNotIn("yolo11_status", out)
        self.assertNotIn("yolo11_degraded", out)

    def test_pack_hides_provider_technical_fields(self):
        raw = {
            "analysis_id": "a3",
            "type": "plus",
            "keyframes": [],
            "phase_keyframes": {},
            "prediction": {},
            "trajectory": {},
            "scores": {},
            "issues": [],
            "training": {},
            "analysis_reliability": {},
            "final_keyframe_gate_pass": False,
            "optional_modules": {"detection_active": False, "tracking_active": False},
            "provider_debug": {"pose_provider": {"provider_name": "mediapipe"}},
        }
        out = pack_plus_response(raw)
        self.assertNotIn("optional_modules", out)
        self.assertNotIn("provider_debug", out)
        self.assertNotIn("provider_summary", out)
        self.assertTrue(out["result_partial"])

    def test_pack_preserves_pose_and_video_meta_for_overlay(self):
        raw = {
            "analysis_id": "a4",
            "type": "plus",
            "keyframes": [{"phase": "address", "image_base64": "e30="}],
            "phase_keyframes": {"address": 0},
            "prediction": {"predicted_distance": 220.0},
            "trajectory": {},
            "scores": {},
            "issues": [],
            "training": {},
            "analysis_reliability": {},
            "final_keyframe_gate_pass": True,
            "pose_frames": [
                {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "joints": [{"name": "left_wrist", "x": 1, "y": 2, "z": 0, "visibility": 0.9, "normalized": {"x": 0.5, "y": 0.5}}],
                    "connections": [[0, 0]],
                    "angles": {"x_factor": 40.0},
                    "frame_size": {"width": 320, "height": 568},
                }
            ],
            "skeleton_data": {"frames": [], "total_frames": 0},
            "video_meta": {"fps": 30.0, "source_frame_count": 90, "total_pose_frames": 1},
            "swing_phases": [],
            "summary_zh": "测试摘要",
        }
        out = pack_plus_response(raw)
        self.assertIn("pose_frames", out)
        self.assertEqual(len(out["pose_frames"]), 1)
        self.assertIn("video_meta", out)
        self.assertEqual(out["video_meta"].get("source_frame_count"), 90)
        self.assertIn("skeleton_data", out)
        self.assertEqual(out.get("summary_zh"), "测试摘要")

    def test_pack_preserves_keyframe_base64_even_if_substring_matches_forbidden_token(self):
        """JPEG base64 is pseudo-random; 'yolo' / 'mmaction' etc. can appear by chance — must not wipe images."""
        b64 = "yoloMMActionMediapipeUltralytics" + ("Z" * 500)
        raw = {
            "analysis_id": "a6",
            "type": "plus",
            "keyframes": [{"phase": "address", "label_en": "A", "label_zh": "准备", "timestamp": 0.0, "image_base64": b64}],
            "phase_keyframes": {"address": 0},
            "prediction": {},
            "trajectory": {},
            "scores": {},
            "issues": [],
            "training": {},
            "analysis_reliability": {},
            "final_keyframe_gate_pass": True,
        }
        out = pack_plus_response(raw)
        self.assertEqual(out["keyframes"][0]["image_base64"], b64)
        self.assertFalse(out["image_missing"])

    def test_pack_forwards_gemini_observation(self):
        gem = {
            "available": True,
            "mode": "observation_only",
            "summary_zh": "可见帧观察",
            "summary_en": "Visible-frame note",
            "bullets_zh": [],
            "bullets_en": [],
        }
        raw = {
            "analysis_id": "a5",
            "type": "plus",
            "keyframes": [{"phase": "address", "image_base64": "e30="}],
            "phase_keyframes": {"address": 0},
            "prediction": {},
            "trajectory": {},
            "scores": {},
            "issues": [],
            "training": {},
            "analysis_reliability": {},
            "final_keyframe_gate_pass": False,
            "final_keyframe_source": "smart_gate_failed",
            "gemini_observation": gem,
        }
        out = pack_plus_response(raw)
        self.assertEqual(out.get("gemini_observation"), gem)


if __name__ == "__main__":
    unittest.main()
