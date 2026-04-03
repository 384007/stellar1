import unittest
from unittest.mock import patch

from services.detection_provider_service import get_video_detections
from services.object_tracking_service import build_tracks_from_detections
from services.motion_3d_service import lift_motion_3d
from services.phase_segment_service import segment_swing_phases
from services.phase_chain_solver_service import solve_full_phase_chain
from services.biomech_validation_service import validate_phase_chain_hard
from services.pose_backend_service import extract_pose_stream


class TestProviderRuntimeContract(unittest.TestCase):
    @staticmethod
    def _poses(n: int = 24):
        poses = []
        for i in range(n):
            poses.append({
                "frame_index": i,
                "timestamp": i / 30.0,
                "pose_quality": 0.7 + (0.2 if 10 <= i <= 18 else 0.0),
                "joints": [
                    {"name": "left_wrist", "visibility": 0.9, "normalized": {"x": 0.35 + i * 0.01, "y": 0.5}},
                    {"name": "right_wrist", "visibility": 0.9, "normalized": {"x": 0.45 + i * 0.01, "y": 0.5}},
                    {"name": "left_hip", "visibility": 0.9, "normalized": {"x": 0.4, "y": 0.7}},
                    {"name": "right_hip", "visibility": 0.9, "normalized": {"x": 0.5, "y": 0.7}},
                    {"name": "left_shoulder", "visibility": 0.9, "normalized": {"x": 0.4, "y": 0.4}},
                    {"name": "right_shoulder", "visibility": 0.9, "normalized": {"x": 0.5, "y": 0.4}},
                ],
                "angles": {"x_factor": 10 + i, "shoulder_rotation": 20 + i},
                "detection": {"angles": {"x_factor": 10 + i, "shoulder_rotation": 20 + i}},
            })
        return poses

    def test_detection_disabled_reports_disabled_status(self):
        with patch("os.getenv", side_effect=lambda k, d=None: "disabled" if k == "STELLAR_DETECTION_BACKEND" else (d or "")):
            out = get_video_detections("demo.mp4", poses=[])
        self.assertEqual(out["provider"], "yolo11")
        self.assertEqual(out["status"], "unavailable")
        self.assertFalse(out["enabled"])

    def test_tracking_disabled_reports_disabled_status(self):
        with patch("os.getenv", side_effect=lambda k, d=None: "disabled" if k == "STELLAR_TRACKING_BACKEND" else (d or "")):
            out = build_tracks_from_detections({"detections": [{"class_name": "person", "frame_index": 1, "confidence": 0.7}]})
        self.assertEqual(out["provider"], "disabled")
        self.assertEqual(out["status"], "disabled")

    def test_motion3d_reports_unavailable_when_provider_fails(self):
        with patch("services.providers.pose3d_motionbert_provider.run", return_value={"provider_name": "motionbert", "status": "dependency_missing", "payload": {}}), \
             patch("services.providers.pose3d_mediapipe_world_provider.run", return_value={"provider_name": "mediapipe_world", "status": "insufficient_joints", "payload": {}}):
            out = lift_motion_3d([{"frame_index": 1, "timestamp": 0.1}])
        self.assertFalse(out["enabled"])
        self.assertEqual(out["status"], "insufficient_joints")

    def test_motion3d_mediapipe_world_fallback_when_motionbert_unavailable(self):
        poses = [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "joints": [
                    {"name": "nose", "x": 10.0, "y": 20.0, "z": 1.0},
                    {"name": "left_shoulder", "x": 0.0, "y": 0.0, "z": 0.0},
                ],
                "world_landmarks": {
                    "nose": {"x": 0.01, "y": 0.02, "z": 0.03},
                    "left_shoulder": {"x": 0.1, "y": 0.2, "z": 0.3},
                },
            }
        ]
        with patch(
            "services.providers.pose3d_motionbert_provider.run",
            return_value={"provider_name": "motionbert", "status": "checkpoint_unset", "payload": {}},
        ):
            out = lift_motion_3d(poses)
        self.assertTrue(out["enabled"])
        self.assertEqual(out["provider"], "mediapipe_world")
        self.assertEqual(len(out["motion_3d"]), 1)
        self.assertEqual(out["motion_3d"][0][0], [0.01, 0.02, 0.03])

    def test_phase_windows_and_chain_use_detection_tracking(self):
        poses = self._poses(24)
        detections = []
        for i in range(8, 20):
            detections.append({"frame_index": i, "class_name": "person", "confidence": 0.9})
        detections += [
            {"frame_index": 15, "class_name": "club", "confidence": 0.92},
            {"frame_index": 16, "class_name": "ball", "confidence": 0.88},
        ]
        tracks = {
            "person_tracks": [{"frame_index": i, "track_id": 1} for i in range(8, 20)],
            "club_tracks": [{"frame_index": 15, "track_id": 2}],
            "ball_tracks": [{"frame_index": 16, "track_id": 3}],
        }
        seg = segment_swing_phases(poses, tracks=tracks, detections=detections, video_path="demo.mp4")
        self.assertGreater(len(seg.get("phase_boundaries") or []), 0)
        self.assertGreater(len(seg.get("phase_logits") or []), 0)
        impact_w = seg["phase_windows"]["impact"]
        self.assertGreaterEqual(impact_w[0], 8)
        self.assertLessEqual(impact_w[1], 22)
        coarse = dict(seg["phase_keyframes"])
        solved = solve_full_phase_chain(poses, coarse, seg["phase_windows"], detections, tracks)
        self.assertTrue(solved["ok"])
        self.assertIn("impact", solved["evidence"])
        self.assertNotEqual(solved["phase_keyframes"], coarse)
        self.assertIn("det_classes", solved["evidence"]["impact"])
        self.assertIn("track_supported", solved["evidence"]["impact"])

    def test_impact_window_not_simple_coarse_radius(self):
        poses = self._poses(30)
        detections = [{"frame_index": 22, "class_name": "club", "confidence": 0.95}]
        tracks = {"person_tracks": [{"frame_index": i, "track_id": 1} for i in range(10, 29)]}
        seg = segment_swing_phases(poses, tracks=tracks, detections=detections, video_path="demo.mp4")
        coarse_impact = int(seg["phase_keyframes"]["impact"])
        w = seg["phase_windows"]["impact"]
        center = (w[0] + w[1]) / 2.0
        self.assertGreater(abs(center - coarse_impact), 1.0)
        self.assertGreaterEqual(w[0], 18)

    def test_high_precision_request_does_not_report_active_high_precision(self):
        fake = {"provider_name": "mediapipe", "payload": {"poses": [], "pose_quality_bundle": {}}}
        with patch("services.pose_backend_service._pose_mode", return_value="high_precision"), \
             patch(
                 "services.providers.pose_rtmpose_provider.run",
                 return_value={"status": "dependency_missing", "payload": {"poses": [], "pose_quality_bundle": {}}},
             ), \
             patch("services.pose_backend_service.pose_mediapipe_provider.run", return_value=fake):
            out = extract_pose_stream("demo.mp4", max_frames=3)
        meta = out.get("provider_meta") or {}
        self.assertEqual(out["backend_profile"], "mediapipe")
        self.assertEqual(meta.get("active_backend"), "mediapipe")
        self.assertEqual(meta.get("requested_backend"), "high_precision")

    def test_hard_biomech_blocks_invalid_chain(self):
        poses = self._poses(20)
        bad = {
            "address": 2, "takeaway": 3, "backswing": 4, "top": 5,
            "downswing": 6, "impact": 5, "follow_through": 7, "finish": 8,
        }
        out = validate_phase_chain_hard(poses, bad)
        self.assertFalse(out["passed"])
        self.assertIn("impact_not_after_top", out["reasons"])

    def test_follow_finish_spacing_from_dynamics(self):
        poses = self._poses(28)
        detections = [{"frame_index": 17, "class_name": "club", "confidence": 0.9}, {"frame_index": 18, "class_name": "ball", "confidence": 0.9}]
        tracks = {"person_tracks": [{"frame_index": i, "track_id": 1} for i in range(6, 27)]}
        seg = segment_swing_phases(poses, tracks=tracks, detections=detections, video_path="demo.mp4")
        solved = solve_full_phase_chain(poses, seg["phase_keyframes"], seg["phase_windows"], detections, tracks)
        self.assertTrue(solved["ok"])
        fin = solved["phase_keyframes"]["finish"]
        fol = solved["phase_keyframes"]["follow_through"]
        self.assertGreater(fin, fol)
        self.assertIn("speed", solved["evidence"]["finish"])

    def test_motion3d_changes_post_impact_selection(self):
        poses = self._poses(30)
        detections = [{"frame_index": 18, "class_name": "club", "confidence": 0.9}, {"frame_index": 19, "class_name": "ball", "confidence": 0.9}]
        tracks = {"person_tracks": [{"frame_index": i, "track_id": 1} for i in range(8, 29)]}
        coarse = {"address": 2, "takeaway": 4, "backswing": 7, "top": 12, "downswing": 16, "impact": 19, "follow_through": 23, "finish": 27}
        windows = {
            "address": [0, 4], "takeaway": [3, 7], "backswing": [6, 11], "top": [10, 15],
            "downswing": [14, 18], "impact": [17, 21], "follow_through": [20, 27], "finish": [22, 29],
        }
        base = solve_full_phase_chain(poses, coarse, windows, detections, tracks, motion_3d=None)
        motion3d = []
        for i in range(30):
            z = 1.0 if i < 23 else 0.2
            motion3d.append([[0.0, 0.0, z], [0.2, 0.1, z * 0.9]])
        with_m = solve_full_phase_chain(poses, coarse, windows, detections, tracks, motion_3d=motion3d)
        self.assertTrue(base["ok"])
        self.assertTrue(with_m["ok"])
        self.assertNotEqual(
            (base["phase_keyframes"]["follow_through"], base["phase_keyframes"]["finish"]),
            (with_m["phase_keyframes"]["follow_through"], with_m["phase_keyframes"]["finish"]),
        )


if __name__ == "__main__":
    unittest.main()
