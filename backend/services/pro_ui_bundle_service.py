"""Stellar Pro UI payload: same skeleton / HUD / trajectory / prediction shape as legacy /analyze/pro.

Analysis keyframes stay on the motion-engine chain; this layer only formats poses for the frontend.
"""

from __future__ import annotations

import logging
from typing import Any

from services.handedness_service import detect_handedness
from services.hud_service import generate_hud_data
from services.pose_service import pose_for_skeleton_render
from services.shot_predictor import predict_shot
from services.swing_flow_utils import compute_wrist_trajectory, detect_swing_phases

logger = logging.getLogger(__name__)


def _empty_ui_bundle() -> dict[str, Any]:
    return {
        "skeleton_data": {
            "frames": [],
            "total_frames": 0,
            "joint_space": "analysis_frame",
            "joint_sources": {
                "raw_detection_joints": "pose.raw_detection_joints",
                "analysis_joints": "pose.joints",
                "render_joints": "pose.render_joints->pose.joints",
            },
        },
        "pose_frames": [],
        "prediction": {
            "predicted_distance": 0,
            "lateral_offset": 0,
            "shot_shape": "unknown",
            "shot_shape_zh": "未知",
            "club_head_speed": 0,
            "ball_speed": 0,
            "launch_angle": 0,
            "spin_rate": 0,
            "smash_factor": 0,
            "trajectory": [],
        },
        "trajectory": [],
        "video_meta": {
            "fps": 30.0,
            "total_pose_frames": 0,
            "duration_s": 0.0,
            "source_frame_count": 0,
        },
        "hand_detection": {
            "hand": "UNKNOWN",
            "confidence": 0.0,
            "reason": "no_pose_frames",
            "fallback_applied": True,
        },
    }


def build_stellar_pro_ui_bundle(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    *,
    fps: float,
    source_frame_count: int | None = None,
    analysis_frame_count: int | None = None,
    detected_club: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build skeleton_data (HUD frames), slim pose_frames, wrist trajectory, predict_shot output."""
    if not poses:
        return _empty_ui_bundle()

    swing_phases = detect_swing_phases(poses)
    for i, pose in enumerate(poses):
        if i < len(swing_phases):
            pose["phase_data"] = swing_phases[i]

    hand_info = detect_handedness(poses, swing_phases=swing_phases)
    resolved_hand = str(hand_info.get("hand") or "UNKNOWN")
    hand_confidence = float(hand_info.get("confidence") or 0.0)
    hand_detection: dict[str, Any] = {
        "hand": resolved_hand,
        "confidence": round(hand_confidence, 4),
        "reason": str(hand_info.get("reason") or ""),
        "fallback_applied": bool(hand_info.get("fallback_applied", False)),
    }
    logger.info(
        "[STELLAR_PRO][HAND_DETECT] hand=%s conf=%.3f reason=%s",
        resolved_hand,
        hand_confidence,
        hand_detection.get("reason") or "n/a",
    )

    hud_frames: list[dict[str, Any]] = []
    for pose in poses:
        hud = generate_hud_data(
            pose_for_skeleton_render(pose),
            mode="pro",
            hand=resolved_hand,
        )
        hud["frame_index"] = pose.get("frame_index")
        hud["timestamp"] = pose.get("timestamp")
        if pose.get("phase_data") is not None:
            hud["phase"] = pose["phase_data"]
        hud_frames.append(hud)

    dc = detected_club if isinstance(detected_club, dict) else {}
    club_type = dc.get("club_type")
    club_group = dc.get("club_group")

    mid = max(0, len(poses) // 2)
    representative = poses[mid]
    swing_dur = (
        float(poses[-1].get("timestamp") or 0) - float(poses[0].get("timestamp") or 0)
        if len(poses) >= 2
        else 1.2
    )
    if swing_dur < 0.15:
        swing_dur = 1.2
    all_angles = [p.get("angles") or {} for p in poses if p.get("angles")]
    impact_idx = int(phase_keyframes.get("impact", mid))

    prediction = predict_shot(
        representative,
        swing_duration=swing_dur,
        all_frame_angles=all_angles,
        club_type=str(club_type) if club_type else None,
        club_group=str(club_group) if club_group else None,
        hand=resolved_hand,
        hand_confidence=hand_confidence,
        poses=poses,
        impact_pose_idx=impact_idx,
    )

    trajectory_data = compute_wrist_trajectory(poses)

    pose_frames: list[dict[str, Any]] = []
    for p in poses:
        pose_frames.append({k: v for k, v in p.items() if k != "image_base64"})

    dur_ts = float(poses[-1].get("timestamp") or 0) if poses else 0.0
    vm: dict[str, Any] = {
        "fps": float(fps),
        "total_pose_frames": len(poses),
        "duration_s": round(dur_ts, 4),
        "source_frame_count": int(source_frame_count or 0),
    }
    if analysis_frame_count:
        vm["analysis_frame_count"] = int(analysis_frame_count)

    logger.info(
        "[STELLAR_PRO][UI_BUNDLE] hud_frames=%s pose_frames=%s traj=%s",
        len(hud_frames),
        len(pose_frames),
        len(trajectory_data),
    )

    return {
        "skeleton_data": {
            "frames": hud_frames,
            "total_frames": len(hud_frames),
            "joint_space": "analysis_frame",
            "joint_sources": {
                "raw_detection_joints": "pose.raw_detection_joints",
                "analysis_joints": "pose.joints",
                "render_joints": "pose.render_joints->pose.joints",
            },
        },
        "pose_frames": pose_frames,
        "prediction": prediction,
        "trajectory": trajectory_data,
        "video_meta": vm,
        "hand_detection": hand_detection,
    }
