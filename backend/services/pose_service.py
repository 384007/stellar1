import copy
import cv2
import logging
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple

from services.golf_pose_schema import GOLF_CONNECTIONS, GOLF_KEYPOINTS, calculate_angle, compute_golf_angles
from services.video_utils import get_video_rotation, apply_rotation, read_frame_pose_pipeline

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose


def get_render_joints(pose: dict) -> list:
    """Joints for on-video skeleton drawing.

    Keep render/analysis in the same coordinate space by default:
      render_joints -> joints -> raw_detection_joints.
    """
    rj = pose.get("render_joints")
    if isinstance(rj, list) and len(rj) > 0:
        return rj
    aj = pose.get("joints")
    if isinstance(aj, list) and len(aj) > 0:
        return aj
    raw = pose.get("raw_detection_joints")
    if isinstance(raw, list) and len(raw) > 0:
        return raw
    det = pose.get("detection")
    if isinstance(det, dict):
        j = det.get("joints")
        if isinstance(j, list) and len(j) > 0:
            return j
    return list(pose.get("joints") or [])


def pose_for_skeleton_render(pose: dict) -> dict:
    """Shallow copy for HUD/sketch overlay using render-space == analysis-space by default."""
    det = pose.get("detection") if isinstance(pose.get("detection"), dict) else {}
    rj = get_render_joints(pose)
    ra = pose.get("angles")
    if not isinstance(ra, dict) or not ra:
        ra = det.get("angles")
    if not isinstance(ra, dict) or not ra:
        ra = dict(pose.get("angles") or {})
    out = dict(pose)
    out["joints"] = rj
    out["angles"] = ra
    out["analysis_joints"] = list(pose.get("joints") or [])
    out["render_joints"] = list(rj)
    out["raw_detection_joints"] = list(pose.get("raw_detection_joints") or det.get("joints") or [])
    out["joint_space"] = "analysis_frame"
    return out
mp_drawing = mp.solutions.drawing_utils

DEFAULT_JOINTS = ["left_hip", "right_hip", "left_knee", "right_shoulder", "left_wrist"]
EXTENDED_JOINTS = ["head", "left_elbow", "right_elbow", "left_ankle"]


def extract_pose_from_frame(frame: np.ndarray) -> Optional[dict]:
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = frame.shape[:2]
        joints = []

        for name, idx in GOLF_KEYPOINTS.items():
            lm = results.pose_landmarks.landmark[idx]
            joints.append(
                {
                    "name": name,
                    "x": round(lm.x * w, 2),
                    "y": round(lm.y * h, 2),
                    "z": round(lm.z * w, 2),
                    "visibility": round(lm.visibility, 3),
                    "normalized": {"x": round(lm.x, 4), "y": round(lm.y, 4)},
                }
            )

        connections = []
        for conn in GOLF_CONNECTIONS:
            idx_a = list(GOLF_KEYPOINTS.keys()).index(conn[0])
            idx_b = list(GOLF_KEYPOINTS.keys()).index(conn[1])
            connections.append([idx_a, idx_b])

        angles = compute_golf_angles(joints)

        return {
            "joints": joints,
            "connections": connections,
            "angles": angles,
            "frame_size": {"width": w, "height": h},
            "detection": {
                "joints": copy.deepcopy(joints),
                "angles": dict(angles),
            },
        }


def compute_golf_angles_3d(world_joints: dict) -> dict:
    """Compute golf angles using MediaPipe world_landmarks (3D metric coordinates).

    World landmarks give real 3D positions in meters, making rotation angles
    meaningful even from side-view cameras (where 2D projection collapses depth).
    """
    def get_wj(name: str) -> np.ndarray:
        j = world_joints.get(name, {})
        return np.array([j.get("x", 0.0), j.get("y", 0.0), j.get("z", 0.0)])

    angles = {}

    # Joint angles (3D)
    angles["left_elbow"] = calculate_angle(
        get_wj("left_shoulder"), get_wj("left_elbow"), get_wj("left_wrist")
    )
    angles["right_elbow"] = calculate_angle(
        get_wj("right_shoulder"), get_wj("right_elbow"), get_wj("right_wrist")
    )
    angles["left_knee"] = calculate_angle(
        get_wj("left_hip"), get_wj("left_knee"), get_wj("left_ankle")
    )
    angles["right_knee"] = calculate_angle(
        get_wj("right_hip"), get_wj("right_knee"), get_wj("right_ankle")
    )
    angles["left_shoulder"] = calculate_angle(
        get_wj("left_elbow"), get_wj("left_shoulder"), get_wj("left_hip")
    )
    angles["right_shoulder"] = calculate_angle(
        get_wj("right_elbow"), get_wj("right_shoulder"), get_wj("right_hip")
    )

    # Shoulder rotation: use X-Z plane (horizontal plane in 3D) — not affected by camera angle
    ls = get_wj("left_shoulder")
    rs = get_wj("right_shoulder")
    shoulder_dx = rs[0] - ls[0]
    shoulder_dz = rs[2] - ls[2]
    angles["shoulder_rotation"] = round(
        float(np.degrees(np.arctan2(shoulder_dz, shoulder_dx))), 1
    )

    # Hip rotation: X-Z plane
    lh = get_wj("left_hip")
    rh = get_wj("right_hip")
    hip_dx = rh[0] - lh[0]
    hip_dz = rh[2] - lh[2]
    angles["hip_rotation"] = round(
        float(np.degrees(np.arctan2(hip_dz, hip_dx))), 1
    )

    angles["x_factor"] = round(
        abs(angles["shoulder_rotation"] - angles["hip_rotation"]), 1
    )

    # Spine tilt: 3D vector from mid-hip to mid-shoulder
    mid_hip = (lh + rh) / 2
    mid_shoulder = (ls + rs) / 2
    spine_vec = mid_shoulder - mid_hip
    angles["spine_tilt"] = round(
        float(np.degrees(np.arctan2(
            np.sqrt(spine_vec[0]**2 + spine_vec[2]**2),
            -spine_vec[1]
        ))), 1
    )

    return angles


def _scout_swing_anchors(cap, rotation: int, total_frames: int) -> tuple[int, int]:
    """Pass 1 — fast lightweight scan to locate swing anchor points.

    Uses complexity=1 at 320 px, ~25 frames → finishes in ~2-3 s.
    Returns (top_frame_idx, impact_frame_idx) in the original video's frame
    coordinate space.
    """
    scan_n = min(25, total_frames)
    scan_indices = np.unique(np.linspace(0, total_frames - 1, scan_n, dtype=int))

    wrist_ys: list[float] = []
    frame_ids: list[int] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25,
    ) as pose:
        for idx in scan_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = apply_rotation(frame, rotation)
            h, w = frame.shape[:2]
            if max(h, w) > 320:
                sc = 320 / max(h, w)
                frame = cv2.resize(frame, (int(w * sc), int(h * sc)),
                                   interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue
            rw_y = res.pose_landmarks.landmark[16].y
            lw_y = res.pose_landmarks.landmark[15].y
            wrist_ys.append((rw_y + lw_y) / 2.0)
            frame_ids.append(int(idx))

    if len(wrist_ys) < 5:
        return int(total_frames * 0.42), int(total_frames * 0.65)

    wy = np.array(wrist_ys)
    if len(wy) > 3:
        kernel = np.ones(3) / 3.0
        wy = np.convolve(wy, kernel, mode="same")

    search_top = max(3, int(len(wy) * 0.70))
    top_local = int(np.argmin(wy[:search_top]))
    top_frame = frame_ids[top_local]

    # Impact: use velocity peak (max downward speed) instead of position
    post_wy = wy[top_local:]
    if len(post_wy) > 2:
        velocities = np.diff(post_wy)  # positive = moving down in frame coords
        search_impact = max(1, int(len(velocities) * 0.65))
        impact_local = top_local + int(np.argmax(velocities[:search_impact])) + 1
    else:
        search_impact = max(1, int(len(post_wy) * 0.65))
        impact_local = top_local + int(np.argmax(post_wy[:search_impact]))
    impact_local = min(impact_local, len(frame_ids) - 1)
    impact_frame = frame_ids[impact_local]

    impact_frame = max(impact_frame, top_frame + 2)
    impact_frame = min(impact_frame, total_frames - 3)

    logger.info("Scout pass: top_frame=%d impact_frame=%d (from %d detections)",
                top_frame, impact_frame, len(wrist_ys))
    return top_frame, impact_frame


def _dense_window(center: int, radius: int, total_frames: int) -> np.ndarray:
    """Build a dense integer frame window around an anchor."""
    start = max(0, center - radius)
    end = min(total_frames - 1, center + radius)
    if end < start:
        return np.array([], dtype=int)
    return np.arange(start, end + 1, dtype=int)


def extract_poses_from_video(
    video_path: str,
    max_frames: int = 15,
    include_images: bool = True,
    apply_smoothing: bool = True,
    *,
    frame_index_range: Optional[Tuple[int, int]] = None,
    target_pose_count: int = 180,
) -> Tuple[list[dict], dict]:
    """Dense uniform pose extraction for golf swing analysis.

    Single-pass approach: sample every N-th frame uniformly across the video,
    using static_image_mode=True (correct for non-sequential frame reads).
    This avoids the scout-pass misalignment bug and gives every swing phase
    equal representation.

    Typical: 14s video @ 30fps = 420 frames → sample every 3rd frame → ~140 poses
    (capped at 180 for very long clips).

    **Stellar Pro / high-fps:** pass ``frame_index_range=(lo, hi)`` to sample only a
    time window around the swing (e.g. impact ± a few seconds). Otherwise 180 poses
    are spread over the *entire* clip and phase windows land seconds apart — wrong
    keyframe strips on long uploads.

    With apply_smoothing, ``detection`` holds pre-smooth joints/angles for kinematic
    phase detection; joint/angle fields used for UI use the smoothed sequence.
    """
    import base64

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    rotation = get_video_rotation(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0

    # Dense uniform sampling: every 3–5 frames for typical golf length; cap 180 for runtime bound.
    if total_frames <= 0:
        cap.release()
        from services.pose_refine_service import empty_pose_quality_bundle

        return [], empty_pose_quality_bundle("NO_VIDEO_FRAMES")

    if frame_index_range is not None:
        lo, hi = int(frame_index_range[0]), int(frame_index_range[1])
        lo = max(0, min(lo, total_frames - 1))
        hi = max(0, min(hi, total_frames - 1))
        if hi < lo:
            lo, hi = 0, total_frames - 1
        span = hi - lo + 1
        cap_target = max(8, min(int(target_pose_count), span))
        n_lin = max(1, min(cap_target, span))
        sample_indices = np.unique(np.linspace(lo, hi, num=n_lin, dtype=int))
        if len(sample_indices) < 8 and span >= 8:
            sample_indices = np.unique(np.linspace(lo, hi, num=min(120, span), dtype=int))
        logger.info(
            "Swing-window sampling: [%d,%d] (%d frames) → %d sample indices (%.2fs–%.2fs @ %.0ffps)",
            lo,
            hi,
            span,
            len(sample_indices),
            lo / max(fps, 1e-6),
            hi / max(fps, 1e-6),
            fps,
        )
    else:
        step = max(2, min(5, total_frames // max(max_frames, 40)))
        sample_indices = np.arange(0, total_frames, step, dtype=int)

        if len(sample_indices) > 180:
            sample_indices = np.unique(np.linspace(0, total_frames - 1, 180, dtype=int))

        logger.info(
            "Dense uniform sampling: step=%d → %d frames from %d total (%.1fs @ %.0ffps)",
            step, len(sample_indices), total_frames, duration, fps,
        )

    # ── Single-pass extraction with static_image_mode=True ──
    # static_image_mode=True is REQUIRED when frames are not sequential.
    # Using False with cap.set() seek causes MediaPipe tracking to break.
    poses = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.3,
    ) as pose:
        for idx in sample_indices:
            frame = read_frame_pose_pipeline(cap, int(idx), rotation)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            max_dim = 480
            if max(h, w) > max_dim:
                sc = max_dim / max(h, w)
                frame = cv2.resize(
                    frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA
                )
                h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            joints = []
            world_joints = {}
            for name, mp_idx in GOLF_KEYPOINTS.items():
                lm = results.pose_landmarks.landmark[mp_idx]
                joints.append(
                    {
                        "name": name,
                        "x": round(lm.x * w, 2),
                        "y": round(lm.y * h, 2),
                        "z": round(lm.z * w, 2),
                        "visibility": round(lm.visibility, 3),
                        "normalized": {"x": round(lm.x, 4), "y": round(lm.y, 4)},
                    }
                )
                # Extract world landmarks (3D metric coordinates) for better angle computation
                if results.pose_world_landmarks:
                    wlm = results.pose_world_landmarks.landmark[mp_idx]
                    world_joints[name] = {
                        "x": round(wlm.x, 5),
                        "y": round(wlm.y, 5),
                        "z": round(wlm.z, 5),
                        "visibility": round(wlm.visibility, 3),
                    }

            connections = []
            for conn in GOLF_CONNECTIONS:
                idx_a = list(GOLF_KEYPOINTS.keys()).index(conn[0])
                idx_b = list(GOLF_KEYPOINTS.keys()).index(conn[1])
                connections.append([idx_a, idx_b])

            # Use world landmarks for angle computation when available (3D is more accurate)
            if world_joints and len(world_joints) >= 10:
                angles = compute_golf_angles_3d(world_joints)
            else:
                angles = compute_golf_angles(joints)

            image_b64 = ""
            if include_images:
                thumb_w = 360
                thumb_scale = thumb_w / w
                thumb = cv2.resize(
                    frame,
                    (thumb_w, int(h * thumb_scale)),
                    interpolation=cv2.INTER_AREA,
                )
                _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
                image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            entry: dict = {
                "frame_index": int(idx),
                "timestamp": round(float(idx) / fps, 3),
                "joints": joints,
                "connections": connections,
                "angles": angles,
                "frame_size": {"width": w, "height": h},
                "world_landmarks": world_joints if world_joints else None,
                "segmentation_mask": None,
                "phase_data": None,
            }
            if image_b64:
                entry["image_base64"] = image_b64

            # Raw detection snapshot before any temporal smoothing / refinement (used for HUD).
            entry["detection"] = {
                "joints": copy.deepcopy(joints),
                "angles": dict(angles),
            }
            entry["raw_detection_joints"] = copy.deepcopy(joints)
            entry["raw_detection_angles"] = dict(angles)
            entry["analysis_joints"] = copy.deepcopy(joints)
            entry["render_joints"] = copy.deepcopy(joints)

            poses.append(entry)

    cap.release()

    if apply_smoothing and len(poses) >= 3:
        from services.swing_flow_utils import smooth_pose_sequence

        poses = smooth_pose_sequence(poses, alpha=0.35)

    if len(poses) >= 1:
        from services.pose_refine_service import refine_pose_sequence_pipeline

        pose_bundle = refine_pose_sequence_pipeline(poses, float(fps))
    else:
        from services.pose_refine_service import empty_pose_quality_bundle

        pose_bundle = empty_pose_quality_bundle("NO_POSES_EXTRACTED")

    # Keep all downstream consumers (HUD / keyframe snapshots / analysis) on one pose space.
    for p in poses:
        if not isinstance(p.get("analysis_joints"), list) or not p.get("analysis_joints"):
            p["analysis_joints"] = copy.deepcopy(p.get("joints") or [])
        p["render_joints"] = copy.deepcopy(p.get("joints") or p.get("analysis_joints") or [])
        if not isinstance(p.get("raw_detection_joints"), list):
            det = p.get("detection") if isinstance(p.get("detection"), dict) else {}
            p["raw_detection_joints"] = copy.deepcopy(det.get("joints") or p.get("joints") or [])

    logger.info(
        f"Extracted {len(poses)} poses from {total_frames} frames "
        f"({duration:.1f}s @ {fps:.0f}fps)"
    )
    return poses, pose_bundle
