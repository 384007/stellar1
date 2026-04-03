"""Optional RTMPose / MMPose backend: MMPoseInferencer + COCO-17 → stellar pose schema (MediaPipe-compatible fields)."""

from __future__ import annotations

import base64
import copy
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.golf_pose_schema import GOLF_CONNECTIONS, GOLF_KEYPOINTS, compute_golf_angles
from services.provider_registry import role_log
from services.provider_schema import provider_result
from services.video_utils import get_video_rotation, read_frame_pose_pipeline

# COCO 17 body keypoint order (OpenMMLab / COCO).
_GOLF_NAME_TO_COCO17 = {
    "head": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

_INF_LOCK = threading.Lock()
_INF_CACHE: dict[tuple[str, str, str, str], Any] = {}

# Shipped with pip ``mmpose`` (``.mim/configs``). COCO-17 body — matches ``coco17_to_stellar_joints``.
_DEFAULT_RTMPOSE_CONFIG_REL = Path(".mim/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py")
# Official weight for that config (``rtmpose_coco.yml``).
_DEFAULT_RTMPOSE_CHECKPOINT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"
)


def _default_rtmpose_config_path() -> str | None:
    try:
        import mmpose
    except Exception:
        return None
    root = Path(mmpose.__file__).resolve().parent
    p = root / _DEFAULT_RTMPOSE_CONFIG_REL
    return str(p) if p.is_file() else None


def _resolved_rtmpose_config_and_checkpoint() -> tuple[str, str]:
    """Env overrides; else bundled config path + OpenMMLab checkpoint URL."""
    cfg = (os.getenv("STELLAR_RTMPOSE_CONFIG") or "").strip()
    ckpt = (os.getenv("STELLAR_RTMPOSE_CHECKPOINT") or "").strip()
    if not cfg:
        cfg = _default_rtmpose_config_path() or ""
    if not ckpt:
        ckpt = _DEFAULT_RTMPOSE_CHECKPOINT
    return cfg, ckpt


def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.float64)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _squeeze_kpts(k: np.ndarray) -> np.ndarray:
    arr = np.asarray(k, dtype=np.float64)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def coco17_to_stellar_joints(
    keypoints_xy: np.ndarray,
    keypoint_scores: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple[list[dict], list, dict]:
    """Map COCO-17 (pixel xy) + scores to product joint list + connections + angles."""
    k = _squeeze_kpts(keypoints_xy)
    s = _to_numpy(keypoint_scores).reshape(-1)
    if k.shape[0] < 17 or k.shape[1] < 2:
        raise ValueError("coco17_to_stellar_joints expects keypoints (17,2+)")
    if len(s) < 17:
        s = np.pad(s, (0, 17 - len(s)), constant_values=0.5)
    w = max(int(frame_w), 1)
    h = max(int(frame_h), 1)
    names_in_order = list(GOLF_KEYPOINTS.keys())
    joints: list[dict] = []
    for name in names_in_order:
        ci = _GOLF_NAME_TO_COCO17[name]
        x, y = float(k[ci, 0]), float(k[ci, 1])
        vis = float(min(1.0, max(0.0, float(s[ci]) if ci < len(s) else 0.5)))
        joints.append(
            {
                "name": name,
                "x": round(x, 2),
                "y": round(y, 2),
                "z": 0.0,
                "visibility": round(vis, 3),
                "normalized": {"x": round(x / w, 4), "y": round(y / h, 4)},
            },
        )
    connections: list = []
    keys_list = list(GOLF_KEYPOINTS.keys())
    for conn in GOLF_CONNECTIONS:
        idx_a = keys_list.index(conn[0])
        idx_b = keys_list.index(conn[1])
        connections.append([idx_a, idx_b])
    angles = compute_golf_angles(joints)
    return joints, connections, angles


def _slice_coco17_body(keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    k = _squeeze_kpts(keypoints)
    s = _to_numpy(scores).reshape(-1)
    if k.shape[0] < 17:
        return None
    if k.shape[0] > 17:
        role_log(f"[ROLE=POSE_BACKEND] rtmpose n_kpt={k.shape[0]} using first 17 as COCO body")
    k17 = k[:17, :2]
    s17 = s[:17] if len(s) >= 17 else np.pad(s, (0, 17 - len(s)), constant_values=0.5)[:17]
    return k17, s17


def _pick_best_instance(prediction_root: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve MMPoseInferencer ``predictions`` nesting to one (kpts, scores)."""
    if prediction_root is None:
        return None
    if isinstance(prediction_root, list) and len(prediction_root) == 0:
        return None
    first = prediction_root[0] if isinstance(prediction_root, list) else prediction_root
    instances: list
    if isinstance(first, list):
        instances = first
    elif isinstance(first, dict):
        instances = [first]
    else:
        return None
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_mean = -1.0
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        k = inst.get("keypoints")
        if k is None:
            continue
        k = _squeeze_kpts(_to_numpy(k))
        sc = inst.get("keypoint_scores")
        s = _to_numpy(sc).reshape(-1) if sc is not None else np.ones(max(k.shape[0], 1), dtype=np.float64)
        if k.ndim < 2 or k.shape[1] < 2:
            continue
        mean_s = float(np.mean(s)) if len(s) else 0.0
        if mean_s > best_mean:
            best_mean = mean_s
            best = (k, s)
    return best


def _parse_inferencer_batch(out: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    preds = out.get("predictions")
    if preds is None and isinstance(out.get("pose2d"), dict):
        preds = out["pose2d"].get("predictions")
    picked = _pick_best_instance(preds)
    if picked is None:
        return None
    return _slice_coco17_body(picked[0], picked[1])


def _uniform_sample_indices(total_frames: int, max_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    step = max(2, min(5, total_frames // max(max_frames, 40)))
    sample_indices = np.arange(0, total_frames, step, dtype=int)
    if len(sample_indices) > 180:
        sample_indices = np.unique(np.linspace(0, total_frames - 1, 180, dtype=int))
    return sample_indices


def _get_cached_inferencer(cfg: str, ckpt: str, det: str, device: str | None):
    key = (cfg, ckpt, det, device or "")
    with _INF_LOCK:
        if key not in _INF_CACHE:
            from mmpose.apis import MMPoseInferencer

            kwargs: dict[str, Any] = {
                "pose2d": cfg,
                "pose2d_weights": ckpt,
                "det_model": det,
                "show_progress": False,
            }
            if device:
                kwargs["device"] = device
            _INF_CACHE[key] = MMPoseInferencer(**kwargs)
        return _INF_CACHE[key]


def run(video_path: str, max_frames: int = 45) -> dict:
    cfg, ckpt = _resolved_rtmpose_config_and_checkpoint()
    try:
        import mmpose  # noqa: F401
    except Exception as exc:
        role_log(f"[ROLE=POSE_BACKEND] rtmpose status=dependency_missing err={exc}")
        return provider_result(
            role="pose",
            provider_name="rtmpose",
            status="dependency_missing",
            error_reason="mmpose_import_failed",
            frame_count=0,
            payload={"poses": [], "pose_quality_bundle": {}},
        )
    if not cfg or not ckpt:
        role_log("[ROLE=POSE_BACKEND] rtmpose status=model_config_missing")
        return provider_result(
            role="pose",
            provider_name="rtmpose",
            status="model_config_missing",
            error_reason="config_or_checkpoint_unset",
            frame_count=0,
            payload={"poses": [], "pose_quality_bundle": {}},
        )

    device_raw = (os.getenv("STELLAR_RTMPOSE_DEVICE") or "").strip()
    device = device_raw if device_raw else None
    det = (os.getenv("STELLAR_RTMPOSE_DET_MODEL") or "whole_image").strip()

    try:
        inferencer = _get_cached_inferencer(cfg, ckpt, det, device)
    except Exception as exc:
        role_log(f"[ROLE=POSE_BACKEND] rtmpose status=init_failed err={exc}")
        return provider_result(
            role="pose",
            provider_name="rtmpose",
            status="init_failed",
            error_reason=str(exc),
            frame_count=0,
            payload={"poses": [], "pose_quality_bundle": {}},
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return provider_result(
            role="pose",
            provider_name="rtmpose",
            status="inference_failed",
            error_reason="video_open_failed",
            frame_count=0,
            payload={"poses": [], "pose_quality_bundle": {}},
        )

    rotation = get_video_rotation(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0:
        fps = 30.0

    sample_indices = _uniform_sample_indices(total_frames, max_frames)
    poses: list[dict] = []

    for idx in sample_indices:
        frame = read_frame_pose_pipeline(cap, int(idx), rotation)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        max_dim = 480
        if max(h, w) > max_dim:
            sc = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        try:
            gen = inferencer([frame], return_vis=False, batch_size=1)
            out = next(gen)
        except StopIteration:
            continue
        except Exception as exc:
            role_log(f"[ROLE=POSE_BACKEND] rtmpose frame_infer_failed idx={idx} err={exc}")
            continue

        if not isinstance(out, dict):
            continue
        parsed = _parse_inferencer_batch(out)
        if parsed is None:
            continue
        k17, s17 = parsed
        try:
            joints, connections, angles = coco17_to_stellar_joints(k17, s17, w, h)
        except Exception:
            continue

        thumb_w = 360
        thumb_scale = thumb_w / w
        thumb = cv2.resize(frame, (thumb_w, int(h * thumb_scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        entry: dict = {
            "frame_index": int(idx),
            "timestamp": round(float(idx) / fps, 3),
            "joints": joints,
            "connections": connections,
            "angles": angles,
            "frame_size": {"width": w, "height": h},
            "world_landmarks": None,
            "segmentation_mask": None,
            "phase_data": None,
            "image_base64": image_b64,
            "detection": {"joints": copy.deepcopy(joints), "angles": dict(angles)},
            "raw_detection_joints": copy.deepcopy(joints),
            "raw_detection_angles": dict(angles),
            "analysis_joints": copy.deepcopy(joints),
            "render_joints": copy.deepcopy(joints),
        }
        poses.append(entry)

    cap.release()

    if len(poses) >= 3:
        from services.swing_flow_utils import smooth_pose_sequence

        poses = smooth_pose_sequence(poses, alpha=0.35)

    if len(poses) >= 1:
        from services.pose_refine_service import refine_pose_sequence_pipeline

        pose_bundle = refine_pose_sequence_pipeline(poses, float(fps))
    else:
        from services.pose_refine_service import empty_pose_quality_bundle

        pose_bundle = empty_pose_quality_bundle("NO_POSES_EXTRACTED")

    for p in poses:
        if not isinstance(p.get("analysis_joints"), list) or not p.get("analysis_joints"):
            p["analysis_joints"] = copy.deepcopy(p.get("joints") or [])
        p["render_joints"] = copy.deepcopy(p.get("joints") or p.get("analysis_joints") or [])
        if not isinstance(p.get("raw_detection_joints"), list):
            det = p.get("detection") if isinstance(p.get("detection"), dict) else {}
            p["raw_detection_joints"] = copy.deepcopy(det.get("joints") or p.get("joints") or [])

    avg_visible = 0.0
    if poses:
        vis = []
        for p in poses:
            joints = p.get("joints", [])
            vis.append(sum(1 for j in joints if float(j.get("visibility", 0.0)) >= 0.3))
        avg_visible = float(sum(vis) / max(len(vis), 1))

    if not poses:
        role_log(f"[ROLE=POSE_BACKEND] rtmpose status=no_poses clip_len={max_frames}")
        return provider_result(
            role="pose",
            provider_name="rtmpose",
            status="inference_failed",
            error_reason="no_poses_detected",
            frame_count=0,
            payload={"poses": [], "pose_quality_bundle": pose_bundle},
        )

    role_log(
        f"[ROLE=POSE_BACKEND] rtmpose status=ok frames={len(poses)} avg_visible_joints={avg_visible:.2f}",
    )
    return provider_result(
        role="pose",
        provider_name="rtmpose",
        provider_version="mmpose_inferencer",
        backend_profile="high_precision",
        status="ok",
        frame_count=len(poses),
        timestamps=[float(p.get("timestamp", 0.0)) for p in poses],
        frame_indices=[int(p.get("frame_index", i)) for i, p in enumerate(poses)],
        confidence_summary={"avg_visible_joints": avg_visible},
        payload={"poses": poses, "pose_quality_bundle": pose_bundle},
    )
