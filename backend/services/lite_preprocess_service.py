"""Lite preprocess: local temp only, fake-CFR analysis timeline (no R2, no prov3)."""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2

from services.lite_preview_sample import lite_sample_preview_bgr
from services.lite_timeline_motion import lite_build_uniform_timeline
from services.lite_video_cleanup import lite_light_clean_video
from services.pose_backend_service import extract_pose_stream

logger = logging.getLogger(__name__)
_LOG = "[lite_pre]"

_LITE_POSE_FRAMES = int(os.getenv("STELLAR_LITE_POSE_PREVIEW_FRAMES", "40"))


def _probe_source_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 1e-6 else 30.0


def run_lite_preprocess(source_video_path: str, work_dir: str) -> dict[str, Any]:
    """
    Build Lite analysis asset: scaled H.264 + constant fake analysis fps (default 240),
    uniform timeline samples, pose stream, preview BGR strips for downstream A/B.

    Returns a bundle consumed only by Lite orchestrator / A / B (not sent to clients).
    """
    source_fps = _probe_source_fps(source_video_path)
    clean = lite_light_clean_video(source_video_path, work_dir)
    analysis_path = str(clean["path"])
    analysis_fps = float(clean["fps"])
    total_frames = int(clean["total_frames"])
    duration_s = float(clean.get("duration_s") or (total_frames / max(analysis_fps, 1e-6)))

    timeline = lite_build_uniform_timeline(total_frames, analysis_fps)
    analysis_frames = [
        {"frame_index": int(t["frame_index"]), "time_ms": int(t["time_ms"])} for t in timeline
    ]

    pose_bundle = extract_pose_stream(analysis_path, _LITE_POSE_FRAMES)
    poses = list(pose_bundle.get("poses") or [])

    preview_bgr = lite_sample_preview_bgr(analysis_path, (0.25, 0.4, 0.6))

    preprocess_meta = {
        "source_fps": round(source_fps, 4),
        "analysis_fps": round(analysis_fps, 4),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 4),
        "timeline_len": len(timeline),
        "pose_count": len(poses),
        "work_dir": work_dir,
    }
    logger.info(
        "%s preprocess done analysis_fps=%s frames=%d poses=%d timeline=%d",
        _LOG,
        analysis_fps,
        total_frames,
        len(poses),
        len(timeline),
    )

    return {
        "analysis_video_path": analysis_path,
        "analysis_fps": analysis_fps,
        "source_fps": source_fps,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "preprocess_meta": preprocess_meta,
        "analysis_frames": analysis_frames,
        "timeline": timeline,
        "poses": poses,
        "preview_bgr": preview_bgr,
        "enhanced_local_frames": [],
    }
