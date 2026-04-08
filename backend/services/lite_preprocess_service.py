"""Lite preprocess: fake-240 clean video + Prov3-style ``generate_analysis_frames`` (no R2, no prov3 imports)."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import cv2

from services.internal.frame_enhance_service import generate_analysis_frames
from services.lite_preview_sample import lite_sample_preview_bgr
from services.lite_video_cleanup import lite_light_clean_video
from services.pose_backend_service import extract_pose_stream
from services.provider_registry import role_log

logger = logging.getLogger(__name__)
_LOG = "[lite_pre]"

# Align with Pro v3 preprocess pose cap (``STELLAR_PLUS_POSE_MAX_FRAMES``).
_LITE_POSE_FRAMES = int(os.getenv("STELLAR_LITE_POSE_PREVIEW_FRAMES", os.getenv("STELLAR_PLUS_POSE_MAX_FRAMES", "45")))


def _probe_source_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 1e-6 else 30.0


def run_lite_preprocess(source_video_path: str, work_dir: str) -> dict[str, Any]:
    """
    1. ``lite_light_clean_video`` — scale + H.264 + **fake CFR** (default 240 via ``STELLAR_LITE_FAKE_ANALYSIS_FPS``).
    2. ``generate_analysis_frames`` — same frame index lattice as Pro v3 (internal helper, not prov3 package).
    3. ``extract_pose_stream`` on analysis clip.
    4. Preview BGR samples for club vision.
    """
    role_log(f"[ROLE=LITE_PIPELINE] preprocess_start src={os.path.basename(source_video_path)!r}")
    source_fps = _probe_source_fps(source_video_path)
    clean = lite_light_clean_video(source_video_path, work_dir)
    analysis_path = str(clean["path"])
    analysis_fps = float(clean["fps"])
    total_frames = int(clean["total_frames"])
    duration_s = float(clean.get("duration_s") or (total_frames / max(analysis_fps, 1e-6)))
    role_log(
        f"[ROLE=LITE_PIPELINE] after_ffmpeg_clean fps={analysis_fps:.1f} total_frames={total_frames} "
        f"duration_s={duration_s:.2f}"
    )

    analysis_id = f"lite_{uuid.uuid4().hex[:12]}"
    local_dir = str(Path(work_dir) / analysis_id)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    afps_int = int(round(analysis_fps))
    frames_bundle = generate_analysis_frames(
        analysis_path,
        local_dir,
        analysis_fps=afps_int,
    )
    analysis_frames = list(frames_bundle.get("analysis_frames") or [])
    enhanced_local_frames = list(frames_bundle.get("enhanced_local_frames") or [])
    role_log(
        f"[ROLE=LITE_PIPELINE] after_generate_analysis_frames "
        f"analysis_points={len(analysis_frames)} enhanced={len(enhanced_local_frames)}"
    )

    role_log("[ROLE=LITE_PIPELINE] pose_extract_start (dense sampling up to ~180 poses on analysis clip)")
    pose_bundle = extract_pose_stream(analysis_path, _LITE_POSE_FRAMES)
    poses = list(pose_bundle.get("poses") or [])

    preview_bgr = lite_sample_preview_bgr(analysis_path, (0.25, 0.4, 0.6))

    logger.info(
        "%s preprocess done analysis_fps=%s frames=%d poses=%d analysis_frame_points=%d enhanced=%d",
        _LOG,
        analysis_fps,
        total_frames,
        len(poses),
        len(analysis_frames),
        len(enhanced_local_frames),
    )
    role_log(
        f"[ROLE=LITE_PIPELINE] preprocess_done poses={len(poses)} analysis_id={analysis_id} "
        f"next=swingnet_or_heuristic_ab"
    )

    return {
        "analysis_id": analysis_id,
        "analysis_video_path": analysis_path,
        "analysis_fps": analysis_fps,
        "source_fps": source_fps,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "analysis_frames": analysis_frames,
        "enhanced_local_frames": enhanced_local_frames,
        "poses": poses,
        "preview_bgr": preview_bgr,
        "screen_mode_corrected": False,
    }
