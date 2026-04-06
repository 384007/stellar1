from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_analysis_frames(
    analysis_video: str,
    work_dir: str,
    *,
    analysis_fps: int = 240,
    frame_count: int = 240,
) -> Dict[str, List[dict]]:
    """Sample real frames from the analysis video; build timeline indices on ``analysis_fps`` grid."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(analysis_video)
    if not cap.isOpened():
        logger.warning("[prov3] frame_enhance: cannot open %s — empty frame list", analysis_video)
        return {"analysis_frames": [], "enhanced_local_frames": []}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vfps = float(cap.get(cv2.CAP_PROP_FPS) or analysis_fps)
    if vfps <= 1e-6:
        vfps = float(analysis_fps)

    if total <= 0:
        cap.release()
        return {"analysis_frames": [], "enhanced_local_frames": []}

    n_samples = max(24, min(frame_count, total))
    base = np.linspace(0, total - 1, num=n_samples, dtype=np.int64)

    mid_l = int(round((total - 1) * 0.15))
    mid_r = int(round((total - 1) * 0.85))
    if mid_r <= mid_l:
        mid_l, mid_r = 0, total - 1
    mid_count = max(48, min(total, 360))
    mid_band = np.linspace(mid_l, mid_r, num=mid_count, dtype=np.int64)

    core_l = int(round((total - 1) * 0.35))
    core_r = int(round((total - 1) * 0.75))
    if core_r <= core_l:
        core_l, core_r = mid_l, mid_r
    core_count = max(64, min(total, 420))
    core_band = np.linspace(core_l, core_r, num=core_count, dtype=np.int64)

    indices = np.unique(np.concatenate([base, mid_band, core_band], axis=0))

    analysis_frames: list[dict] = []
    enhanced_local_frames: list[dict] = []

    enhanced_idx: set[int] = set()
    for idx in indices.tolist():
        for d in (-2, -1, 0, 1, 2):
            x = int(idx) + d
            if 0 <= x < total:
                enhanced_idx.add(x)

    for j, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(idx)))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        fi = int(idx)
        time_ms = int(round(fi * 1000.0 / vfps))
        analysis_frames.append(
            {
                "frame_index": fi,
                "time_ms": time_ms,
            }
        )
        if int(fi) in enhanced_idx or j % 2 == 0:
            enhanced_local_frames.append({"frame_index": fi, "enhanced": True})
            # Optional mild sharpen for B-layer hints (no persistence — metadata only)
            try:
                blur = cv2.GaussianBlur(frame_bgr, (0, 0), 2.0)
                sharp = cv2.addWeighted(frame_bgr, 1.35, blur, -0.35, 0)
                _ = sharp  # future: could write thumbs; gate uses indices only
            except cv2.error:
                pass

    cap.release()

    if not analysis_frames:
        logger.warning("[prov3] frame_enhance: no frames decoded from %s", analysis_video)

    return {
        "analysis_frames": analysis_frames,
        "enhanced_local_frames": enhanced_local_frames,
    }


_EVENT_FILE_NAMES: dict[str, str] = {
    "Address": "address.jpg",
    "Toe-up": "toe_up.jpg",
    "Mid-backswing": "mid_backswing.jpg",
    "Top": "top.jpg",
    "Mid-downswing": "mid_downswing.jpg",
    "Impact": "impact.jpg",
    "Mid-follow-through": "mid_follow_through.jpg",
    "Finish": "finish.jpg",
}


def persist_final_keyframe_images(
    analysis_video: str,
    keyframes: List[dict[str, Any]],
    output_dir: str,
) -> list[dict[str, Any]]:
    """Persist final keyframe JPEGs sampled from the analysis timeline video."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(analysis_video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot_open_analysis_video:{analysis_video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"analysis_video_has_no_frames:{analysis_video}")

    saved: list[dict[str, Any]] = []
    try:
        for row in keyframes:
            event_name = str(row.get("event_name") or "")
            file_name = _EVENT_FILE_NAMES.get(event_name)
            if not file_name:
                continue
            frame_index = int(row.get("frame_index") or 0)
            frame_index = max(0, min(frame_index, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"decode_failed:event={event_name}:frame_index={frame_index}")
            out_path = str(Path(output_dir) / file_name)
            if not cv2.imwrite(out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"write_failed:{out_path}")
            saved.append(
                {
                    "event_name": event_name,
                    "frame_index": frame_index,
                    "file_name": file_name,
                    "file_path": out_path,
                    "keyframe_image_source": "analysis_video",
                }
            )
    finally:
        cap.release()

    if len(saved) != 8:
        raise RuntimeError(f"persist_keyframes_incomplete: expected=8 got={len(saved)}")
    return saved
