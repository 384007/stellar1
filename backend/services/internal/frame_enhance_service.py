from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_analysis_frames(
    analysis_video: str,
    work_dir: str,
    *,
    analysis_fps: int = 240,
    frame_count: int = 96,
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

    n_samples = max(8, min(frame_count, total))
    indices = np.unique(np.linspace(0, total - 1, num=n_samples, dtype=np.int64))

    analysis_frames: list[dict] = []
    enhanced_local_frames: list[dict] = []

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
        if j % 4 == 0:
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
