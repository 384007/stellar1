"""Lite-only: light ffmpeg transcode (no true-240 / minterpolate). Local temp files only."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


def lite_light_clean_video(source_path: str, work_dir: str) -> dict[str, Any]:
    """
    Scale + H.264 + fixed nominal fps (30). Strips audio.
    Returns path, fps, total_frames for downstream single-chain timeline.
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out = str(Path(work_dir) / "lite_clean.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-vf",
        "scale='min(960,iw)':-2,fps=30",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        out,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=600,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as exc:
        logger.warning("[lite] ffmpeg cleanup failed (%s) — copying source", exc)
        shutil.copy2(source_path, out)
    cap = cv2.VideoCapture(out)
    if not cap.isOpened():
        raise RuntimeError("lite_clean_video_unreadable")
    vfps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        raise RuntimeError("lite_clean_video_empty")
    if vfps <= 1e-6:
        vfps = 30.0
    return {"path": out, "fps": vfps, "total_frames": total, "duration_s": total / vfps}
