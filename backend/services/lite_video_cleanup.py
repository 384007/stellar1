"""Lite-only: light ffmpeg transcode. Local temp only — no R2 / remote storage."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

from services.video_utils import get_video_rotation

logger = logging.getLogger(__name__)

def _lite_ffmpeg_vf_rotation_prefix(rotation_deg: int) -> str:
    """Match ``video_utils.apply_rotation`` using ffmpeg filters (storage pixels, no container tag).

    Used with ``-noautorotate`` so behavior aligns with OpenCV + ``get_video_rotation`` on the source file.
    """
    r = int(rotation_deg) % 360
    if r == 90:
        return "transpose=1,"  # 90° CW — same sense as cv2.ROTATE_90_CLOCKWISE
    if r == 180:
        return "transpose=1,transpose=1,"
    if r == 270:
        return "transpose=2,"  # 90° CCW — same as cv2.ROTATE_90_COUNTERCLOCKWISE
    return ""


def _lite_fake_analysis_fps() -> int:
    raw = (os.getenv("STELLAR_LITE_FAKE_ANALYSIS_FPS", "240") or "240").strip() or "240"
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        v = 240
    return max(1, min(480, v))


def lite_light_clean_video(source_path: str, work_dir: str) -> dict[str, Any]:
    """
    Scale + H.264 + constant fake analysis fps (default 240 CFR via ffmpeg ``fps=`` — duplicated frames,
    not optical flow). Strips audio. Output stays under ``work_dir`` only.

    ``STELLAR_LITE_FAKE_ANALYSIS_FPS`` overrides the target CFR (clamped 1–480).
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out = str(Path(work_dir) / "lite_clean.mp4")
    target_fps = _lite_fake_analysis_fps()
    src_rot = int(get_video_rotation(source_path))
    rot_prefix = _lite_ffmpeg_vf_rotation_prefix(src_rot)
    if src_rot in (90, 180, 270):
        logger.info(
            "[lite] clean video: burning source rotation=%s° into pixels (-noautorotate + vf)",
            src_rot,
        )
    vf = f"{rot_prefix}scale='min(960,iw)':-2,fps={target_fps}"
    cmd = [
        "ffmpeg",
        "-y",
        "-noautorotate",
        "-i",
        source_path,
        "-vf",
        vf,
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
        logger.info("[lite] clean video -> fake CFR fps=%s (local temp, no R2)", target_fps)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as exc:
        logger.warning("[lite] ffmpeg cleanup failed (%s) — copying source (no fake-%s CFR)", exc, target_fps)
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
