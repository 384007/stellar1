from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from services.internal.prov3_ffmpeg import (
    FFmpegNotFoundError,
    ffmpeg_has_filter,
    ffprobe_video_meta,
    run_ffmpeg,
)

logger = logging.getLogger(__name__)

TARGET_FPS = 240
_MINTERPOLATE_TIMEOUT_S = 3600


def build_analysis_timeline(video_path: str, work_dir: str) -> Dict[str, object]:
    """Build Pro v3 single-authority analysis timeline: true-240 via minterpolate only."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(work_dir) / "analysis_240fps.mp4")
    logger.info("[prov3] true240_required=1")

    meta = {}
    try:
        meta = ffprobe_video_meta(video_path)
    except Exception as exc:
        logger.warning("[prov3] 240fps: ffprobe input failed (%s), proceeding anyway", exc)
    dur = float(meta.get("duration_s") or 0.0)
    w = int(meta.get("width") or 0)
    h = int(meta.get("height") or 0)
    logger.info(
        "[prov3] 240fps input meta: duration_s=%.2f size=%dx%d",
        dur,
        w,
        h,
    )

    if dur <= 0:
        logger.warning(
            "[prov3] 240fps: input duration still unknown after probes — using minterpolate anyway (may be slow)",
        )

    _has_mci = ffmpeg_has_filter("minterpolate")
    if not _has_mci:
        raise RuntimeError(
            "Pro v3 true240_required but ffmpeg filter 'minterpolate' is unavailable. "
            "True240 cannot proceed and no fallback is allowed."
        )

    vf = f"minterpolate=fps={TARGET_FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    logger.info("[prov3] true240_started=1 timeout_s=%s", _MINTERPOLATE_TIMEOUT_S)

    try:
        run_ffmpeg(
            [
                "-i",
                video_path,
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                out_path,
            ],
            label="prov3_240fps",
            timeout_s=_MINTERPOLATE_TIMEOUT_S,
            loglevel="info",
            stats_period_s=30,
        )
    except FFmpegNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"prov3_240fps: ffmpeg failed: {exc}") from exc

    om = ffprobe_video_meta(out_path)
    out_fps = float(om.get("fps") or 0.0)
    if out_fps <= 0:
        raise RuntimeError("prov3_240fps: output fps probe failed")
    if abs(out_fps - TARGET_FPS) > 0.5:
        raise RuntimeError(f"prov3_240fps: output fps mismatch {out_fps:.3f} != {TARGET_FPS}")
    logger.info("[prov3] true240_completed=1 output_fps=%.3f", out_fps)

    return {
        "analysis_video": out_path,
        "analysis_fps": TARGET_FPS,
    }
