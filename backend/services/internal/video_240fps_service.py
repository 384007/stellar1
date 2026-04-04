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


def build_analysis_timeline(video_path: str, work_dir: str) -> Dict[str, object]:
    """Pro v3 **第二步**：在 ``cleanup_video`` 产物上生成 **恒定 240fps** 分析用 MP4（SwingNet / 抽帧用）。

    Uses motion-compensated interpolation when ``minterpolate`` is available; otherwise ``fps=240``
    (frame duplication / sampling — still a valid 240 Hz time base).
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(work_dir) / "analysis_240fps.mp4")

    if ffmpeg_has_filter("minterpolate"):
        vf = f"minterpolate=fps={TARGET_FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        logger.info("[prov3] 240fps pipeline: minterpolate (motion-compensated)")
    else:
        vf = f"fps={TARGET_FPS}"
        logger.warning(
            "[prov3] 240fps pipeline: fps=%s only (minterpolate missing on this ffmpeg build)",
            TARGET_FPS,
        )

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
            timeout_s=1200,
        )
    except FFmpegNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"prov3_240fps: ffmpeg failed: {exc}") from exc

    try:
        om = ffprobe_video_meta(out_path)
        out_fps = float(om.get("fps") or TARGET_FPS)
    except Exception:
        out_fps = float(TARGET_FPS)

    return {
        "analysis_video": out_path,
        "analysis_fps": int(round(out_fps)) if out_fps > 1 else TARGET_FPS,
    }
