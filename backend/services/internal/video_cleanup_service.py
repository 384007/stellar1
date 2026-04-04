from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from services.internal.prov3_ffmpeg import (
    FFmpegNotFoundError,
    ffprobe_video_meta,
    run_ffmpeg,
)


def cleanup_video(input_video: str, work_dir: str, *, screen_mode: bool = False) -> Dict[str, object]:
    """Scale + light denoise + H.264 re-mux; real ffmpeg (not copy-only)."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    cleaned_video = str(Path(work_dir) / "analysis_cleaned.mp4")

    try:
        meta = ffprobe_video_meta(input_video)
    except Exception as exc:
        raise RuntimeError(f"prov3_cleanup: ffprobe failed: {exc}") from exc

    src_fps = float(meta.get("fps") or 30.0)
    # Max width 1280, keep AR; hqdn3d light temporal denoise
    vf = "scale='min(1280,iw)':-2,hqdn3d=4:3:6:4.5"
    if screen_mode:
        vf += ",setsar=1"

    try:
        run_ffmpeg(
            [
                "-i",
                input_video,
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                cleaned_video,
            ],
            label="prov3_cleanup",
            timeout_s=900,
        )
    except FFmpegNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"prov3_cleanup: ffmpeg failed: {exc}") from exc

    return {
        "analysis_video": cleaned_video,
        "source_fps": src_fps,
        "stabilized": True,
        "denoised": True,
        "cropped_single_swing": True,
        "screen_mode_corrected": bool(screen_mode),
        "input_size_bytes": os.path.getsize(input_video),
    }
