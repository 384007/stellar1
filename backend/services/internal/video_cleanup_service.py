from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from services.internal.prov3_ffmpeg import (
    FFmpegNotFoundError,
    ffprobe_video_meta,
    run_ffmpeg,
)


def prov3_cleanup_max_dims() -> Tuple[int, int]:
    """Max analysis box for Pro v3 cleanup / true240 (env overridable)."""
    try:
        max_w = max(320, int(os.getenv("STELLAR_PROV3_CLEANUP_MAX_W") or "1280"))
    except ValueError:
        max_w = 1280
    try:
        max_h = max(240, int(os.getenv("STELLAR_PROV3_CLEANUP_MAX_H") or "720"))
    except ValueError:
        max_h = 720
    return max_w, max_h


def prov3_cleanup_vf(*, screen_mode: bool = False) -> str:
    """Video filter chain shared by standalone cleanup and merged cleanup+true240 (no minterpolate)."""
    max_w, max_h = prov3_cleanup_max_dims()
    vf = (
        f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease,"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2,"
        f"hqdn3d=4:3:6:4.5"
    )
    if screen_mode:
        vf += ",setsar=1"
    return vf


def cleanup_video(input_video: str, work_dir: str, *, screen_mode: bool = False) -> Dict[str, object]:
    """Pro v3 **第一步**：缩放 + 轻去噪 + H.264（非 copy）。

    输出仍为**源时间轴近似帧率**的清洗片段；**恒定 240fps 分析轨**由后续的
    ``build_analysis_timeline`` 生成。``screen_mode=True`` 时追加 ``setsar=1``，便于拍屏素材。

    产品主路径已改为 ``run_prov3_cleanup_and_true240`` 单次编码；本函数保留用于需要
    **仅清洗**的中间产物（测试、调试或未来分步 API）。

    分辨率默认限制在 ``STELLAR_PROV3_CLEANUP_MAX_W`` × ``STELLAR_PROV3_CLEANUP_MAX_H``（默认 1280×720）
    边界框内 ``force_original_aspect_ratio=decrease``，再 ``trunc(iw/2)*2`` 保证宽高为偶数（libx264+yuv420p），
    竖屏 1080p 约 **404×720**，显著减轻后续 minterpolate CPU。
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    cleaned_video = str(Path(work_dir) / "analysis_cleaned.mp4")

    try:
        meta = ffprobe_video_meta(input_video)
    except Exception as exc:
        raise RuntimeError(f"prov3_cleanup: ffprobe failed: {exc}") from exc

    src_fps = float(meta.get("fps") or 30.0)
    vf = prov3_cleanup_vf(screen_mode=screen_mode)

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
