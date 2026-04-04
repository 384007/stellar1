"""Pro v2 — FFmpeg: 240fps analysis clip, frontend playback, optional impact window."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from services.ffmpeg_preprocess_service import (
    build_frontend_playback_from_analysis,
    build_full_240fps_playback,
    ffprobe_fps,
    probe_video,
    safe_duration_s,
    trim_video_segment_h264,
)

logger = logging.getLogger(__name__)


@dataclass
class ProV2FFmpegOutputs:
    analysis_240_path: str
    playback_path: str
    impact_window_path: str | None
    fps: float
    duration_s: float


def run_pro_v2_ffmpeg_preprocess(
    input_path: str,
    work_dir: str,
    *,
    rough_impact_time_s: float | None = None,
    impact_window_pre_s: float = 0.10,
    impact_window_duration_s: float = 0.26,
    playback_crf: int = 32,
    analysis_vf_prefix: str | None = None,
) -> ProV2FFmpegOutputs:
    """Original video → 240fps analysis MP4 → browser playback MP4; optional impact trim on 240fps timeline."""
    from pathlib import Path

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    analysis_path = str(work / "pro_v2_analysis_240.mp4")
    playback_path = str(work / "pro_v2_playback.mp4")

    build_full_240fps_playback(
        input_path,
        analysis_path,
        fast=True,
        vf_prefix=analysis_vf_prefix,
    )
    meta = probe_video(analysis_path)
    fps = float(ffprobe_fps(meta))
    duration_s = float(safe_duration_s(meta))

    build_frontend_playback_from_analysis(
        analysis_path,
        playback_path,
        crf=playback_crf,
    )

    impact_window_path: str | None = None
    if rough_impact_time_s is not None:
        imp = max(0.0, min(float(rough_impact_time_s), max(0.0, duration_s - 0.05)))
        start_s = max(0.0, imp - impact_window_pre_s)
        iw = str(work / "pro_v2_impact_window.mp4")
        trim_video_segment_h264(
            analysis_path,
            iw,
            start_s=start_s,
            duration_s=impact_window_duration_s,
        )
        impact_window_path = iw

    logger.info(
        "[PRO_V2][FFMPEG] analysis=%s playback=%s fps=%.3f dur=%.3fs impact_clip=%s vf_prefix=%s",
        analysis_path,
        playback_path,
        fps,
        duration_s,
        impact_window_path,
        repr((analysis_vf_prefix or "")[:100]),
    )

    return ProV2FFmpegOutputs(
        analysis_240_path=analysis_path,
        playback_path=playback_path,
        impact_window_path=impact_window_path,
        fps=fps,
        duration_s=duration_s,
    )
