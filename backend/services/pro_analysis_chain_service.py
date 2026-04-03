"""Pro analysis input preparation: FFmpeg 240fps artifacts and stable paths (FFMPEG_PREP → PRO_CHAIN)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.ffmpeg_preprocess_service import (
    FFmpegBinaryMissingError,
    FFmpegProcessError,
    build_frontend_playback_from_analysis,
    build_full_240fps_playback,
    probe_video,
    suggest_impact_window,
    trim_video_segment_h264,
    verify_ffmpeg,
)
from services.stellar_pro_role_log import (
    ROLE_FFMPEG_PREP,
    ROLE_PRO_CHAIN,
    log_stage_done,
    log_stage_failed,
    log_stage_start,
)


@dataclass
class ProAnalysisChainSettings:
    """Tuning for staged Pro rollout (kept small; env can extend later)."""

    analysis_240_fast: bool = True
    frontend_240_fast: bool = True
    impact_window_pre_s: float = 0.10
    impact_window_duration_s: float = 0.22


@dataclass
class ProAnalysisArtifacts:
    work_dir: str
    input_path: str
    analysis_video_path: str
    frontend_video_path: str
    impact_window_video_path: str | None = None
    analysis_video_meta: dict[str, Any] = field(default_factory=dict)
    frontend_video_meta: dict[str, Any] = field(default_factory=dict)
    impact_window_meta: dict[str, Any] = field(default_factory=dict)
    impact_window_start_s: float | None = None


def prepare_pro_analysis_artifacts(
    input_video_path: str,
    work_dir: str,
    *,
    rough_impact_time_s: float | None = None,
    settings: ProAnalysisChainSettings | None = None,
) -> ProAnalysisArtifacts:
    """Run FFmpeg/ffprobe prep and decide analysis / frontend / impact paths."""
    settings = settings or ProAnalysisChainSettings()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    inp = str(Path(input_video_path).resolve())

    t_all = time.perf_counter()
    log_stage_start(ROLE_FFMPEG_PREP, input=inp, work_dir=str(work))
    try:
        verify_ffmpeg()
    except (FFmpegBinaryMissingError, FFmpegProcessError, OSError, ValueError) as exc:
        wall = round(time.perf_counter() - t_all, 3)
        log_stage_failed(ROLE_FFMPEG_PREP, wall_s=wall, reason=type(exc).__name__, detail=str(exc))
        raise

    analysis_path = str(work / "analysis_240fps.mp4")
    frontend_path = str(work / "frontend_240fps.mp4")

    try:
        build_full_240fps_playback(
            inp,
            analysis_path,
            fast=settings.analysis_240_fast,
        )
        build_frontend_playback_from_analysis(
            analysis_path,
            frontend_path,
        )

        impact_path: str | None = None
        impact_start: float | None = None
        impact_meta: dict[str, Any] = {}
        if rough_impact_time_s is not None:
            start_s, duration_s = suggest_impact_window(
                float(rough_impact_time_s),
                pre_s=settings.impact_window_pre_s,
                duration_s=settings.impact_window_duration_s,
            )
            impact_start = start_s
            impact_path = str(work / "impact_window_240fps.mp4")
            trim_video_segment_h264(
                analysis_path,
                impact_path,
                start_s=start_s,
                duration_s=duration_s,
            )
            impact_meta = probe_video(impact_path)
    except (FFmpegBinaryMissingError, FFmpegProcessError, OSError, ValueError) as exc:
        wall = round(time.perf_counter() - t_all, 3)
        log_stage_failed(ROLE_FFMPEG_PREP, wall_s=wall, reason=type(exc).__name__, detail=str(exc))
        raise

    wall2 = round(time.perf_counter() - t_all, 3)
    log_stage_done(
        ROLE_FFMPEG_PREP,
        wall_s=wall2,
        analysis=analysis_path,
        frontend=frontend_path,
        impact_window=impact_path or "none",
    )

    t3 = time.perf_counter()
    log_stage_start(ROLE_PRO_CHAIN)
    artifacts = ProAnalysisArtifacts(
        work_dir=str(work),
        input_path=inp,
        analysis_video_path=analysis_path,
        frontend_video_path=frontend_path,
        impact_window_video_path=impact_path,
        analysis_video_meta=probe_video(analysis_path),
        frontend_video_meta=probe_video(frontend_path),
        impact_window_meta=impact_meta,
        impact_window_start_s=impact_start,
    )
    log_stage_done(
        ROLE_PRO_CHAIN,
        wall_s=round(time.perf_counter() - t3, 3),
        analysis_video_path=artifacts.analysis_video_path,
        frontend_video_path=artifacts.frontend_video_path,
        impact_window_video_path=artifacts.impact_window_video_path or "none",
    )
    return artifacts
