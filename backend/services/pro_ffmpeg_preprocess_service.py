"""Pro Stage 0: unified 240fps FFmpeg bundle + stable meta for the motion-first Pro chain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from services.ffmpeg_preprocess_service import (
    estimated_frame_count,
    ffprobe_fps,
    probe_video,
    safe_duration_s,
)
from services.pro_analysis_chain_service import (
    ProAnalysisChainSettings,
    prepare_pro_analysis_artifacts,
)

logger = logging.getLogger(__name__)


def run_pro_ffmpeg_preprocess(
    input_video_path: str,
    work_dir: str,
    *,
    rough_impact_time_s: float | None = None,
    settings: ProAnalysisChainSettings | None = None,
) -> dict[str, Any]:
    """
    Build analysis (240fps), frontend playback, optional impact trim — all from analysis timeline.
    On FFmpeg failure raises (no silent downgrade to native fps).
    """
    t0 = __import__("time").perf_counter()
    inp = str(Path(input_video_path).resolve())
    src_meta = probe_video(inp)
    src_dur = safe_duration_s(src_meta)
    if rough_impact_time_s is None:
        rough_impact_time_s = max(0.1, src_dur * 0.72)

    logger.info("[STELLAR_PRO][FFMPEG_PREP] stage=start input=%s rough_impact_s=%.3f", inp, rough_impact_time_s)
    art = prepare_pro_analysis_artifacts(
        inp,
        work_dir,
        rough_impact_time_s=float(rough_impact_time_s),
        settings=settings,
    )
    am = art.analysis_video_meta
    fm = art.frontend_video_meta
    im = art.impact_window_meta or {}
    fps_a = ffprobe_fps(am)
    out = {
        "fps": fps_a,
        "duration_s": round(safe_duration_s(am), 4),
        "source_frame_count": estimated_frame_count(src_meta),
        "analysis_frame_count": estimated_frame_count(am),
        "frame_size": {
            "width": int(am.get("width") or 0),
            "height": int(am.get("height") or 0),
        },
        "analysis_video_path": art.analysis_video_path,
        "frontend_video_path": art.frontend_video_path,
        "impact_window_video_path": art.impact_window_video_path,
        "impact_window_start_s": art.impact_window_start_s,
        "frontend_fps": ffprobe_fps(fm),
        "impact_window_fps": ffprobe_fps(im) if im else None,
    }
    wall = round(__import__("time").perf_counter() - t0, 3)
    logger.info(
        "[STELLAR_PRO][FFMPEG_PREP] stage=done wall_s=%s analysis=%s frontend=%s impact=%s fps=%.1f frames=%s",
        wall,
        out["analysis_video_path"],
        out["frontend_video_path"],
        out["impact_window_video_path"] or "none",
        fps_a,
        out["analysis_frame_count"],
    )
    return out
