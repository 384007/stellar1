from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict

from lib.prov3.keyframes.types import PreprocessMeta, PreprocessResult
from services.internal.frame_enhance_service import generate_analysis_frames
from services.internal.video_240fps_service import build_analysis_timeline
from services.internal.video_cleanup_service import cleanup_video


def run_preprocess(input_video: str, work_dir: str, *, screen_mode: bool = False) -> PreprocessResult:
    analysis_id = f"prov3_{uuid.uuid4().hex[:12]}"
    local_dir = str(Path(work_dir) / analysis_id)

    cleanup = cleanup_video(input_video, local_dir, screen_mode=screen_mode)
    timeline = build_analysis_timeline(str(cleanup["analysis_video"]), local_dir)
    afps = int(timeline.get("analysis_fps", 240))
    frames = generate_analysis_frames(
        str(timeline["analysis_video"]),
        local_dir,
        analysis_fps=afps,
    )

    meta = PreprocessMeta(
        source_fps=float(cleanup.get("source_fps", 30.0)),
        analysis_fps=int(timeline.get("analysis_fps", 240)),
        stabilized=bool(cleanup.get("stabilized", True)),
        denoised=bool(cleanup.get("denoised", True)),
        cropped_single_swing=bool(cleanup.get("cropped_single_swing", True)),
        screen_mode_corrected=bool(cleanup.get("screen_mode_corrected", False)),
    )

    return PreprocessResult(
        analysis_id=analysis_id,
        analysis_video=str(timeline["analysis_video"]),
        preprocess_meta=meta,
        analysis_frames=frames["analysis_frames"],
        enhanced_local_frames=frames["enhanced_local_frames"],
    )
