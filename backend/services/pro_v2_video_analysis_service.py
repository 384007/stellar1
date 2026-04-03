"""Pro v2 — single orchestrator (240fps → swing window → dense scan → keyframes → impact → gate → sheet → AI).

Does **not** use MMAction2 / TSN or any legacy Pro pose chain — FFmpeg + OpenCV motion + Gemini text only.
Gemini uses ``PRO_V2_REPORT_PROMPT``: one-line summaries; 3 phase-tagged issues; 3 actionable suggestions.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from services.pro_contact_sheet_service import build_pro_keyframe_contact_sheet
from services.pro_v2_dense_scan_service import DenseFrame, dense_scan_swing_region
from services.pro_v2_ffmpeg_service import run_pro_v2_ffmpeg_preprocess
from services.pro_v2_impact_refine_service import refine_impact_keyframe_only
from services.pro_v2_keyframe_picker_service import pick_eight_keyframes_motion_only
from services.pro_v2_report_service import write_pro_v2_ai_report
from services.pro_v2_screen_preprocess_service import run_pro_v2_screen_preprocess
from services.pro_v2_simple_gate_service import run_simple_gate
from services.pro_v2_swing_window_service import find_swing_window_seconds

logger = logging.getLogger(__name__)


def _nearest_dense_motion(dense: list[DenseFrame], frame_index: int) -> float:
    if not dense:
        return 0.0
    best = dense[0]
    bd = abs(best.frame_index - frame_index)
    for d in dense[1:]:
        dd = abs(d.frame_index - frame_index)
        if dd < bd:
            bd = dd
            best = d
    return float(best.motion_energy_smooth)


def _build_motion_context(
    *,
    fps: float,
    swing_t0: float,
    swing_t1: float,
    dense: list[DenseFrame],
    keyframes: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for k in keyframes:
        fi = int(k.get("frame_index") or 0)
        rows.append(
            {
                "phase": k.get("phase"),
                "label_en": k.get("label_en"),
                "timestamp_s": float(k.get("timestamp") or 0.0),
                "frame_index": fi,
                "dense_motion_proxy": round(_nearest_dense_motion(dense, fi), 6),
            }
        )
    return {
        "pipeline": "pro_v2",
        "motion_authority": "dense_opencv_only",
        "fps": fps,
        "swing_window_s": [round(swing_t0, 4), round(swing_t1, 4)],
        "dense_frame_count": len(dense),
        "keyframes": rows,
    }


async def run_pro_v2_video_analysis(
    input_video_path: str,
    work_dir: str,
    *,
    rough_impact_time_s: float | None = None,
    screen_mode: bool = False,
    region: str = "global",
) -> dict[str, Any]:
    """End-to-end Pro v2; returns minimal public JSON for the frontend."""
    analysis_id = str(uuid.uuid4())
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[PRO_V2] start analysis_id=%s chain=ffmpeg_opencv_motion_gemini mmaction2_used=false",
        analysis_id,
    )

    ffmpeg_input_path = input_video_path
    analysis_input = "raw"
    if screen_mode:
        try:
            screen = run_pro_v2_screen_preprocess(input_video_path=input_video_path, work_dir=str(work))
            ffmpeg_input_path = str(screen.get("cropped_video_path") or input_video_path)
            analysis_input = "screen_cropped" if ffmpeg_input_path != input_video_path else "raw"
        except Exception as exc:
            logger.warning("[PRO_V2][SCREEN] fallback_raw_input=true reason=%s", exc)
            logger.warning("[PRO_V2][SCREEN] screen preprocess failed")
            logger.warning("[PRO_V2][SCREEN] fallback to raw input")
            ffmpeg_input_path = input_video_path
            analysis_input = "raw"
    logger.info("[PRO_V2][PIPELINE] analysis_input=%s", analysis_input)

    ff = run_pro_v2_ffmpeg_preprocess(
        ffmpeg_input_path,
        str(work),
        rough_impact_time_s=rough_impact_time_s,
    )

    t0, t1 = find_swing_window_seconds(
        ff.analysis_240_path,
        fps=ff.fps,
        duration_s=ff.duration_s,
    )

    dense = dense_scan_swing_region(
        ff.analysis_240_path,
        fps=ff.fps,
        t_start_s=t0,
        t_end_s=t1,
    )
    if len(dense) < 16:
        raise RuntimeError("pro_v2: swing region too short or static — record a clearer swing clip")

    keyframes = pick_eight_keyframes_motion_only(ff.analysis_240_path, dense)
    keyframes = refine_impact_keyframe_only(ff.analysis_240_path, keyframes)

    keyframes, _gate_ok, _gate_issues = run_simple_gate(
        keyframes,
        fps=ff.fps,
        analysis_video_path=ff.analysis_240_path,
        dense=dense,
    )

    motion_context = _build_motion_context(
        fps=ff.fps,
        swing_t0=t0,
        swing_t1=t1,
        dense=dense,
        keyframes=keyframes,
    )
    report = await write_pro_v2_ai_report(motion_context, region=region)

    issues = list(report.get("issues") or [])[:3]
    issues_zh = list(report.get("issues_zh") or [])[:3]
    suggestions = list(report.get("suggestions") or [])[:3]
    suggestions_zh = list(report.get("suggestions_zh") or [])[:3]
    report["issues"] = issues
    report["issues_zh"] = issues_zh
    report["suggestions"] = suggestions
    report["suggestions_zh"] = suggestions_zh

    sheet_path = str(work / "pro_v2_contact_sheet.jpg")
    build_pro_keyframe_contact_sheet(keyframes, sheet_path)

    total_score = report.get("total_score")
    try:
        total_score_f = float(total_score) if total_score is not None else 0.0
    except (TypeError, ValueError):
        total_score_f = 0.0

    minimal: dict[str, Any] = {
        "analysis_id": analysis_id,
        "status": "completed",
        "summary": str(report.get("summary") or ""),
        "summary_zh": str(report.get("summary_zh") or report.get("summary") or ""),
        "total_score": total_score_f,
        "keyframes": keyframes,
        "contact_sheet_url": sheet_path,
        "video_url": ff.playback_path,
        "issues": issues,
        "issues_zh": issues_zh,
        "suggestions": suggestions,
        "suggestions_zh": suggestions_zh,
        "scores": report.get("scores") or {},
        "type": "pro",
    }

    logger.info("[PRO_V2] done analysis_id=%s kfs=%s", analysis_id, len(keyframes))
    return minimal
