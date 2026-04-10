"""Stellar Pro — single orchestrator for /stellar-pro/analyze (240fps motion-first chain)."""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

import cv2

from services.keyframe_service import PHASE_ORDER
from services.pose_service import extract_poses_from_video
from services.pro_analysis_chain_service import ProAnalysisChainSettings
from services.pro_contact_sheet_service import build_pro_keyframe_contact_sheet
from services.pro_final_gate_service import run_pro_final_gate
from services.pro_impact_refine_service import refine_impact_pose_index
from services.pro_minimal_public_result_service import pack_pro_minimal_public_result
from services.pro_motion_feature_service import extract_motion_features
from services.pro_motion_keyframe_service import (
    enforce_monotonic_phase_picks,
    select_motion_keyframe_picks,
)
from services.pro_motion_phase_window_service import build_motion_phase_windows
from services.pro_ffmpeg_preprocess_service import run_pro_ffmpeg_preprocess
from services.pro_report_service import run_pro_ai_report
from services.pro_ui_bundle_service import build_stellar_pro_ui_bundle
from services.stellar_pro_club_vision_service import (
    apply_impact_club_vision_to_result,
    mirror_hand_club_flags_to_top_level,
)

logger = logging.getLogger(__name__)


def _wall(t0: float) -> float:
    return round(time.perf_counter() - t0, 3)


def _pose_frame_range_for_swing(
    analysis_video_path: str,
    *,
    fps: float,
    duration_s: float,
    rough_impact_time_s: float | None,
) -> tuple[int, int] | None:
    """Restrict pose sampling to a few seconds around expected impact (analysis timeline).

    Full-clip uniform sampling spreads ~180 poses across long idle + follow-through,
    so eight phase windows span multi-second gaps — keyframes look like a broken strip.
    """
    cap = cv2.VideoCapture(analysis_video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 1:
        return None
    fps = max(fps, 1e-3)
    dur = max(float(duration_s), total / fps)
    imp_t = (
        float(rough_impact_time_s)
        if rough_impact_time_s is not None
        else max(0.05, min(dur * 0.72, dur - 0.05))
    )
    imp_t = max(0.0, min(imp_t, dur - 0.04))
    before_s = 3.8
    after_s = 2.6
    t_lo = max(0.0, imp_t - before_s)
    t_hi = min(dur - 1.0 / fps, imp_t + after_s)
    fi_lo = int(t_lo * fps)
    fi_hi = int(math.ceil(t_hi * fps))
    fi_lo = max(0, min(fi_lo, total - 1))
    fi_hi = max(0, min(fi_hi, total - 1))
    span = fi_hi - fi_lo + 1
    if span < int(0.7 * fps):
        return None
    if fi_hi <= fi_lo:
        return None
    logger.info(
        "[STELLAR_PRO][POSE] swing_window frames=[%s,%s] span=%s imp_t≈%.2fs dur=%.2fs",
        fi_lo,
        fi_hi,
        span,
        imp_t,
        dur,
    )
    return fi_lo, fi_hi


def _sync_stellar_pro_core(
    input_video_path: str,
    work_dir: str,
    analysis_id: str,
    *,
    rough_impact_time_s: float | None,
    chain_settings: ProAnalysisChainSettings | None,
    keyframe_width: int,
) -> dict[str, Any]:
    """Thread-pool body: FFmpeg → pose → motion → keyframes → gate (sync parts)."""
    t_chain = time.perf_counter()

    t_ff = time.perf_counter()
    logger.info("[STELLAR_PRO][FFMPEG_PREP] stage=start input=%s", input_video_path)
    ff = run_pro_ffmpeg_preprocess(
        input_video_path,
        work_dir,
        rough_impact_time_s=rough_impact_time_s,
        settings=chain_settings,
    )
    analysis_path = str(ff["analysis_video_path"])
    fps_a = float(ff["fps"])
    logger.info(
        "[STELLAR_PRO][FFMPEG_PREP] stage=done wall_s=%s analysis=%s fps=%.1f frames=%s",
        _wall(t_ff),
        analysis_path,
        fps_a,
        ff.get("analysis_frame_count"),
    )

    t_pose = time.perf_counter()
    logger.info("[STELLAR_PRO][POSE] stage=start video=%s", analysis_path)
    dur_s = float(ff.get("duration_s") or 0.0)
    swing_range = _pose_frame_range_for_swing(
        analysis_path,
        fps=fps_a,
        duration_s=dur_s,
        rough_impact_time_s=rough_impact_time_s,
    )
    poses, _pose_bundle = extract_poses_from_video(
        analysis_path,
        max_frames=120,
        include_images=True,
        apply_smoothing=True,
        frame_index_range=swing_range,
        target_pose_count=220,
    )
    if len(poses) < 40 and swing_range is not None:
        logger.warning(
            "[STELLAR_PRO][POSE] swing_window only got %d poses; retry full-clip sampling",
            len(poses),
        )
        poses, _pose_bundle = extract_poses_from_video(
            analysis_path,
            max_frames=120,
            include_images=True,
            apply_smoothing=True,
        )
    logger.info(
        "[STELLAR_PRO][POSE] stage=done wall_s=%s poses=%d",
        _wall(t_pose),
        len(poses),
    )
    if len(poses) < 8:
        raise RuntimeError("STELLAR_PRO: insufficient poses on analysis video")

    t_mf = time.perf_counter()
    logger.info("[STELLAR_PRO][MOTION_FEATURE] stage=start")
    feats = extract_motion_features(poses)
    logger.info(
        "[STELLAR_PRO][MOTION_FEATURE] stage=done wall_s=%s n=%s",
        _wall(t_mf),
        feats.get("n"),
    )

    t_pw = time.perf_counter()
    logger.info("[STELLAR_PRO][PHASE_WINDOW] stage=start")
    imp_hint_pi: int | None = None
    if rough_impact_time_s is not None and poses and fps_a > 1e-3:
        target_fi = int(round(float(rough_impact_time_s) * fps_a))
        imp_hint_pi = min(
            range(len(poses)),
            key=lambda i: abs(int(poses[i].get("frame_index", i)) - target_fi),
        )
    windows, events = build_motion_phase_windows(
        poses, feats, rough_impact_pose_idx=imp_hint_pi
    )
    logger.info(
        "[STELLAR_PRO][PHASE_WINDOW] stage=done wall_s=%s windows=%d top=%s impact=%s",
        _wall(t_pw),
        len(windows),
        events.get("top_pose_idx"),
        events.get("impact_pose_idx"),
    )

    t_mk = time.perf_counter()
    logger.info("[STELLAR_PRO][MOTION_KEYFRAME] stage=start")
    picks = select_motion_keyframe_picks(windows, poses, feats, events)
    logger.info(
        "[STELLAR_PRO][MOTION_KEYFRAME] stage=done wall_s=%s picks=%s",
        _wall(t_mk),
        {k: picks[k] for k in PHASE_ORDER},
    )

    imp_clip = ff.get("impact_window_video_path")
    imp_start = ff.get("impact_window_start_s")
    t_ir = time.perf_counter()
    logger.info("[STELLAR_PRO][IMPACT_REFINE] stage=start rough=%s", picks.get("impact"))
    refined_imp, imp_meta = refine_impact_pose_index(
        str(imp_clip) if imp_clip else None,
        analysis_path,
        poses,
        int(picks["impact"]),
        impact_window_start_s=float(imp_start) if imp_start is not None else None,
        analysis_fps=fps_a,
    )
    picks["impact"] = int(refined_imp)
    picks = enforce_monotonic_phase_picks(picks, len(poses))
    logger.info(
        "[STELLAR_PRO][IMPACT_REFINE] stage=done wall_s=%s final=%s clip_used=%s",
        _wall(t_ir),
        picks["impact"],
        imp_meta.get("clip_used"),
    )

    dur = float(ff.get("duration_s") or 1.0)
    min_time_gap = max(dur * (1.0 / 24.0), 0.04)

    def _refine_impact_again(rough: int) -> int:
        r, _ = refine_impact_pose_index(
            str(imp_clip) if imp_clip else None,
            analysis_path,
            poses,
            int(rough),
            impact_window_start_s=float(imp_start) if imp_start is not None else None,
            analysis_fps=fps_a,
        )
        return int(r)

    t_g = time.perf_counter()
    logger.info("[STELLAR_PRO][FINAL_GATE] stage=start")
    kfs, pk, gate_summary = run_pro_final_gate(
        analysis_path,
        poses,
        picks,
        windows,
        features=feats,
        events=events,
        analysis_fps=fps_a,
        keyframe_width=keyframe_width,
        min_time_gap=min_time_gap,
        impact_refine_fn=_refine_impact_again,
    )
    logger.info(
        "[STELLAR_PRO][FINAL_GATE] stage=done wall_s=%s pass=%s retry=%s",
        _wall(t_g),
        gate_summary.get("pass"),
        gate_summary.get("retry_used"),
    )

    if len(kfs) != 8:
        raise RuntimeError("STELLAR_PRO: expected 8 keyframes")
    for i, row in enumerate(kfs):
        if str(row.get("phase")) != PHASE_ORDER[i]:
            raise RuntimeError("STELLAR_PRO: keyframe phase order mismatch")
    prev_fi = -1
    for row in kfs:
        fi = int(row.get("source_frame_index") or row.get("frame_index") or -1)
        if fi < prev_fi:
            raise RuntimeError("STELLAR_PRO: source_frame_index not monotonic")
        prev_fi = fi

    logger.info(
        "[STELLAR_PRO][PUBLIC_PACK] stage=presync wall_s=%s total_sync=%.2fs",
        _wall(t_chain),
        time.perf_counter() - t_chain,
    )

    t_ui = time.perf_counter()
    logger.info("[STELLAR_PRO][UI_BUNDLE] stage=start")
    ui_bundle = build_stellar_pro_ui_bundle(
        poses,
        pk,
        fps=fps_a,
        source_frame_count=int(ff.get("source_frame_count") or 0),
        analysis_frame_count=int(ff.get("analysis_frame_count") or 0),
        detected_club=None,
    )
    logger.info("[STELLAR_PRO][UI_BUNDLE] stage=done wall_s=%s", _wall(t_ui))

    return {
        "analysis_id": analysis_id,
        "keyframes": kfs,
        "poses": poses,
        "ff": ff,
        "phase_keyframes": pk,
        "fps_a": fps_a,
        "imp_meta": imp_meta,
        "gate_summary": gate_summary,
        "ui_bundle": ui_bundle,
    }


async def run_stellar_pro_video_analysis(
    input_video_path: str,
    work_dir: str,
    *,
    rough_impact_time_s: float | None = None,
    region: str = "global",
    chain_settings: ProAnalysisChainSettings | None = None,
    keyframe_width: int = 320,
) -> dict[str, Any]:
    """End-to-end Stellar Pro video analysis → minimal public JSON for the client."""
    t_all = time.perf_counter()
    analysis_id = str(uuid.uuid4())
    work = Path(work_dir)

    partial = await asyncio.to_thread(
        _sync_stellar_pro_core,
        input_video_path,
        work_dir,
        analysis_id,
        rough_impact_time_s=rough_impact_time_s,
        chain_settings=chain_settings,
        keyframe_width=keyframe_width,
    )

    poses: list[dict] = list(partial["poses"])
    ff: dict[str, Any] = partial["ff"]
    kfs: list[dict] = list(partial["keyframes"])

    raw: dict[str, Any] = {
        "analysis_id": partial["analysis_id"],
        "type": "stellar_pro",
        "status": "completed",
        "summary": None,
        "summary_zh": None,
        "total_score": 0,
        "keyframes": kfs,
        "phase_keyframes": dict(partial.get("phase_keyframes") or {}),
        "video_url": str(ff.get("frontend_video_path") or "") or None,
        "contact_sheet_url": None,
    }

    ub = partial.get("ui_bundle")
    if isinstance(ub, dict):
        raw.update(ub)

    await apply_impact_club_vision_to_result(
        raw,
        kfs,
        region=region,
        source_video_path=input_video_path,
    )

    t_cs = time.perf_counter()
    logger.info("[STELLAR_PRO][CONTACT_SHEET] stage=start")
    try:
        if kfs:
            out_path = str(work / "contact_sheet.jpg")
            build_pro_keyframe_contact_sheet(kfs, out_path)
            raw["contact_sheet_url"] = out_path
            logger.info(
                "[STELLAR_PRO][CONTACT_SHEET] stage=done wall_s=%s path=%s",
                _wall(t_cs),
                out_path,
            )
    except Exception as exc:
        logger.warning("[STELLAR_PRO][CONTACT_SHEET] stage=failed err=%s", exc)

    t_rep = time.perf_counter()
    logger.info("[STELLAR_PRO][REPORT] stage=start (async)")
    try:
        report = await run_pro_ai_report(
            poses,
            kfs,
            impact_meta=partial.get("imp_meta"),
            region=region,
        )
        raw["summary"] = report.get("summary")
        raw["summary_zh"] = report.get("summary_zh")
        raw["total_score"] = int(report.get("total_score") or 0)
        raw["issues"] = report.get("issues") or []
        raw["issues_zh"] = report.get("issues_zh") or []
        raw["suggestions"] = report.get("suggestions") or []
        raw["suggestions_zh"] = report.get("suggestions_zh") or []
        if report.get("scores") is not None:
            raw["scores"] = report["scores"]
        if report.get("advanced_metrics") is not None:
            raw["advanced_metrics"] = report["advanced_metrics"]
        if report.get("training_plan") is not None:
            raw["training_plan"] = report["training_plan"]
    except asyncio.TimeoutError:
        logger.warning("[STELLAR_PRO][REPORT] timeout")
        raw["summary_zh"] = raw["summary_zh"] or "报告生成超时"
    except Exception as exc:
        logger.warning("[STELLAR_PRO][REPORT] failed: %s", exc)
        raw["summary_zh"] = raw["summary_zh"] or f"报告暂不可用 ({type(exc).__name__})"

    logger.info("[STELLAR_PRO][REPORT] stage=async_done wall_s=%s", _wall(t_rep))

    mirror_hand_club_flags_to_top_level(raw)

    public = pack_pro_minimal_public_result(raw)
    logger.info(
        "[STELLAR_PRO][PUBLIC_PACK] stage=done wall_s=%s total_wall_s=%s analysis_id=%s kf=%s",
        _wall(t_all),
        round(time.perf_counter() - t_all, 3),
        public.get("analysis_id"),
        len(public.get("keyframes") or []),
    )
    return public
