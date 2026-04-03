"""Pro motion-first keyframe orchestrator: 240fps → pose → windows → window-AI → impact refine → gate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2

from services.keyframe_service import (
    PHASE_ORDER,
    recompute_keyframe_details_from_final_strip,
    validate_final_keyframes_for_ai,
)
from services.pose_service import extract_poses_from_video
from services.pro_ffmpeg_preprocess_service import run_pro_ffmpeg_preprocess
from services.pro_impact_refine_service import refine_impact_pose_index
from services.pro_motion_feature_service import extract_motion_features
from services.pro_motion_keyframe_service import (
    build_keyframes_from_motion_picks as _build_keyframes_from_picks,
    enforce_monotonic_phase_picks as _enforce_monotonic_picks,
)
from services.pro_motion_phase_window_service import build_motion_phase_windows
from services.pro_window_ai_keyframe_service import select_frames_with_window_ai
from services.swing_flow_utils import detect_swing_phases
from services.video_utils import get_video_rotation

logger = logging.getLogger(__name__)


# Legacy chain: window-AI path still uses _gate_and_maybe_retry with these imports.


@dataclass
class ProMotionKeyframeResult:
    poses: list[dict]
    pose_quality_bundle: dict[str, Any]
    swing_phases: list[dict]
    phase_keyframes: dict[str, int]
    keyframes: list[dict[str, Any]]
    kf_validation: dict[str, Any]
    ffmpeg_meta: dict[str, Any]
    internal: dict[str, Any] = field(default_factory=dict)


def _gate_and_maybe_retry(
    analysis_video_path: str,
    poses: list[dict],
    phase_keyframes: dict[str, int],
    windows: list[dict[str, Any]],
    *,
    analysis_fps: float,
    keyframe_width: int,
    min_time_gap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    pk = dict(phase_keyframes)
    kfs = _build_keyframes_from_picks(
        analysis_video_path, poses, pk, analysis_fps=analysis_fps, keyframe_width=keyframe_width,
    )
    cap = cv2.VideoCapture(analysis_video_path)
    if not cap.isOpened():
        raise RuntimeError("analysis_video_missing")
    rotation = get_video_rotation(analysis_video_path)
    try:
        det = recompute_keyframe_details_from_final_strip(
            cap, rotation, float(analysis_fps), poses, kfs, min_time_gap,
        )
        gate = validate_final_keyframes_for_ai(kfs, pk, det, poses=poses, fps=float(analysis_fps))
    finally:
        cap.release()

    if gate.get("pass"):
        return kfs, {
            "details": det,
            "near_duplicates": int(gate.get("near_duplicates", 0)),
            "time_too_close": int(gate.get("time_too_close_count", 0)),
            "final_keyframe_gate_pass": True,
            "final_phase_keyframes": dict(pk),
            "final_keyframe_validation": dict(gate),
            "final_keyframe_order_ok": bool(gate.get("final_keyframe_order_ok")),
            "final_keyframe_time_order_ok": bool(gate.get("final_keyframe_time_order_ok")),
            "all_passed": True,
            "final_keyframe_source": "motion_chain_v1",
        }, pk

    # One retry: snap failing phases to window center
    bad_phases = {
        str(d.get("phase"))
        for d in det
        if isinstance(d, dict)
        and (not d.get("validation_passed", True) or d.get("is_near_duplicate") or d.get("time_too_close"))
    }
    wby = {str(w["phase"]): w for w in windows}
    for ph in bad_phases:
        if ph in wby:
            pk[ph] = int(wby[ph]["center_pose_idx"])
    pk = _enforce_monotonic_picks(pk, len(poses))
    kfs = _build_keyframes_from_picks(
        analysis_video_path, poses, pk, analysis_fps=analysis_fps, keyframe_width=keyframe_width,
    )
    cap = cv2.VideoCapture(analysis_video_path)
    rotation = get_video_rotation(analysis_video_path)
    try:
        det = recompute_keyframe_details_from_final_strip(
            cap, rotation, float(analysis_fps), poses, kfs, min_time_gap,
        )
        gate = validate_final_keyframes_for_ai(kfs, pk, det, poses=poses, fps=float(analysis_fps))
    finally:
        cap.release()

    return kfs, {
        "details": det,
        "near_duplicates": int(gate.get("near_duplicates", 0)),
        "time_too_close": int(gate.get("time_too_close_count", 0)),
        "final_keyframe_gate_pass": bool(gate.get("pass")),
        "final_phase_keyframes": dict(pk),
        "final_keyframe_validation": dict(gate),
        "final_keyframe_order_ok": bool(gate.get("final_keyframe_order_ok")),
        "final_keyframe_time_order_ok": bool(gate.get("final_keyframe_time_order_ok")),
        "all_passed": bool(gate.get("pass")),
        "final_keyframe_source": "motion_chain_v1",
    }, pk


async def run_pro_motion_keyframe_chain(
    upload_path: str,
    *,
    work_dir: str,
    keyframe_width: int = 320,
    region: str = "global",
) -> ProMotionKeyframeResult:
    t_chain = time.perf_counter()

    ff = run_pro_ffmpeg_preprocess(upload_path, work_dir)
    analysis_path = ff["analysis_video_path"]
    fps_a = float(ff["fps"])

    t0 = time.perf_counter()
    logger.info("[PRO][pose] stage=start video=%s", analysis_path)
    poses, pose_bundle = extract_poses_from_video(
        analysis_path,
        max_frames=120,
        include_images=True,
        apply_smoothing=True,
    )
    logger.info(
        "[PRO][pose] stage=done wall_s=%.2f poses=%s",
        time.perf_counter() - t0,
        len(poses),
    )
    if len(poses) < 8:
        raise ValueError("PRO_MOTION_CHAIN: insufficient poses on 240fps analysis video")

    t1 = time.perf_counter()
    feats = extract_motion_features(poses)
    logger.info("[PRO][motion_features] stage=done wall_s=%.2f", time.perf_counter() - t1)

    t2 = time.perf_counter()
    windows, _phase_events = build_motion_phase_windows(poses, feats)
    logger.info("[PRO][phase_windows] stage=done wall_s=%.2f", time.perf_counter() - t2)

    t3 = time.perf_counter()
    ai_picks = await select_frames_with_window_ai(
        windows, poses, analysis_path, region=region, max_candidates=5,
    )
    logger.info("[PRO][window_ai] stage=done wall_s=%.2f", time.perf_counter() - t3)

    picks = {ph: int(ai_picks[ph]["pose_idx"]) for ph in PHASE_ORDER}
    rough_imp = picks["impact"]
    imp_start = ff.get("impact_window_start_s")
    imp_clip = ff.get("impact_window_video_path")

    t4 = time.perf_counter()
    refined_imp, imp_meta = refine_impact_pose_index(
        str(imp_clip) if imp_clip else None,
        analysis_path,
        poses,
        rough_imp,
        impact_window_start_s=float(imp_start) if imp_start is not None else None,
        analysis_fps=fps_a,
    )
    picks["impact"] = refined_imp
    logger.info("[PRO][impact_refine] stage=done wall_s=%.2f", time.perf_counter() - t4)

    picks = _enforce_monotonic_picks(picks, len(poses))

    dur = float(ff.get("duration_s") or 1.0)
    min_time_gap = max(dur * (1.0 / 24.0), 0.04)

    kfs, kf_val, pk = _gate_and_maybe_retry(
        analysis_path,
        poses,
        picks,
        windows,
        analysis_fps=fps_a,
        keyframe_width=keyframe_width,
        min_time_gap=min_time_gap,
    )

    swing_phases = detect_swing_phases(poses)
    wall = round(time.perf_counter() - t_chain, 3)
    logger.info(
        "[PRO][motion_chain] stage=complete wall_s=%s gate_pass=%s",
        wall,
        kf_val.get("final_keyframe_gate_pass"),
    )

    return ProMotionKeyframeResult(
        poses=poses,
        pose_quality_bundle=pose_bundle,
        swing_phases=swing_phases,
        phase_keyframes=pk,
        keyframes=kfs,
        kf_validation=kf_val,
        ffmpeg_meta=ff,
        internal={"impact_refine": imp_meta, "window_ai": ai_picks},
    )
