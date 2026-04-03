import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional

from routers.auth import get_current_user
from routers.plus_analyze import _stellar_modal_upload_echo
from services.json_sanitize import log_non_finite_if_any, sanitize_json_floats

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)

router = APIRouter()

_PRO_POSE_UNIFORM_TIMEOUT_S = 480.0
_PRO_PHASE_DETECT_OUTER_S = float(os.getenv("STELLAR_PRO_PHASE_OUTER_S", "105"))
_PRO_KEYFRAME_SMART_TIMEOUT_S = 300.0
_PRO_ENSURE_ORDERED_TIMEOUT_S = 300.0
_PRO_AI_VISION_STRIP_TIMEOUT_S = 180.0
_PRO_SEMANTIC_REPORT_TIMEOUT_S = 120.0
_PRO_FUSION_DETECT_CLUB_S = 45.0
_PRO_FUSION_BLUR_S = 90.0
_PRO_FUSION_TRAJECTORY_S = 90.0
_PRO_FUSION_CALIBRATE_S = 30.0

_LATE_STRIP_PHASES = ("downswing", "impact", "follow_through", "finish")

# Motion-chain Pro responses: omit implementation / gate-debug fields from JSON (logs keep full detail).
_MOTION_PRO_DROP_TOP = frozenset({
    "ai_provider",
    "ai_key",
    "phase_debug",
    "keyframe_validation",
    "final_phase_keyframes",
    "final_keyframe_validation",
    "final_keyframe_order_ok",
    "final_keyframe_time_order_ok",
    "final_keyframe_source",
    "final_keyframe_gate_pass",
    "phase_detector_version",
    "phase_detector_confidence",
    "top_candidate_debug",
    "impact_candidate_debug",
    "top_keyframe_vs_event",
    "impact_keyframe_vs_event",
    "top_semantic_at_keyframe",
    "impact_semantic_at_keyframe",
    "top_semantic_ok",
    "impact_semantic_ok",
    "semantic_validation",
    "final_phase_semantic_ok",
    "phase_evaluations_reliable",
    "phase_evaluations_warning",
    "phase_boundary",
    "sweet_spot_warning",
    "sweet_spot_confidence",
})


def _motion_pro_strip_internal(resp: dict) -> dict:
    out = {k: v for k, v in resp.items() if k not in _MOTION_PRO_DROP_TOP}
    out["status"] = "ok"
    out.setdefault("contact_sheet_url", None)
    out.setdefault("video_url", None)
    kfs_out: list[dict] = []
    for kf in out.get("keyframes") or []:
        if not isinstance(kf, dict):
            continue
        kfs_out.append({kk: vv for kk, vv in kf.items() if kk not in ("selection_reason", "fallback_used")})
    out["keyframes"] = kfs_out
    return out


def _log_late_strip_after_ensure(logger_: logging.Logger, keyframes: list, kf_validation: dict) -> None:
    by_phase = {str(k.get("phase")): k for k in (keyframes or []) if k.get("phase")}
    for pid in _LATE_STRIP_PHASES:
        k = by_phase.get(pid)
        if not k:
            continue
        logger_.info(
            "[PRO] late_strip phase=%s source_pose_idx=%s source_frame_index=%s timestamp=%s",
            pid,
            k.get("source_pose_idx"),
            k.get("source_frame_index", k.get("frame_index")),
            k.get("timestamp"),
        )
    fv = dict(kf_validation.get("final_keyframe_validation") or {})
    logger_.info(
        "[PRO] late_strip_cleanup_summary applied=%s resolved=%s remaining=%s "
        "impact_preserved=%s impact_shift_frames=%s follow_shift_frames=%s finish_shift_frames=%s",
        fv.get("late_strip_cleanup_applied"),
        fv.get("late_strip_cleanup_resolved"),
        fv.get("remaining_near_duplicate_phases"),
        fv.get("impact_preserved"),
        fv.get("impact_shift_frames"),
        fv.get("follow_shift_frames"),
        fv.get("finish_shift_frames"),
    )


def _log_late_strip_final_for_422(logger_: logging.Logger, keyframes: list, kf_validation: dict) -> None:
    by_phase = {str(k.get("phase")): k for k in (keyframes or []) if k.get("phase")}
    logger_.warning("[PRO] strict_gate_fail late_strip_final_dump begin")
    for pid in _LATE_STRIP_PHASES:
        k = by_phase.get(pid)
        if not k:
            logger_.warning("[PRO] strict_gate_fail late_strip phase=%s MISSING", pid)
            continue
        logger_.warning(
            "[PRO] strict_gate_fail late_strip phase=%s source_pose_idx=%s source_frame_index=%s "
            "timestamp=%s selection_reason=%s fallback_used=%s",
            pid,
            k.get("source_pose_idx"),
            k.get("source_frame_index", k.get("frame_index")),
            k.get("timestamp"),
            k.get("selection_reason"),
            k.get("fallback_used"),
        )
    fv = dict(kf_validation.get("final_keyframe_validation") or {})
    logger_.warning(
        "[PRO] strict_gate_fail late_strip_cleanup applied=%s resolved=%s remaining=%s "
        "rounds=%s changed=%s improved=%s reverted=%s gate_pass=%s near_dup=%s tc=%s source=%s",
        fv.get("late_strip_cleanup_applied"),
        fv.get("late_strip_cleanup_resolved"),
        fv.get("remaining_near_duplicate_phases"),
        fv.get("late_strip_cleanup_rounds"),
        fv.get("late_strip_cleanup_changed_phase_ids"),
        fv.get("late_strip_cleanup_improved"),
        fv.get("late_strip_cleanup_reverted"),
        kf_validation.get("final_keyframe_gate_pass"),
        kf_validation.get("near_duplicates"),
        kf_validation.get("time_too_close"),
        kf_validation.get("final_keyframe_source"),
    )
    logger_.warning("[PRO] strict_gate_fail late_strip_final_dump end")


async def _timed_pro_stage(name: str, awaitable, timeout_s: float):
    t0 = time.perf_counter()
    try:
        r = await asyncio.wait_for(awaitable, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.error("[PRO] stage=%s TIMEOUT after %.0fs", name, timeout_s)
        raise HTTPException(
            status_code=504,
            detail=(
                f"分析步骤超时: {name}（>{timeout_s:.0f}s）/ Analysis step timed out: {name}"
            ),
        ) from None
    logger.info("[PRO] stage=%s wall_s=%.2f", name, time.perf_counter() - t0)
    return r


def _extract_uniform_16_pro(tmp_path: str):
    from services.keyframe_service import extract_all_frames_base64_with_indices

    return extract_all_frames_base64_with_indices(tmp_path, 16, 384)


class ProAnalyzeRequest(BaseModel):
    video_url: str
    user_id: Optional[str] = None


def _load_services():
    """Lazy-load heavy dependencies (mediapipe, opencv, genai) at call time."""
    from services.pose_service import extract_poses_from_video
    from services.gemini_service import analyze_swing_pro
    from services.hud_service import generate_hud_data
    from services.shot_predictor import predict_shot, calibrate_prediction
    return extract_poses_from_video, analyze_swing_pro, generate_hud_data, predict_shot, calibrate_prediction


def _extract_poses_pack_pro(tmp_path: str, max_frames: int):
    from services.pose_service import extract_poses_from_video

    return extract_poses_from_video(tmp_path, max_frames=max_frames)


def _load_fusion_services():
    """Lazy-load the club detection + speed fusion modules."""
    from services.club_detector import detect_club
    from services.blur_speed_service import detect_blur_speed
    from services.trajectory_service import track_trajectory
    from services.fusion_service import fuse_speed
    return detect_club, detect_blur_speed, track_trajectory, fuse_speed


def _extract_impact_frames(video_path: str, poses: list[dict], swing_phases: list[dict], window: int = 3) -> tuple[list, int, np.ndarray | None]:
    """
    Extract raw BGR frames around the impact moment for blur/trajectory analysis.

    Returns (frames_around_impact, impact_frame_index, impact_keyframe_or_None).
    """
    import cv2 as _cv2
    from services.video_utils import get_video_rotation, read_frame_pose_pipeline

    impact_pose_idx = len(poses) // 2
    for i, phase in enumerate(swing_phases):
        if phase.get("phase_id") == "impact":
            impact_pose_idx = i
            break

    impact_frame_idx = poses[impact_pose_idx].get("frame_index", 0) if impact_pose_idx < len(poses) else 0

    cap = _cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], impact_frame_idx, None

    rotation = get_video_rotation(video_path)
    total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, impact_frame_idx - window)
    end = min(total - 1, impact_frame_idx + window)

    frames: list = []
    keyframe = None
    for idx in range(start, end + 1):
        frame = read_frame_pose_pipeline(cap, idx, rotation)
        if frame is None:
            continue
        frames.append(frame)
        if idx == impact_frame_idx:
            keyframe = frame

    cap.release()
    return frames, impact_frame_idx, keyframe


@router.post("/pro")
async def analyze_pro(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
):
    if current_user and not current_user.get("is_pro"):
        raise HTTPException(status_code=403, detail="Pro membership required")

    _stellar_modal_upload_echo("PRO", request)
    try:
        extract_poses_from_video, analyze_swing_pro, generate_hud_data, predict_shot, calibrate_prediction = _load_services()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Analysis services unavailable: {e}")

    tmp_path = None
    try:
        content_type = request.headers.get("content-type", "")

        if "multipart" in content_type:
            form = await request.form()
            uploaded_file = form.get("file")
            if not uploaded_file or not hasattr(uploaded_file, "read"):
                raise HTTPException(status_code=400, detail="No file provided")

            file_bytes = await uploaded_file.read()
            if len(file_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty file")

            filename = getattr(uploaded_file, "filename", "video.mp4") or "video.mp4"
            suffix = ".mov" if ".mov" in filename.lower() else ".mp4"
            if ".webm" in filename.lower():
                suffix = ".webm"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

        else:
            body = await request.json()
            video_url = body.get("video_url", "")
            if not video_url:
                raise HTTPException(status_code=400, detail="No video_url provided")

            if video_url.startswith("blob:"):
                raise HTTPException(
                    status_code=400,
                    detail="Blob URLs cannot be fetched by the server. Please upload the file directly.",
                )

            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.get(video_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to download video")

            suffix = ".mov" if ".mov" in video_url.lower() else ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

        t0 = time.time()

        loop = asyncio.get_event_loop()
        region = "CN" if request.headers.get("CF-IPCountry", "").upper() == "CN" else "global"
        use_legacy = os.getenv("STELLAR_PRO_LEGACY_CHAIN", "").strip().lower() in ("1", "true", "yes")
        ff_meta_for_response = None

        from services.handedness_service import detect_handedness
        from services.phase_analysis_gate import (
            build_phase_alignment_fail_detail,
            should_run_phase_analysis_strict,
        )
        from services.pose_service import pose_for_skeleton_render
        from services.shot_predictor import estimate_sweet_spot_robust
        from services.swing_flow_utils import (
            compute_phase_evaluations_reliable,
            compute_wrist_trajectory,
            build_phase_boundary_flags,
        )
        from services.gemini_service import cap_confidence, PRO_AI_TIMEOUT_S

        if use_legacy:
            from services.keyframe_service import (
                extract_keyframes_smart,
                ensure_keyframes_ordered_for_ai,
                build_ai_vision_images_from_phase_keyframes,
            )
            from services.swing_flow_utils import (
                detect_swing_phases,
                get_phase_keyframes,
                map_gemini_uniform_indices_to_pose_indices,
                validate_phase_keyframes,
                respace_phase_keyframes,
                build_semantic_phase_report,
                assess_gemini_uniform_map_vs_final_phase_strip,
            )
            from services.gemini_service import detect_phases_from_frames

            # Stage 1: parallel — pose extraction + uniform frames (exact video index per thumbnail)
            poses_future = loop.run_in_executor(_executor, _extract_poses_pack_pro, tmp_path, 60)
            uniform_future = loop.run_in_executor(_executor, _extract_uniform_16_pro, tmp_path)
            (poses, pose_quality_bundle), (uniform_frames, ai_video_frames_for_gemini) = await _timed_pro_stage(
                "pose_extraction+uniform_frames",
                asyncio.gather(poses_future, uniform_future),
                _PRO_POSE_UNIFORM_TIMEOUT_S,
            )

            t1 = time.time()
            logger.info(
                "[PRO] pose_uniform summary: wall_since_start=%.1fs poses=%d uniform=%d",
                t1 - t0,
                len(poses),
                len(uniform_frames),
            )

            if not poses:
                raise HTTPException(status_code=422, detail="No poses detected in video. Ensure the golfer is clearly visible.")

            # Stage 2: Gemini phase detection (network-bound) in parallel with kinematic detection (CPU)
            gemini_phase_task = asyncio.create_task(detect_phases_from_frames(uniform_frames, region="global"))
            swing_phases = detect_swing_phases(poses)
            gemini_phases = await _timed_pro_stage(
                "detect_phases_from_frames",
                gemini_phase_task,
                _PRO_PHASE_DETECT_OUTER_S,
            )

            phase_source = "kinematic"
            phase_validation = None
            phase_debug: dict = {}
            kinematic_keyframes = get_phase_keyframes(swing_phases, poses)
            kinematic_validation = validate_phase_keyframes(kinematic_keyframes, poses, source="kinematic")

            if gemini_phases:
                n_poses = len(poses)
                n_frames = len(uniform_frames)
                if ai_video_frames_for_gemini and n_frames == len(ai_video_frames_for_gemini):
                    gemini_mapped = map_gemini_uniform_indices_to_pose_indices(
                        gemini_phases, ai_video_frames_for_gemini, poses
                    )
                else:
                    logger.warning(
                        "[PRO] AI vf len %d vs uniform %d — proportional Gemini map (degraded)",
                        len(ai_video_frames_for_gemini or []),
                        n_frames,
                    )
                    gemini_mapped = {
                        pid: min(round(fidx / max(n_frames - 1, 1) * (n_poses - 1)), n_poses - 1)
                        for pid, fidx in gemini_phases.items()
                    }
                gemini_validation = validate_phase_keyframes(gemini_mapped, poses, source="gemini")
                phase_debug["gemini_raw"] = gemini_phases
                phase_debug["gemini_mapped"] = gemini_mapped
                phase_debug["gemini_validation"] = gemini_validation
                phase_debug["kinematic_keyframes"] = kinematic_keyframes
                phase_debug["kinematic_validation"] = kinematic_validation
                if gemini_validation["passed"]:
                    phase_keyframes = gemini_mapped
                    phase_source = "gemini"
                    phase_validation = gemini_validation
                    logger.info("[PRO] Gemini phase detection PASSED validation: %s", gemini_mapped)
                else:
                    logger.warning(
                        "[PRO] Gemini phase FAILED validation (issues=%s), fallback kinematic",
                        gemini_validation["issues"],
                    )
                    if kinematic_validation["passed"]:
                        phase_keyframes = kinematic_keyframes
                        phase_source = "kinematic"
                        phase_validation = kinematic_validation
                    else:
                        phase_keyframes = kinematic_keyframes
                        phase_source = "kinematic_degraded"
                        phase_validation = kinematic_validation
                        logger.warning("[PRO] Both Gemini and kinematic validation failed")
            else:
                phase_keyframes = kinematic_keyframes
                phase_validation = kinematic_validation
                phase_debug["kinematic_keyframes"] = kinematic_keyframes
                phase_debug["kinematic_validation"] = kinematic_validation
                logger.info("[PRO] Gemini phase None, using kinematic")

            need_respace = phase_source == "kinematic_degraded" or (
                phase_validation is not None and not phase_validation.get("spacing_ok", True)
            )
            if need_respace:
                _src_before_respace = phase_source
                phase_keyframes = respace_phase_keyframes(dict(phase_keyframes), len(poses))
                phase_debug["phase_keyframes_respaced"] = True
                phase_debug["phase_keyframes_after_respace"] = dict(phase_keyframes)
                phase_validation = validate_phase_keyframes(phase_keyframes, poses, source=phase_source)
                phase_debug["phase_validation_after_respace"] = phase_validation
                if phase_validation.get("passed"):
                    if _src_before_respace == "kinematic_degraded":
                        phase_source = "kinematic_respaced"
                    elif _src_before_respace == "gemini":
                        phase_source = "gemini_respaced"

            # ── Smart keyframes: images extracted at actual phase moments ──
            phase_keyframes_snapshot = dict(phase_keyframes)
            keyframes_result = await _timed_pro_stage(
                "extract_keyframes_smart",
                loop.run_in_executor(
                    _executor,
                    extract_keyframes_smart, tmp_path, poses, swing_phases, phase_keyframes, 320,
                ),
                _PRO_KEYFRAME_SMART_TIMEOUT_S,
            )
            if isinstance(keyframes_result, tuple):
                keyframes, kf_validation = keyframes_result
            else:
                keyframes = keyframes_result
                kf_validation = {"total_keyframes": len(keyframes), "near_duplicates": 0, "time_too_close": 0, "all_passed": True, "details": []}

            keyframes, kf_validation, phase_keyframes, _final_kf_src = await _timed_pro_stage(
                "ensure_keyframes_ordered_for_ai",
                loop.run_in_executor(
                    _executor,
                    ensure_keyframes_ordered_for_ai,
                    tmp_path,
                    poses,
                    swing_phases,
                    phase_keyframes_snapshot,
                    keyframes,
                    kf_validation,
                    phase_keyframes,
                    320,
                ),
                _PRO_ENSURE_ORDERED_TIMEOUT_S,
            )
            _log_late_strip_after_ensure(logger, keyframes, kf_validation)
            if bool(kf_validation.get("reselected_top")) or bool(kf_validation.get("reselected_impact")):
                phase_validation = validate_phase_keyframes(
                    phase_keyframes, poses, source=f"{phase_source}_reselected",
                )

            t_kf = time.time()
            logger.info("[PRO] smart_keyframes pipeline: %.2fs (%d phases)", t_kf - t1, len(keyframes))

            def _strip_vision_bundle():
                return build_ai_vision_images_from_phase_keyframes(
                    tmp_path, poses, keyframes, phase_keyframes,
                )

            ai_frames = await _timed_pro_stage(
                "build_ai_vision_images_from_phase_keyframes",
                loop.run_in_executor(_executor, _strip_vision_bundle),
                _PRO_AI_VISION_STRIP_TIMEOUT_S,
            )

            sem_report = await _timed_pro_stage(
                "build_semantic_phase_report",
                loop.run_in_executor(
                    _executor,
                    partial(
                        build_semantic_phase_report,
                        poses,
                        dict(phase_keyframes),
                        phase_validation,
                        list(keyframes),
                        dict(kf_validation.get("final_keyframe_validation") or {}),
                    ),
                ),
                _PRO_SEMANTIC_REPORT_TIMEOUT_S,
            )
        else:
            from services.pro_keyframe_orchestrator_service import run_pro_motion_keyframe_chain
            from services.swing_flow_utils import validate_phase_keyframes

            work_dir = tempfile.mkdtemp(prefix="pro240_")
            try:
                motion = await run_pro_motion_keyframe_chain(
                    tmp_path, work_dir=work_dir, keyframe_width=320, region=region
                )
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
            except Exception as e:
                logger.exception("[PRO] motion chain failed")
                raise HTTPException(status_code=503, detail=f"Pro motion pipeline failed: {e}")
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

            poses = motion.poses
            pose_quality_bundle = motion.pose_quality_bundle
            swing_phases = motion.swing_phases
            phase_keyframes = motion.phase_keyframes
            keyframes = motion.keyframes
            kf_validation = motion.kf_validation
            ff_meta_for_response = motion.ffmpeg_meta

            t1 = time.time()
            logger.info(
                "[PRO][motion] wall_since_start=%.1fs poses=%d keyframes=%s analysis_fps=%s",
                t1 - t0,
                len(poses),
                len(keyframes),
                ff_meta_for_response.get("fps"),
            )

            ai_frames = [str(k.get("image_base64") or "") for k in keyframes]
            if len([x for x in ai_frames if x]) < 8:
                raise HTTPException(status_code=422, detail="Incomplete Pro keyframe strip")

            phase_validation = validate_phase_keyframes(dict(phase_keyframes), poses, source="motion")
            phase_debug = {}
            phase_source = "motion_chain"
            sem_report = {
                "final_phase_semantic_ok_strict": True,
                "final_phase_semantic_ok_strict_reasons": [],
                "keyframe_semantic_ok": True,
                "phase_validation_passed": bool(phase_validation.get("passed", True)),
                "align_top": True,
                "align_impact": True,
                "phase_detector_version": "motion_chain_v1",
                "phase_reselection_failed": False,
                "final_phase_semantic_ok": True,
            }
            t_kf = time.time()

        kf_src = str(kf_validation.get("final_keyframe_source") or "")
        gate_pass_kf = bool(kf_validation.get("final_keyframe_gate_pass", False))
        sem_ok = bool(sem_report.get("final_phase_semantic_ok"))
        sem_ok_strict = bool(sem_report.get("final_phase_semantic_ok_strict"))
        pv_pass = bool(sem_report.get("phase_validation_passed"))
        phase_validation_for_cap = {**(phase_validation or {}), "passed": pv_pass}
        final_pf = dict(kf_validation.get("final_phase_keyframes") or phase_keyframes)
        if use_legacy:
            gemini_assess = assess_gemini_uniform_map_vs_final_phase_strip(
                phase_source,
                phase_debug.get("gemini_mapped"),
                final_pf,
            )
        else:
            gemini_assess = {
                "gemini_uniform_thumbnail_map_applies": False,
                "gemini_map_aligned_with_final_strip": True,
            }
        g_applies = bool(gemini_assess.get("gemini_uniform_thumbnail_map_applies"))
        g_aligned = gemini_assess.get("gemini_map_aligned_with_final_strip")

        hand_info_gate = detect_handedness(poses, swing_phases=swing_phases)
        hand_for_sweet = str(hand_info_gate.get("hand") or "UNKNOWN")
        imp_for_sweet = int(phase_keyframes.get("impact", len(poses) // 2))
        sweet_spot_bundle = estimate_sweet_spot_robust(poses, imp_for_sweet, hand=hand_for_sweet)
        strict_decision = should_run_phase_analysis_strict(
            pose_quality_bundle=pose_quality_bundle,
            sem_report=sem_report,
            kf_validation=kf_validation,
            keyframe_count=len(keyframes),
            ai_vision_count=len(ai_frames),
            gemini_assess=gemini_assess,
            sweet_spot_bundle=sweet_spot_bundle,
        )
        if not strict_decision["pass"]:
            _log_late_strip_final_for_422(logger, keyframes, kf_validation)
            raise HTTPException(
                status_code=422,
                detail=build_phase_alignment_fail_detail(strict_decision),
            )
        phase_evaluations_reliable = compute_phase_evaluations_reliable(
            final_phase_semantic_ok=sem_ok_strict,
            phase_validation_passed=pv_pass,
            final_keyframe_source=kf_src,
            final_keyframe_gate_pass=gate_pass_kf,
            ai_vision_frame_count=len(ai_frames),
            keyframe_strip_frame_count=len(keyframes),
            gemini_uniform_map_applies=g_applies,
            gemini_map_aligned_with_final_strip=g_aligned,
        )
        phase_images_reliable = phase_evaluations_reliable
        phase_boundary = build_phase_boundary_flags(
            final_keyframe_source=kf_src,
            keyframe_strip_frame_count=len(keyframes),
            ai_vision_frame_count=len(ai_frames),
            gemini_strip_assessment=gemini_assess,
            analysis_route="pro",
            plus_grade_phase_evaluations=False,
        )
        phase_evaluations_warning = ""
        if not gate_pass_kf or kf_src == "ordered_fallback_empty":
            phase_source = "kinematic_degraded"
            phase_evaluations_warning = "keyframe_gate_failed_or_empty_fallback"
            logger.warning("[PRO] Final keyframe gate failed (%s)", kf_src)
        elif kf_src == "ordered_fallback":
            phase_evaluations_warning = "ordered_fallback_semantics_untrusted"
        elif not sem_ok:
            phase_evaluations_warning = "semantic_validation_failed"
        elif not pv_pass:
            phase_evaluations_warning = "phase_validation_soft_fail"
        if g_applies and g_aligned is False:
            phase_evaluations_warning = (phase_evaluations_warning + ";gemini_uniform_map_diverged_from_strip").strip(";")
            logger.warning("[PRO] Gemini map diverged from strip: %s", gemini_assess)
        if len(ai_frames) < 8 or len(keyframes) < 8:
            phase_evaluations_warning = (phase_evaluations_warning + ";incomplete_phase_vision_strip").strip(";")
            logger.warning("[PRO] Incomplete strip kf=%s ai=%s", len(keyframes), len(ai_frames))

        mid_idx = len(poses) // 2
        representative_pose = poses[mid_idx]

        ai_result = await _timed_pro_stage(
            "analyze_swing_pro",
            analyze_swing_pro(
                pose_data={
                    "angles": representative_pose["angles"],
                    "all_frame_angles": [p["angles"] for p in poses],
                    "frame_count": len(poses),
                },
                keyframe_images=ai_frames,
                region=region,
                phase_images_reliable=phase_images_reliable,
            ),
            PRO_AI_TIMEOUT_S + 60.0,
        )

        t2 = time.time()
        logger.info("[PRO] post_ai_setup: %.2fs since keyframe_strip", t2 - t_kf)

        for i, pose in enumerate(poses):
            if i < len(swing_phases):
                pose["phase_data"] = swing_phases[i]

        hand_info = detect_handedness(poses, swing_phases=swing_phases)
        resolved_hand = str(hand_info.get("hand") or "UNKNOWN")

        hud_frames = []
        for pose in poses:
            hud = generate_hud_data(pose_for_skeleton_render(pose), mode="pro", hand=resolved_hand)
            hud["frame_index"] = pose["frame_index"]
            hud["timestamp"] = pose["timestamp"]
            if pose.get("phase_data"):
                hud["phase"] = pose["phase_data"]
            hud_frames.append(hud)

        detected_club = ai_result.get("detected_club") or {}
        club_type = detected_club.get("club_type") if isinstance(detected_club, dict) else None
        club_group = detected_club.get("club_group") if isinstance(detected_club, dict) else None

        all_angles = [p["angles"] for p in poses if p.get("angles")]
        swing_dur = poses[-1]["timestamp"] - poses[0]["timestamp"] if len(poses) >= 2 else 1.2
        prediction = predict_shot(
            representative_pose,
            swing_duration=swing_dur,
            all_frame_angles=all_angles,
            club_type=club_type,
            club_group=club_group,
            hand=resolved_hand,
            hand_confidence=float(hand_info.get("confidence") or 0.0),
            poses=poses,
            impact_pose_idx=int(phase_keyframes.get("impact", len(poses) // 2)),
        )
        trajectory_data = compute_wrist_trajectory(poses)

        tracking_quality = 1.0
        if len(poses) < 20:
            tracking_quality = 0.3
        elif len(poses) < 40:
            tracking_quality = 0.6
        analysis_reliability = cap_confidence(
            ai_result,
            phase_validation=phase_validation_for_cap,
            hand=resolved_hand,
            club_type=club_type,
            tracking_quality=tracking_quality,
            phase_vision_reliable=phase_evaluations_reliable,
            sweet_spot_unstable=bool(sweet_spot_bundle.get("sweet_spot_unstable")),
            sweet_spot_confidence=sweet_spot_bundle.get("sweet_spot_confidence"),
        )
        if prediction.get("club_assumed") or prediction.get("hand_assumed"):
            analysis_reliability = cap_confidence(
                ai_result,
                phase_validation=phase_validation_for_cap,
                hand=resolved_hand,
                club_type=club_type,
                tracking_quality=tracking_quality,
                club_assumed=bool(prediction.get("club_assumed")),
                phase_vision_reliable=phase_evaluations_reliable,
                sweet_spot_unstable=bool(sweet_spot_bundle.get("sweet_spot_unstable")),
                sweet_spot_confidence=sweet_spot_bundle.get("sweet_spot_confidence"),
            )
        if isinstance(kf_validation, dict) and (
            kf_validation.get("near_duplicates", 0) > 2 or not kf_validation.get("all_passed", True)
        ):
            ar = dict(analysis_reliability)
            reasons = list(ar.get("reasons", []))
            cc = int(ar.get("capped_confidence", 50))
            reasons.append("keyframe_quality_low")
            ar["reasons"] = reasons
            ar["capped_confidence"] = max(20, cc - 15)
            ar["level"] = "high" if ar["capped_confidence"] >= 75 else (
                "medium" if ar["capped_confidence"] >= 50 else "low"
            )
            analysis_reliability = ar
        if isinstance(kf_validation, dict) and not bool(kf_validation.get("final_keyframe_gate_pass", True)):
            ar = dict(analysis_reliability)
            reasons = list(ar.get("reasons", []))
            reasons.append("keyframe_order_unsafe_uniform_vision")
            ar["reasons"] = reasons
            cc = int(ar.get("capped_confidence", 50))
            ar["capped_confidence"] = max(15, cc - 25)
            ar["level"] = "high" if ar["capped_confidence"] >= 75 else (
                "medium" if ar["capped_confidence"] >= 50 else "low"
            )
            analysis_reliability = ar

        analysis_id = str(uuid.uuid4())
        fps_val = 30.0
        source_frame_count = 0
        analysis_frame_count = 0
        if ff_meta_for_response:
            fps_val = float(ff_meta_for_response.get("fps") or 240.0)
            source_frame_count = int(ff_meta_for_response.get("source_frame_count") or 0)
            analysis_frame_count = int(ff_meta_for_response.get("analysis_frame_count") or 0)
        else:
            try:
                import cv2 as _cv2
                _cap = _cv2.VideoCapture(tmp_path)
                fps_val = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
                source_frame_count = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                _cap.release()
            except Exception:
                pass
        if source_frame_count < 1 and poses:
            source_frame_count = int(poses[-1].get("frame_index", 0)) + 1

        # ── Club detection + speed fusion (best-effort, non-blocking) ──
        fusion_data: dict = {}
        try:
            detect_club, detect_blur_speed, track_trajectory_fn, fuse_speed = _load_fusion_services()

            impact_frames, _impact_idx, impact_keyframe = _extract_impact_frames(
                tmp_path, poses, swing_phases, window=3,
            )

            club_frame = impact_keyframe if impact_keyframe is not None else (
                impact_frames[len(impact_frames) // 2] if impact_frames else None
            )

            if club_frame is not None:
                try:
                    club_info = await asyncio.wait_for(
                        detect_club(club_frame, region),
                        timeout=_PRO_FUSION_DETECT_CLUB_S,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[PRO] fusion stage=detect_club timeout after %.0fs",
                        _PRO_FUSION_DETECT_CLUB_S,
                    )
                    club_info = {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0}
            else:
                club_info = {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0}

            def _blur_job():
                return (
                    detect_blur_speed(impact_frames, fps=fps_val)
                    if impact_frames
                    else {
                        "ball_speed": 0.0,
                        "streak_length_px": 0,
                        "confidence": "low",
                    }
                )

            try:
                blur_result = await asyncio.wait_for(
                    loop.run_in_executor(_executor, _blur_job),
                    timeout=_PRO_FUSION_BLUR_S,
                )
            except asyncio.TimeoutError:
                logger.warning("[PRO] fusion stage=detect_blur_speed timeout")
                blur_result = {
                    "ball_speed": 0.0,
                    "streak_length_px": 0,
                    "confidence": "low",
                }

            def _traj_job():
                return (
                    track_trajectory_fn(impact_frames, fps=fps_val)
                    if impact_frames
                    else {
                        "ball_speed": 0.0,
                        "tracked_frames": 0,
                        "confidence": "low",
                    }
                )

            try:
                traj_result = await asyncio.wait_for(
                    loop.run_in_executor(_executor, _traj_job),
                    timeout=_PRO_FUSION_TRAJECTORY_S,
                )
            except asyncio.TimeoutError:
                logger.warning("[PRO] fusion stage=track_trajectory timeout")
                traj_result = {
                    "ball_speed": 0.0,
                    "tracked_frames": 0,
                    "confidence": "low",
                }

            override_club = None
            if "multipart" in content_type:
                try:
                    override_club = form.get("override_club_type") if form else None
                    if override_club and hasattr(override_club, "read"):
                        override_club = None
                except Exception:
                    pass
            else:
                try:
                    body_raw = await request.body()
                    import json as _json
                    body_data = _json.loads(body_raw)
                    override_club = body_data.get("override_club_type")
                except Exception:
                    pass

            formula_speed = float(prediction.get("ball_speed", 0.0))
            fusion_result = fuse_speed(
                club_group=club_info["club_group"],
                blur_result=blur_result,
                trajectory_result=traj_result,
                formula_speed=formula_speed,
                override_club_type=override_club,
            )

            fusion_data = {
                "club_type": club_info["club_type"],
                "club_group": club_info["club_group"],
                "club_detection_confidence": club_info["confidence"],
                "blur_speed": blur_result["ball_speed"],
                "blur_confidence": blur_result.get("confidence", "low"),
                "trajectory_speed": traj_result["ball_speed"],
                "trajectory_confidence": traj_result.get("confidence", "low"),
                "trajectory_tracked_frames": traj_result.get("tracked_frames", 0),
                "fused_speed": fusion_result["fused_speed"],
                "fusion_weights": fusion_result["fusion_weights"],
                "speed_confidence": fusion_result["speed_confidence"],
                "error_estimate_pct": fusion_result["error_estimate_pct"],
            }
            logger.info("[PRO] Fusion complete: club=%s fused=%.1f mph", club_info["club_type"], fusion_result["fused_speed"])

            # Recalibrate carry distance using fused speed + detected/overridden club.
            effective_club_type = (override_club or club_info.get("club_type") or "").upper().strip() or None
            effective_club_group = club_info.get("club_group")
            try:
                prediction = await asyncio.wait_for(
                    loop.run_in_executor(
                        _executor,
                        partial(
                            calibrate_prediction,
                            prediction,
                            club_type=effective_club_type,
                            club_group=effective_club_group,
                            preferred_ball_speed=float(fusion_result["fused_speed"]),
                        ),
                    ),
                    timeout=_PRO_FUSION_CALIBRATE_S,
                )
            except asyncio.TimeoutError:
                logger.warning("[PRO] fusion stage=calibrate_prediction timeout")
        except Exception as e:
            logger.warning("[PRO] Speed fusion failed (non-fatal): %s", e)

        if fusion_data:
            prediction.update(fusion_data)

        logger.info(f"[PRO] Total analysis: {time.time() - t0:.1f}s")

        _pro_gate_tr = strict_decision.get("gate_decision_trace") or {}
        pro_response = {
            "analysis_id": analysis_id,
            "type": "pro",
            "status": "ok",
            "contact_sheet_url": None,
            "video_url": None,
            "analysis_mode": "pose_pro",
            "phase_pipeline_applied": True,
            "ai_provider": ai_result.get("ai_provider", "unknown"),
            "ai_key": ai_result.get("ai_key"),
            "scores": ai_result.get("scores", {}),
            "total_score": ai_result.get("total_score", 0),
            "issues": ai_result.get("issues", []),
            "issues_zh": ai_result.get("issues_zh", []),
            "suggestions": ai_result.get("suggestions", []),
            "suggestions_zh": ai_result.get("suggestions_zh", []),
            "summary": ai_result.get("summary", ""),
            "summary_zh": ai_result.get("summary_zh", ""),
            "advanced_metrics": ai_result.get("advanced_metrics", {}),
            "training_plan": ai_result.get("training_plan", {}),
            "keyframes": [
                {
                    "phase": kf["phase"],
                    "label_en": kf["label_en"],
                    "label_zh": kf["label_zh"],
                    "frame_index": kf.get("frame_index"),
                    "source_pose_idx": kf.get("source_pose_idx"),
                    "source_frame_index": kf.get("source_frame_index", kf.get("frame_index")),
                    "timestamp": kf["timestamp"],
                    "confidence": kf.get("confidence"),
                    "selection_reason": kf.get("selection_reason"),
                    "fallback_used": kf.get("fallback_used", False),
                    "image_base64": kf["image_base64"],
                    "pose_snapshot": kf.get("pose_snapshot"),
                }
                for kf in keyframes
            ],
            "skeleton_data": {
                "frames": hud_frames,
                "total_frames": len(hud_frames),
                "joint_space": "analysis_frame",
                "joint_sources": {
                    "raw_detection_joints": "pose.raw_detection_joints",
                    "analysis_joints": "pose.joints",
                    "render_joints": "pose.render_joints->pose.joints",
                },
            },
            "pose_frames": [
                {k: v for k, v in p.items() if k != "image_base64"}
                for p in poses
            ],
            "prediction": prediction,
            "trajectory": trajectory_data,
            "swing_phases": swing_phases,
            "phase_keyframes": phase_keyframes,
            "phase_source": phase_source,
            "phase_validation": phase_validation_for_cap,
            "phase_debug": phase_debug,
            "keyframe_validation": kf_validation,
            "final_phase_keyframes": kf_validation.get("final_phase_keyframes"),
            "final_keyframe_validation": kf_validation.get("final_keyframe_validation"),
            "final_keyframe_order_ok": kf_validation.get("final_keyframe_order_ok"),
            "final_keyframe_time_order_ok": kf_validation.get("final_keyframe_time_order_ok"),
            "final_keyframe_source": kf_validation.get("final_keyframe_source"),
            "final_keyframe_gate_pass": kf_validation.get("final_keyframe_gate_pass"),
            "phase_detector_version": sem_report.get("phase_detector_version"),
            "phase_detector_confidence": sem_report.get("phase_detector_confidence"),
            "top_candidate_debug": sem_report.get("top_candidate_debug"),
            "impact_candidate_debug": sem_report.get("impact_candidate_debug"),
            "top_keyframe_vs_event": sem_report.get("top_keyframe_vs_event"),
            "impact_keyframe_vs_event": sem_report.get("impact_keyframe_vs_event"),
            "top_semantic_at_keyframe": sem_report.get("top_semantic_at_keyframe"),
            "impact_semantic_at_keyframe": sem_report.get("impact_semantic_at_keyframe"),
            "top_semantic_ok": sem_report.get("top_semantic_ok"),
            "impact_semantic_ok": sem_report.get("impact_semantic_ok"),
            "semantic_validation": sem_report.get("semantic_validation"),
            "final_phase_semantic_ok": sem_report.get("final_phase_semantic_ok"),
            "phase_evaluations_reliable": phase_evaluations_reliable,
            "phase_evaluations_warning": phase_evaluations_warning or None,
            "phase_boundary": phase_boundary,
            "analysis_reliability": analysis_reliability,
            "sweet_spot_warning": bool(_pro_gate_tr.get("sweet_spot_warning")),
            "sweet_spot_confidence": sweet_spot_bundle.get("sweet_spot_confidence"),
            "video_meta": {
                "fps": fps_val,
                "total_pose_frames": len(poses),
                "duration_s": round(poses[-1]["timestamp"] if poses else 0, 3),
                "source_frame_count": source_frame_count,
                **({"analysis_frame_count": analysis_frame_count} if analysis_frame_count else {}),
            },
            "segmentation_available": False,
            "world_3d_available": False,
        }
        rel_lvl = analysis_reliability.get("level", "medium") if isinstance(analysis_reliability, dict) else "medium"
        if rel_lvl == "low":
            pro_response["quality_warning"] = (
                "分析可信度较低，结果仅供参考 / Analysis reliability is low, results are for reference only"
            )
        if phase_source == "kinematic_degraded" and isinstance(kf_validation, dict) and kf_validation.get(
            "near_duplicates", 0
        ) > 2:
            pro_response["keyframe_warning"] = (
                "关键帧质量不佳，部分阶段图像可能相似 / Keyframe quality is poor, some phase images may look similar"
            )
        if prediction.get("hand_assumed"):
            pro_response["hand_assumed"] = prediction["hand_assumed"]
            pro_response["hand_warning"] = prediction.get("hand_warning", "")
        if prediction.get("club_assumed"):
            pro_response["club_assumed"] = prediction["club_assumed"]
            pro_response["club_warning"] = prediction.get("club_warning", "")
        if not use_legacy:
            pro_response = _motion_pro_strip_internal(pro_response)
        log_non_finite_if_any(logger, pro_response, "pro_analyze")
        return sanitize_json_floats(pro_response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pro analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
