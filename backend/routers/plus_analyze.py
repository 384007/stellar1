import asyncio
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Any, Optional, Tuple

from routers.auth import get_current_user
from services.json_sanitize import log_non_finite_if_any, sanitize_json_floats
from services.video_upload_suffix import temp_suffix_for_uploaded_video, temp_suffix_from_url

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


def _stellar_modal_upload_echo(route: str, request: Request) -> None:
    """Ingress trace for upload routes.

    Always emit one line so we can confirm where the request landed, even when
    hostname/env heuristics are inconclusive (e.g. custom domain on Modal).
    """
    host = (request.headers.get("host") or "").lower()
    modal_host = ".modal.run" in host
    modal_env = bool(os.getenv("MODAL_REGION")) or (os.getenv("STELLAR_RUNTIME") or "").lower() == "modal"
    runtime = (os.getenv("STELLAR_RUNTIME") or "").lower() or "unknown"
    msg = (
        f"[stellar-ingress] route={route} method={request.method} path={request.url.path} "
        f"host={host!r} runtime={runtime} modal_host={int(modal_host)} modal_env={int(modal_env)}"
    )
    logger.info("%s", msg)

router = APIRouter()

# Wall-clock cap for the single Prov3+Plus pipeline (pose-on-source + true240 + A/B + bridge).
_PLUS_PROV3_PIPELINE_OUTER_S = float(
    os.getenv("STELLAR_PLUS_PROV3_PIPELINE_OUTER_S", os.getenv("STELLAR_PLUS_PROV3_OUTER_S", "600"))
)


def _plus_keyframe_pack_block_reasons(
    kf_validation: dict,
    strict_fail_rs: set,
) -> Tuple[Optional[str], Optional[str]]:
    """Human/log reasons when formal score + AI report must not be packaged."""
    gate_ok = bool(kf_validation.get("final_keyframe_gate_pass", False))
    src = str(kf_validation.get("final_keyframe_source") or "")
    dup_tc = bool(strict_fail_rs & {"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"})
    if gate_ok and src != "smart_gate_failed" and not dup_tc:
        return None, None
    parts: list[str] = []
    if not gate_ok:
        parts.append("final_keyframe_gate_pass_false")
    if src == "smart_gate_failed":
        parts.append("final_keyframe_source_smart_gate_failed")
    if dup_tc:
        parts.append("strict_NEAR_DUPLICATE_OR_TIME_TOO_CLOSE")
    sem_fail = kf_validation.get("final_phase_semantic_fail_reasons")
    if isinstance(sem_fail, list) and sem_fail:
        parts.append("phase_semantic:" + ",".join(str(x) for x in sem_fail[:6]))
    if not parts:
        parts.append("keyframe_unreliable")
    joined = ";".join(parts)
    return joined, joined


def _merge_gemini_observation_into_withheld_ai_result(
    ai_result: dict,
    obs: dict,
    *,
    withhold: bool,
) -> None:
    """Copy visual-observation text into top-level report fields when formal scores are withheld.

    The observation call already ran for degraded strips; the null bundle cleared summaries, so
    clients that only read ``summary_zh`` / ``issues_zh`` showed an empty \"report\".
    """
    if not withhold or not isinstance(ai_result, dict) or not isinstance(obs, dict):
        return
    if not bool(obs.get("available")):
        return
    sz = str(obs.get("summary_zh") or "").strip()
    se = str(obs.get("summary_en") or "").strip()
    if sz:
        ai_result["summary_zh"] = sz
    if se or sz:
        ai_result["summary"] = se or sz
    bz = obs.get("bullets_zh") or []
    be = obs.get("bullets_en") or []
    if bz and not (ai_result.get("issues_zh") or []):
        ai_result["issues_zh"] = [str(x) for x in bz[:16] if str(x).strip()]
    if be and not (ai_result.get("issues") or []):
        ai_result["issues"] = [str(x) for x in be[:16] if str(x).strip()]
    pd = dict(ai_result.get("primary_diagnosis") or {})
    if sz or se:
        pd.setdefault("title_zh", "视觉观察摘要")
        pd.setdefault("title_en", "Visual observation")
        pd.setdefault("status_zh", "仅供参考（不设正式评分）")
        pd.setdefault("status_en", "Reference only (no formal score)")
        ai_result["primary_diagnosis"] = pd
    tip_zh = str(ai_result.get("quick_tip_zh") or "").strip()
    extra_zh = "关键帧未通过严格校验，以下为视觉观察摘要，不设正式评分。"
    if extra_zh not in tip_zh:
        ai_result["quick_tip_zh"] = f"{tip_zh} {extra_zh}".strip() if tip_zh else extra_zh
    tip_en = str(ai_result.get("quick_tip_en") or "").strip()
    extra_en = "Keyframes did not pass strict validation; summary below is observation-only (no formal score)."
    if extra_en not in tip_en:
        ai_result["quick_tip_en"] = f"{tip_en} {extra_en}".strip() if tip_en else extra_en


def _plus_null_formal_score_bundle() -> dict:
    """No numeric scores — JSON nulls for totals; UI must show withheld messaging."""
    return {
        "ai_provider": "withheld_unreliable_keyframes",
        "posture_score": None,
        "primary_diagnosis": {
            "title_zh": "暂不可评分",
            "title_en": "Score unavailable",
            "status_zh": "关键帧不可靠",
            "status_en": "Unreliable keyframes",
            "ai_confidence": 0,
        },
        "additional_issues": [],
        "quick_tip_zh": "关键帧未通过严格校验，本次不提供正式评分。",
        "quick_tip_en": "Keyframes failed strict validation; no formal score for this run.",
        "problem_description_zh": "",
        "problem_description_en": "",
        "swing_phase_evaluations": [],
        "training": {},
        "recommended_videos": [],
        "scores": None,
        "total_score": None,
        "issues": [],
        "issues_zh": [],
        "suggestions": [],
        "suggestions_zh": [],
        "summary": "",
        "summary_zh": "",
        "advanced_metrics": {},
    }


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on", "y"}


async def _timed_plus_stage(name: str, awaitable, timeout_s: float):
    t0 = time.perf_counter()
    try:
        r = await asyncio.wait_for(awaitable, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.error("[PLUS] stage=%s TIMEOUT after %.0fs", name, timeout_s)
        raise HTTPException(
            status_code=504,
            detail=(
                f"分析步骤超时: {name}（>{timeout_s:.0f}s）/ Analysis step timed out: {name}"
            ),
        ) from None
    logger.info("[PLUS] stage=%s wall_s=%.2f", name, time.perf_counter() - t0)
    return r


def _load_services():
    from services.gemini_service import analyze_swing_plus
    from services.hud_service import generate_hud_data
    from services.shot_predictor import predict_shot
    return analyze_swing_plus, generate_hud_data, predict_shot


# ── Posture Practice Video Generation (Veo) ──

@router.post("/plus/posture-video")
async def generate_posture_practice_video(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Generate a single 8-second posture teaching video via Gemini Veo."""
    try:
        from services.veo_service import generate_posture_video, get_posture_templates
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Video generation service unavailable: {e}")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    video_id = body.get("video_id", "").strip()
    analysis_data = body.get("analysis_data")

    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required")
    if not analysis_data or not isinstance(analysis_data, dict):
        raise HTTPException(status_code=400, detail="analysis_data is required")

    valid_ids = [t["id"] for t in get_posture_templates()]
    if video_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video_id '{video_id}'. Valid: {valid_ids}",
        )

    logger.info("[posture-video] Generating video_id=%s user=%s", video_id, current_user.get("email") if current_user else "anon")

    try:
        result = await generate_posture_video(video_id, analysis_data)
        return result
    except RuntimeError as e:
        logger.error("[posture-video] Generation failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("[posture-video] Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail=f"Video generation error: {str(e)}")


@router.post("/plus")
async def analyze_plus(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
):
    _stellar_modal_upload_echo("PLUS", request)
    try:
        analyze_plus_fn, generate_hud, predict_shot = _load_services()
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
            suffix = temp_suffix_for_uploaded_video(filename)

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
        else:
            body = await request.json()
            video_url = body.get("video_url", "")
            if not video_url:
                raise HTTPException(status_code=400, detail="No video_url provided")
            if video_url.startswith("blob:"):
                raise HTTPException(status_code=400, detail="Blob URLs cannot be fetched by the server.")

            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.get(video_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to download video")

            suffix = temp_suffix_from_url(video_url)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

        t0 = time.time()
        plus_degraded_flags: list[str] = []
        logger.info(
            "[STELLAR_PLUS_PIPELINE] code_marker=plus_degradation_v1 git_sha=%s runtime=%s",
            (os.getenv("STELLAR_GIT_SHA") or "unknown")[:12],
            (os.getenv("STELLAR_RUNTIME") or "unknown"),
        )

        from services.swing_flow_utils import (
            detect_swing_phases,
            compute_wrist_trajectory,
            compute_phase_evaluations_reliable,
            assess_gemini_uniform_map_vs_final_phase_strip,
            build_phase_boundary_flags,
        )
        from services.biomech_validation_service import validate_phase_chain_hard
        from services.internal.prov3_ffmpeg import FFmpegNotFoundError
        from services.plus_prov3_analysis_bridge import run_plus_prov3_keyframe_bridge
        from services.api_pack_service import pack_plus_response
        from services.custom_landmark_training_service import run_research_refine
        from services.handedness_service import detect_handedness
        from services.gemini_service import (
            analyze_plus_visual_observation,
            cap_confidence,
            PLUS_AI_TIMEOUT_S,
        )
        from services.pose_service import pose_for_skeleton_render
        from services.phase_analysis_gate import (
            collect_plus_route_hard_reasons,
            should_run_phase_analysis_strict,
        )

        loop = asyncio.get_event_loop()

        disconnected = threading.Event()

        async def _watch_plus_client_disconnect() -> None:
            try:
                while not disconnected.is_set():
                    if await request.is_disconnected():
                        disconnected.set()
                        return
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                disconnected.set()
                raise

        def _plus_prov3_cancel_check() -> None:
            if disconnected.is_set():
                raise RuntimeError("plus_client_disconnected")

        det_bundle = {
            "enabled": False,
            "yolo11_degraded": False,
            "status": "skipped_plus_prov3_only",
            "detections": [],
            "provider_meta": {},
        }
        tracks = {
            "status": "skipped_plus_prov3_only",
            "person_tracks": [],
            "club_tracks": [],
            "ball_tracks": [],
            "provider_meta": {},
        }
        motion3d_bundle = {
            "enabled": False,
            "motion_3d": [],
            "provider_meta": {},
            "status": "skipped_plus_prov3_only",
        }
        optional_modules = {
            "detection_active": False,
            "tracking_active": False,
            "yolo11_degraded": False,
            "yolo11_status": "skipped_plus_prov3_only",
        }
        phase_c_prompt_ctx = {
            "phase_c_version": "1",
            "temporal_prior_strength": 0.0,
            "phase_confidence": {},
            "phase_boundary_segment_count": 0,
            "action_backend": {"status": "plus_prov3_only", "name": "none"},
        }
        phase_debug: dict = {"plus_single_keyframe_authority": "prov3_bridge"}

        _dc_task = asyncio.create_task(_watch_plus_client_disconnect())
        prov3_work = f"{tmp_path}.prov3_work"
        os.makedirs(prov3_work, exist_ok=True)
        try:
            bridge_out = await _timed_plus_stage(
                "plus_prov3_pipeline",
                loop.run_in_executor(
                    _executor,
                    partial(
                        run_plus_prov3_keyframe_bridge,
                        tmp_path,
                        prov3_work,
                        screen_mode=False,
                        cancel_check=_plus_prov3_cancel_check,
                    ),
                ),
                _PLUS_PROV3_PIPELINE_OUTER_S,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "plus_client_disconnected" in msg or "disconnected" in msg.lower():
                raise HTTPException(status_code=503, detail=msg) from exc
            if "no_poses_detected" in msg:
                raise HTTPException(
                    status_code=422,
                    detail="No poses detected. Ensure the golfer is clearly visible.",
                ) from exc
            raise HTTPException(status_code=422, detail=msg) from exc
        except FFmpegNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            shutil.rmtree(prov3_work, ignore_errors=True)
            _dc_task.cancel()
            try:
                await _dc_task
            except asyncio.CancelledError:
                pass

        poses = list(bridge_out.get("poses") or [])
        pose_quality_bundle = dict(bridge_out.get("pose_quality_bundle") or {})
        pose_stream_meta = dict(bridge_out.get("pose_stream_meta") or {})
        if not poses:
            raise HTTPException(
                status_code=422,
                detail="No poses detected. Ensure the golfer is clearly visible.",
            )
        t1 = time.time()
        logger.info(
            "[PLUS] plus_prov3_pipeline wall_since_start=%.1fs poses=%d",
            t1 - t0,
            len(poses),
        )
        logger.info(
            "[ROLE=POSE_BACKEND] active=%s requested=%s",
            str((pose_stream_meta.get("provider_meta") or {}).get("active_backend") or "mediapipe"),
            str((pose_stream_meta.get("provider_meta") or {}).get("requested_backend") or "mediapipe"),
        )
        swing_phases = detect_swing_phases(poses)

        keyframes = list(bridge_out["plus_keyframes"])
        kf_validation = dict(bridge_out["kf_validation"])
        phase_keyframes = dict(bridge_out["phase_keyframes_pose"])
        sem_report = dict(bridge_out["sem_report"])
        phase_validation = dict(bridge_out["phase_validation_soft"])
        phase_source = str(bridge_out["phase_source"])
        ai_frames = list(bridge_out["ai_vision_base64_list"])
        pv_pass = bool(sem_report.get("phase_validation_passed"))
        phase_validation_for_cap = {**phase_validation, "passed": pv_pass}

        prov3_pub = dict(bridge_out.get("prov3") or {})
        phase_debug["plus_prov3_bridge"] = prov3_pub
        phase_debug["authoritative_phase_chain"] = {
            "ok": bool(kf_validation.get("final_keyframe_gate_pass")),
            "reasons": [],
            "phase_keyframes": dict(phase_keyframes),
            "source": "prov3_true240",
        }
        phase_debug["keyframe_authoritative_handoff"] = {
            "source": "prov3_true240",
            "analysis_id": prov3_pub.get("analysis_id"),
            "low_trust_preview_only": prov3_pub.get("low_trust_preview_only"),
        }
        phase_debug["prov3_motion"] = dict(bridge_out.get("_prov3_motion") or {})

        hard_biomech = validate_phase_chain_hard(poses, phase_keyframes)
        biomech_hard_passed = bool(hard_biomech.get("passed"))
        if not biomech_hard_passed:
            plus_degraded_flags.append("BIOMECH_HARD_FAIL")
            logger.warning(
                "[STELLAR_PLUS_PIPELINE] BIOMECH hard gate failed reasons=%s — continuing 200 degraded (not_422)",
                list(hard_biomech.get("reasons") or []),
            )
        logger.info("[ROLE=BIOMECH] passed=%s reasons=%s", biomech_hard_passed, list(hard_biomech.get("reasons") or []))

        t_kf = time.time()
        logger.info(
            "[PLUS] prov3_keyframe_bridge wall_since_pose=%.2fs keyframes=%d gate_pass=%s",
            t_kf - t1,
            len(keyframes),
            bool(kf_validation.get("final_keyframe_gate_pass")),
        )

        kf_src = str(kf_validation.get("final_keyframe_source") or "")
        gate_pass_kf = bool(kf_validation.get("final_keyframe_gate_pass", False))
        sem_ok = bool(sem_report.get("final_phase_semantic_ok"))
        final_pf = dict(kf_validation.get("final_phase_keyframes") or phase_keyframes)
        gemini_assess = assess_gemini_uniform_map_vs_final_phase_strip(
            phase_source,
            phase_debug.get("gemini_mapped"),
            final_pf,
        )
        g_applies = bool(gemini_assess.get("gemini_uniform_thumbnail_map_applies"))
        g_aligned = gemini_assess.get("gemini_map_aligned_with_final_strip")
        from services.shot_predictor import estimate_sweet_spot_robust

        hand_info_gate = detect_handedness(poses, swing_phases=swing_phases)
        hand_for_sweet = str(hand_info_gate.get("hand") or "UNKNOWN")
        imp_for_sweet = int(phase_keyframes.get("impact", len(poses) // 2))
        sweet_spot_bundle = estimate_sweet_spot_robust(poses, imp_for_sweet, hand=hand_for_sweet)
        sem_ok_strict = bool(sem_report.get("final_phase_semantic_ok_strict"))
        strict_decision = should_run_phase_analysis_strict(
            pose_quality_bundle=pose_quality_bundle,
            sem_report=sem_report,
            kf_validation=kf_validation,
            keyframe_count=len(keyframes),
            ai_vision_count=len(ai_frames),
            gemini_assess=gemini_assess,
            sweet_spot_bundle=sweet_spot_bundle,
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
        fv = kf_validation.get("final_keyframe_validation") if isinstance(kf_validation, dict) else {}
        sgap = list((fv or {}).get("source_frame_gap_reasons") or [])
        plus_hard_reasons = collect_plus_route_hard_reasons(
            gate_pass_kf=gate_pass_kf,
            phase_source=str(phase_source),
            pv_pass=pv_pass,
            sem_ok=sem_ok,
            phase_evaluations_reliable=phase_evaluations_reliable,
            source_frame_gap_reasons=sgap,
        )
        phase_repair_failed = False
        partial_mode = False
        if not strict_decision.get("pass") or plus_hard_reasons:
            merged_reasons = list(strict_decision.get("reasons") or [])
            for r in plus_hard_reasons:
                if r not in merged_reasons:
                    merged_reasons.append(r)
            logger.warning(
                "[PLUS] strict gate failed after repair rounds reasons=%s — returning partial mode",
                merged_reasons,
            )
            phase_repair_failed = True
            partial_mode = True
            phase_evaluations_reliable = False
        _fv_for_score = kf_validation.get("final_keyframe_validation") if isinstance(kf_validation, dict) else {}
        _strict_fail_rs = set((_fv_for_score or {}).get("strict_contract_fail_reasons") or [])
        _withhold_formal_score = (
            not bool(kf_validation.get("final_keyframe_gate_pass", False))
            or str(kf_validation.get("final_keyframe_source") or "") == "smart_gate_failed"
            or bool(_strict_fail_rs & {"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"})
        )
        if _withhold_formal_score:
            partial_mode = True
            phase_repair_failed = True
            phase_evaluations_reliable = False
        phase_images_reliable = phase_evaluations_reliable
        phase_boundary = build_phase_boundary_flags(
            final_keyframe_source=kf_src,
            keyframe_strip_frame_count=len(keyframes),
            ai_vision_frame_count=len(ai_frames),
            gemini_strip_assessment=gemini_assess,
            analysis_route="plus",
            plus_grade_phase_evaluations=True,
        )
        phase_evaluations_warning = ""
        if not gate_pass_kf or kf_src == "ordered_fallback_empty":
            if str(kf_src).startswith("prov3_"):
                phase_evaluations_warning = "prov3_keyframe_gate_failed_or_low_trust"
            else:
                phase_source = "kinematic_degraded"
                phase_evaluations_warning = "keyframe_gate_failed_or_empty_fallback"
            logger.warning("[PLUS] Final keyframe gate failed (%s); vision uses phase strip only", kf_src)
        elif kf_src == "ordered_fallback":
            phase_evaluations_warning = "ordered_fallback_semantics_untrusted"
            logger.warning("[PLUS] ordered_fallback strip; phase evaluations gated")
        elif not sem_ok:
            phase_evaluations_warning = "semantic_validation_failed"
        elif not pv_pass:
            phase_evaluations_warning = "phase_validation_soft_fail"
        if g_applies and g_aligned is False:
            phase_evaluations_warning = (phase_evaluations_warning + ";gemini_uniform_map_diverged_from_strip").strip(";")
            logger.warning("[PLUS] Gemini 16-thumb map diverged from final 8-strip: %s", gemini_assess)
        if len(ai_frames) < 8 or len(keyframes) < 8:
            phase_evaluations_warning = (phase_evaluations_warning + ";incomplete_phase_vision_strip").strip(";")
            logger.warning(
                "[PLUS] Incomplete strip keyframes=%s ai=%s",
                len(keyframes),
                len(ai_frames),
            )

        mid_idx = len(poses) // 2
        representative_pose = poses[mid_idx]

        region = "CN" if request.headers.get("CF-IPCountry", "").upper() == "CN" else "global"
        _keyframes_snapshot_pre_partial = [dict(k) for k in keyframes]
        _phase_keyframes_snapshot_pre_partial = dict(phase_keyframes)
        degraded_strip = (not gate_pass_kf) or bool(partial_mode)
        if partial_mode:
            if _withhold_formal_score:
                ai_result = _plus_null_formal_score_bundle()
            else:
                ai_result = {
                    "ai_provider": "partial",
                    "posture_score": 0.0,
                    "primary_diagnosis": {
                        "title_zh": "关键帧阶段自动修复失败",
                        "title_en": "Phase keyframe repair failed",
                        "status_zh": "需要注意",
                        "status_en": "Needs attention",
                        "ai_confidence": 0,
                    },
                    "additional_issues": [],
                    "quick_tip_zh": "",
                    "quick_tip_en": "",
                    "problem_description_zh": "",
                    "problem_description_en": "",
                    "swing_phase_evaluations": [
                        {
                            "phase": p,
                            "status": "unknown",
                            "note_zh": "相位图条带降级，禁止严格相位断言。",
                            "note_en": "Degraded strip; strict per-phase assertions are disabled.",
                        }
                        for p in ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]
                    ],
                    "training": {},
                    "recommended_videos": [],
                    "scores": {"grip": 0, "stance": 0, "backswing": 0, "downswing": 0, "follow_through": 0},
                    "total_score": 0,
                    "issues": [],
                    "issues_zh": [],
                    "suggestions": [],
                    "suggestions_zh": [],
                    "summary": "",
                    "summary_zh": "",
                    "advanced_metrics": {},
                }
        else:
            ai_result = await _timed_plus_stage(
                "analyze_swing_plus",
                analyze_plus_fn(
                    pose_data={
                        "angles": representative_pose["angles"],
                        "all_frame_angles": [p["angles"] for p in poses],
                        "frame_count": len(poses),
                    },
                    keyframe_images=ai_frames,
                    region=region,
                    phase_images_reliable=phase_images_reliable,
                    phase_c_context=phase_c_prompt_ctx,
                ),
                PLUS_AI_TIMEOUT_S + 60.0,
            )

        t2 = time.time()
        logger.info("[PLUS] post_ai_setup: %.2fs since keyframe_strip", t2 - t_kf)

        for i, pose in enumerate(poses):
            if i < len(swing_phases):
                pose["phase_data"] = swing_phases[i]

        hand_info = detect_handedness(poses, swing_phases=swing_phases)
        resolved_hand = str(hand_info.get("hand") or "UNKNOWN")

        hud_frames = []
        for pose in poses:
            hud = generate_hud(pose_for_skeleton_render(pose), mode="pro", hand=resolved_hand)
            hud["frame_index"] = pose["frame_index"]
            hud["timestamp"] = pose["timestamp"]
            if pose.get("phase_data"):
                hud["phase"] = pose["phase_data"]
            hud_frames.append(hud)

        from services.club_detector import detect_club_three_frames_from_video

        if tmp_path and os.path.isfile(tmp_path):
            vision_club = await detect_club_three_frames_from_video(tmp_path, region=region)
        else:
            vision_club = {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0}
        ai_dc = ai_result.get("detected_club") if isinstance(ai_result.get("detected_club"), dict) else {}
        vt = str(vision_club.get("club_type") or "UNKNOWN").upper()
        if vt != "UNKNOWN":
            detected_club = {
                "club_type": vt,
                "club_group": str(vision_club.get("club_group") or "IRON"),
                "confidence": float(vision_club.get("confidence") or 0.0),
            }
        else:
            detected_club = ai_dc if ai_dc else {
                "club_type": "UNKNOWN",
                "club_group": "IRON",
                "confidence": 0.0,
            }
        if isinstance(ai_result, dict):
            ai_result["detected_club"] = detected_club
        club_type = detected_club.get("club_type") if isinstance(detected_club, dict) else None
        club_group = detected_club.get("club_group") if isinstance(detected_club, dict) else None

        # ── Confidence capping based on evidence quality ──
        tracking_quality = 1.0
        if len(poses) < 20:
            tracking_quality = 0.3
        elif len(poses) < 40:
            tracking_quality = 0.6
        if bool(det_bundle.get("yolo11_degraded")):
            tracking_quality = min(tracking_quality, 0.55)

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

        # Re-run cap_confidence with prediction assumption flags
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
        if not bool(kf_validation.get("final_keyframe_gate_pass", True)):
            ar0 = dict(analysis_reliability) if isinstance(analysis_reliability, dict) else {
                "level": "medium", "capped_confidence": 50, "reasons": [],
            }
            r = list(ar0.get("reasons", []))
            r.append("keyframe_order_unsafe_uniform_vision")
            ar0["reasons"] = r
            cc = int(ar0.get("capped_confidence", 50))
            ar0["capped_confidence"] = max(15, cc - 25)
            ar0["level"] = "high" if ar0["capped_confidence"] >= 75 else (
                "medium" if ar0["capped_confidence"] >= 50 else "low"
            )
            analysis_reliability = ar0
        if plus_degraded_flags:
            arp = dict(analysis_reliability) if isinstance(analysis_reliability, dict) else {
                "level": "medium", "capped_confidence": 50, "reasons": [],
            }
            pr = list(arp.get("reasons", []))
            for _f in plus_degraded_flags:
                _tag = f"pipeline_degraded:{_f}"
                if _tag not in pr:
                    pr.append(_tag)
            arp["reasons"] = pr
            _cc = int(arp.get("capped_confidence", 50))
            arp["capped_confidence"] = max(12, _cc - 8 * min(len(plus_degraded_flags), 4))
            arp["level"] = "high" if arp["capped_confidence"] >= 75 else (
                "medium" if arp["capped_confidence"] >= 45 else "low"
            )
            analysis_reliability = arp
        if _withhold_formal_score:
            arw = dict(analysis_reliability) if isinstance(analysis_reliability, dict) else {
                "level": "low", "capped_confidence": 0, "reasons": [],
            }
            prw = list(arw.get("reasons") or [])
            if "formal_score_withheld_unreliable_keyframes" not in prw:
                prw.append("formal_score_withheld_unreliable_keyframes")
            arw["reasons"] = prw
            arw["capped_confidence"] = min(int(arw.get("capped_confidence", 0)), 15)
            arw["level"] = "low"
            analysis_reliability = arw
        trajectory_data = compute_wrist_trajectory(poses)
        research_bundle = run_research_refine(len(poses), video_path=tmp_path)
        logger.info(
            "[PLUS] auxiliary_modules research_status=%s (non_blocking_auxiliary)",
            str(research_bundle.get("status")),
        )

        logger.info(f"[PLUS] Total Plus analysis: {time.time() - t0:.1f}s")

        analysis_id = str(uuid.uuid4())
        fps_val = 30.0
        source_frame_count = 0
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

        _gate_tr = strict_decision.get("gate_decision_trace") or {}
        debug_mode = _truthy(request.query_params.get("debug_keyframes")) or _truthy(
            request.headers.get("x-stellar-debug-keyframes")
        )
        final_gate_pass = bool(kf_validation.get("final_keyframe_gate_pass", False))
        def _has_missing_images(rows: list[dict]) -> bool:
            return any(not str(k.get("image_base64") or "").strip() for k in rows or [])

        gate_source = str(kf_validation.get("final_keyframe_source") or "")
        product_ready = bool(final_gate_pass and not partial_mode and gate_source != "smart_gate_failed" and not _has_missing_images(keyframes))
        keyframe_display_mode = "product_ready" if product_ready else "degraded_failed"
        display_keyframes = [dict(k) for k in keyframes if str(k.get("image_base64") or "").strip()]
        official_keyframes = list(display_keyframes)
        official_phase_keyframes = dict(phase_keyframes) if display_keyframes else {}
        if not product_ready:
            logger.warning(
                "[PLUS] product strip unavailable gate_pass=%s source=%s partial_mode=%s missing_images=%s",
                final_gate_pass,
                gate_source,
                partial_mode,
                _has_missing_images(keyframes),
            )
        logger.info(
            "[ROLE=KEYFRAME_SERVICE] extracted=%d display=%d product_ready=%s final_source=%s",
            len(keyframes),
            len(display_keyframes),
            product_ready,
            gate_source,
        )

        def _serialize_keyframes(rows: list[dict]) -> list[dict]:
            return [
                {
                    "phase": kf["phase"],
                    "label_en": kf["label_en"],
                    "label_zh": kf["label_zh"],
                    "frame_index": kf.get("frame_index"),
                    "timestamp": kf["timestamp"],
                    "confidence": kf.get("confidence"),
                    "selection_reason": kf.get("selection_reason"),
                    "fallback_used": kf.get("fallback_used", False),
                    "image_base64": kf["image_base64"],
                    "pose_snapshot": kf.get("pose_snapshot"),
                    "width": kf.get("width", 320),
                    "height": kf.get("height", 320),
                    "source_pose_idx": kf.get("source_pose_idx"),
                    "source_frame_index": kf.get("source_frame_index", kf.get("frame_index")),
                }
                for kf in rows
            ]

        _kf_debug_rows = _keyframes_snapshot_pre_partial if partial_mode else keyframes
        _phase_debug_map = _phase_keyframes_snapshot_pre_partial if partial_mode else phase_keyframes
        _gemini_obs_payload: dict[str, Any] = {}
        _gem_obs_issues: list[str] = []
        _obs_images: list[str] = []
        _obs_labels: list[Optional[str]] = []
        _obs_source = "other_actual_frames"
        if official_keyframes:
            _obs_images = [
                str(k.get("image_base64") or "")
                for k in official_keyframes
                if str(k.get("image_base64") or "").strip()
            ]
            _obs_labels = [str(k.get("phase") or "") or None for k in official_keyframes[: len(_obs_images)]]
            _obs_source = "display_keyframes" if product_ready else "degraded_display_keyframes"
        _labels_trusted = bool(final_gate_pass and phase_evaluations_reliable and not partial_mode and product_ready)
        if _obs_images:
            logger.info(
                "[PLUS] gemini_visual_observation_invoked=1 source=%s frames=%d phase_labels_trusted=%s",
                _obs_source,
                len(_obs_images),
                _labels_trusted,
            )
            _gemini_obs_payload = await _timed_plus_stage(
                "gemini_visual_observation",
                analyze_plus_visual_observation(
                    _obs_images,
                    frame_labels=_obs_labels,
                    phase_labels_trusted=_labels_trusted,
                    source=_obs_source,
                    issues=_gem_obs_issues,
                ),
                min(PLUS_AI_TIMEOUT_S, 90.0),
            )
            logger.info(
                "[PLUS] gemini_visual_observation_generated=1 available=%s source=%s",
                bool(_gemini_obs_payload.get("available")),
                str(_gemini_obs_payload.get("source") or _obs_source),
            )
        else:
            _gemini_obs_payload = {
                "available": False,
                "mode": "observation_only",
                "source": _obs_source,
                "phase_labels_trusted": False,
                "summary_zh": "",
                "summary_en": "",
                "bullets_zh": [],
                "bullets_en": [],
                "frame_notes": [],
                "issues": _gem_obs_issues,
                "validation_issues": _gem_obs_issues,
                "used_as_authoritative_source": False,
            }
            if not _obs_images:
                _gemini_obs_payload["skip_reason"] = "no_visible_frame_b64"
        _gemini_obs_payload["observed_phase_keyframes"] = dict(phase_debug.get("gemini_observed_keyframes") or {})
        _gemini_obs_payload["used_as_authoritative_source"] = False
        logger.info(
            "[PLUS] gemini_visual_observation_packed=1 available=%s source=%s visible_frames=%d",
            bool(_gemini_obs_payload.get("available")),
            str(_gemini_obs_payload.get("source") or _obs_source),
            len(_obs_images),
        )
        logger.info(
            "[PLUS] gemini_observation source=%s phase_labels_trusted=%s visible_frames=%d available=%s",
            str(_gemini_obs_payload.get("source") or _obs_source),
            bool(_gemini_obs_payload.get("phase_labels_trusted")),
            len(_obs_images),
            bool(_gemini_obs_payload.get("available")),
        )

        _merge_gemini_observation_into_withheld_ai_result(
            ai_result,
            _gemini_obs_payload,
            withhold=bool(_withhold_formal_score),
        )

        _analysis_mode = "pose_plus_partial" if partial_mode else ("pose_plus_degraded" if plus_degraded_flags else "pose_plus")
        _sp_pack_br, _rp_pack_br = _plus_keyframe_pack_block_reasons(kf_validation, _strict_fail_rs)
        if _withhold_formal_score:
            _report_status_out = "unavailable_due_to_unreliable_keyframes"
            _report_err_out = "KEYFRAME_STRICT_CONTRACT_FAIL"
            _final_ui_safe_score_state = "null_withheld_unreliable_keyframes"
        else:
            _report_status_out = "available"
            _report_err_out = None
            _final_ui_safe_score_state = "numeric_ok"
        response_dict = {
            "analysis_id": analysis_id,
            "type": "plus",
            "analysis_mode": _analysis_mode,
            "partial_mode": bool(partial_mode),
            "plus_pipeline_degraded": bool(plus_degraded_flags),
            "plus_degraded_flags": list(plus_degraded_flags),
            "biomech_hard_passed": bool(biomech_hard_passed),
            "biomech_hard_reasons": list((hard_biomech.get("reasons") or []) if not biomech_hard_passed else []),
            "yolo11_degraded": bool(det_bundle.get("yolo11_degraded")),
            "yolo11_status": str(det_bundle.get("status") or ""),
            "phase_pipeline_applied": not partial_mode,
            "phase_repair_failed": bool(phase_repair_failed),
            "final_keyframes_product_ready": bool(product_ready),
            "unreliable_debug_only": bool(debug_mode and not final_gate_pass and not partial_mode),
            "ai_provider": ai_result.get("ai_provider", "unknown"),
            "ai_key": ai_result.get("ai_key"),
            "posture_score": (None if _withhold_formal_score else ai_result.get("posture_score", 0.0)),
            "primary_diagnosis": ai_result.get("primary_diagnosis", {}),
            "additional_issues": ai_result.get("additional_issues", []),
            "quick_tip_zh": ai_result.get("quick_tip_zh", ""),
            "quick_tip_en": ai_result.get("quick_tip_en", ""),
            "problem_description_zh": ai_result.get("problem_description_zh", ""),
            "problem_description_en": ai_result.get("problem_description_en", ""),
            "swing_phase_evaluations": ai_result.get("swing_phase_evaluations", []),
            "training": ai_result.get("training", {}),
            "recommended_videos": ai_result.get("recommended_videos", []),
            "scores": (None if _withhold_formal_score else ai_result.get("scores", {})),
            "total_score": (None if _withhold_formal_score else ai_result.get("total_score", 0)),
            "report_status": _report_status_out,
            "report_error_code": _report_err_out,
            "score_pack_blocked_reason": _sp_pack_br,
            "report_pack_blocked_reason": _rp_pack_br,
            "final_ui_safe_score_state": _final_ui_safe_score_state,
            "fallback_rebuild_material_change": kf_validation.get("fallback_rebuild_material_change"),
            "issues": ai_result.get("issues", []),
            "issues_zh": ai_result.get("issues_zh", []),
            "suggestions": ai_result.get("suggestions", []),
            "suggestions_zh": ai_result.get("suggestions_zh", []),
            "summary": ai_result.get("summary", ""),
            "summary_zh": ai_result.get("summary_zh", ""),
            "advanced_metrics": ai_result.get("advanced_metrics", {}),
            "keyframes": (
                _serialize_keyframes(_keyframes_snapshot_pre_partial)
                if partial_mode and debug_mode
                else _serialize_keyframes(official_keyframes)
            ),
            "debug_keyframes": _serialize_keyframes(_kf_debug_rows) if debug_mode else [],
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
            "phase_keyframes": official_phase_keyframes,
            "debug_phase_keyframes": dict(_phase_debug_map) if debug_mode else {},
            "phase_source": phase_source,
            "phase_validation": phase_validation_for_cap,
            "keyframe_validation": kf_validation,
            "final_phase_keyframes": kf_validation.get("final_phase_keyframes"),
            "final_keyframe_validation": kf_validation.get("final_keyframe_validation"),
            "final_keyframe_order_ok": kf_validation.get("final_keyframe_order_ok"),
            "final_keyframe_time_order_ok": kf_validation.get("final_keyframe_time_order_ok"),
            "final_keyframe_source": kf_validation.get("final_keyframe_source"),
            "final_keyframe_gate_pass": kf_validation.get("final_keyframe_gate_pass"),
            "keyframes_degraded": bool(not product_ready),
            "phase_keyframes_degraded": bool(not product_ready),
            "keyframe_display_mode": keyframe_display_mode,
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
            "optional_modules": optional_modules,
            "sweet_spot_warning": bool(_gate_tr.get("sweet_spot_warning")),
            "sweet_spot_confidence": sweet_spot_bundle.get("sweet_spot_confidence"),
            "phase_debug": phase_debug,
            "gemini_observation": _gemini_obs_payload,
            "provider_debug": {
                "pose_provider": pose_stream_meta.get("provider_meta"),
                "detection_provider": det_bundle.get("provider_meta"),
                "tracking_provider": tracks.get("provider_meta"),
                "pose3d_provider": motion3d_bundle.get("provider_meta"),
                "action_provider": {"status": "plus_prov3_only", "provider_name": "none"},
                "research_provider": research_bundle.get("provider_meta"),
            },
            "video_meta": {
                "fps": fps_val,
                "total_pose_frames": len(poses),
                "duration_s": round(poses[-1]["timestamp"] if poses else 0, 3),
                "source_frame_count": source_frame_count,
            },
        }

        # ── Quality gate: strip hallucinated fields and add warnings when reliability is low ──
        reliability_level = analysis_reliability.get("level", "medium") if isinstance(analysis_reliability, dict) else "medium"
        if reliability_level == "low":
            response_dict.pop("recommended_videos", None)
            response_dict["quality_warning"] = "分析可信度较低，结果仅供参考 / Analysis reliability is low, results are for reference only"
            # Strip frequency_percent from phase evaluations
            for ev in response_dict.get("swing_phase_evaluations", []):
                ev.pop("frequency_percent", None)
        if phase_source == "kinematic_degraded" and isinstance(kf_validation, dict) and kf_validation.get("near_duplicates", 0) > 2:
            response_dict["keyframe_warning"] = "关键帧质量不佳，部分阶段图像可能相似 / Keyframe quality is poor, some phase images may look similar"
        # Surface prediction assumption flags at top level
        if prediction.get("hand_assumed"):
            response_dict["hand_assumed"] = prediction["hand_assumed"]
            response_dict["hand_warning"] = prediction.get("hand_warning", "")
        if prediction.get("club_assumed"):
            response_dict["club_assumed"] = prediction["club_assumed"]
            response_dict["club_warning"] = prediction.get("club_warning", "")

        logger.info(
            "[PLUS] return summary return_keyframes_count=%d return_phase_keyframes_count=%d partial_mode=%s "
            "final_keyframe_gate_pass=%s keyframes_degraded=%s keyframe_display_mode=%s "
            "plus_pipeline_degraded=%s flags=%s analysis_mode=%s",
            len(response_dict.get("keyframes") or []),
            len(response_dict.get("phase_keyframes") or {}),
            bool(partial_mode),
            bool(response_dict.get("final_keyframe_gate_pass")),
            bool(response_dict.get("keyframes_degraded")),
            str(response_dict.get("keyframe_display_mode")),
            bool(response_dict.get("plus_pipeline_degraded")),
            list(plus_degraded_flags),
            str(response_dict.get("analysis_mode")),
        )
        logger.info(
            "[STELLAR_PLUS_PIPELINE] response_tier=%s partial_mode=%s degraded_flags=%s",
            "partial" if partial_mode else ("degraded" if plus_degraded_flags else "full"),
            bool(partial_mode),
            list(plus_degraded_flags),
        )

        _pod_trace = (kf_validation or {}).get("phase_oriented_semantic_debug") or {}
        logger.info(
            "[PLUS] score_pack_trace score_pack_blocked_reason=%s report_pack_blocked_reason=%s "
            "fallback_rebuild_material_change=%s final_ui_safe_score_state=%s withhold_formal_score=%s "
            "phase_reselect_strategy=%s phase_candidate_window=%s phase_candidate_scores=%s "
            "top_semantic_score=%s impact_semantic_score=%s finish_semantic_score=%s "
            "final_phase_semantic_pass=%s final_phase_semantic_fail_reasons=%s final_keyframe_source=%s "
            "source_pose_idx_list=%s min_gap_frames=%s duplicate_pairs=%s",
            response_dict.get("score_pack_blocked_reason"),
            response_dict.get("report_pack_blocked_reason"),
            response_dict.get("fallback_rebuild_material_change"),
            response_dict.get("final_ui_safe_score_state"),
            bool(_withhold_formal_score),
            _pod_trace.get("phase_reselect_strategy"),
            _pod_trace.get("phase_candidate_window"),
            _pod_trace.get("phase_candidate_scores"),
            _pod_trace.get("top_semantic_score"),
            _pod_trace.get("impact_semantic_score"),
            _pod_trace.get("finish_semantic_score"),
            (kf_validation or {}).get("final_phase_semantic_pass"),
            (kf_validation or {}).get("final_phase_semantic_fail_reasons"),
            str((kf_validation or {}).get("final_keyframe_source") or ""),
            _pod_trace.get("source_pose_idx_list"),
            _pod_trace.get("min_gap_frames"),
            _pod_trace.get("duplicate_pairs"),
        )
        packed_response = pack_plus_response(response_dict)
        logger.info("[ROLE=STRICT_GATE] pass=%s reasons=%s", bool(strict_decision.get("pass")), list(strict_decision.get("reasons") or []))
        logger.info("[ROLE=API_PACK] keyframes=%d partial=%s image_missing=%s", len(packed_response.get("keyframes") or []), bool(packed_response.get("result_partial")), bool(packed_response.get("image_missing")))
        log_non_finite_if_any(logger, packed_response, "plus_analyze")
        return sanitize_json_floats(packed_response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plus analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
