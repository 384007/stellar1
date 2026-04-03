import asyncio
import logging
import os
import tempfile
import uuid
import base64

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional

from routers.auth import get_current_user
from services.json_sanitize import log_non_finite_if_any, safe_float, sanitize_json_floats

router = APIRouter()
logger = logging.getLogger(__name__)


class LiteAnalyzeRequest(BaseModel):
    video_url: str
    user_id: Optional[str] = None


class RecalculatePredictionRequest(BaseModel):
    pose_data: dict
    all_frame_angles: list[dict] = []
    swing_duration: float = 1.2
    club_type: Optional[str] = None
    club_group: Optional[str] = None
    hand: Optional[str] = None
    hand_confidence: Optional[float] = None
    preferred_ball_speed: Optional[float] = None


def _extract_frames_safe(tmp_path: str, max_frames: int = 12):
    """Pose extraction only; smart keyframes are built later in /lite (never uniform 5-frame strip for AI)."""
    from services.pose_service import extract_poses_from_video

    poses, pose_bundle = extract_poses_from_video(tmp_path, max_frames=max_frames)
    return (poses if poses else []), pose_bundle


def _extract_keyframe_images_safe(tmp_path: str, num_frames: int = 5) -> list[str]:
    """Extract raw frame images as base64 without pose detection."""
    try:
        import cv2
        from services.video_utils import get_video_rotation, apply_rotation
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        rotation = get_video_rotation(tmp_path)
        import numpy as np
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        images = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = apply_rotation(frame, rotation)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            images.append(base64.b64encode(buf.tobytes()).decode())
        cap.release()
        return images
    except Exception as e:
        print(f"[analyze] Frame extraction failed: {e}")
        return []


@router.post("/lite")
async def analyze_lite(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Accept either multipart file upload or JSON with video_url."""
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

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.get(video_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to download video")

            suffix = ".mov" if ".mov" in video_url.lower() else ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

        loop = asyncio.get_event_loop()
        _LITE_POSE_EXTRACT_S = float(os.getenv("STELLAR_LITE_POSE_EXTRACT_S", "300"))
        try:
            poses, pose_quality_bundle = await asyncio.wait_for(
                loop.run_in_executor(None, _extract_frames_safe, tmp_path),
                timeout=_LITE_POSE_EXTRACT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"分析步骤超时: pose_extraction（>{_LITE_POSE_EXTRACT_S:.0f}s）/ Pose extraction timed out"
                ),
            )

        if poses and len(poses) > 0:
            from services.gemini_service import LITE_AI_TIMEOUT_S, analyze_swing_lite, cap_confidence
            from services.hud_service import generate_hud_data
            from services.pose_service import pose_for_skeleton_render
            from services.shot_predictor import predict_shot
            from services.handedness_service import detect_handedness
            from services.swing_flow_utils import (
                detect_swing_phases,
                get_phase_keyframes,
                validate_phase_keyframes,
                respace_phase_keyframes,
                compute_wrist_trajectory,
                build_semantic_phase_report,
                compute_phase_evaluations_reliable,
                assess_gemini_uniform_map_vs_final_phase_strip,
                build_phase_boundary_flags,
            )
            from services.keyframe_service import (
                extract_keyframes_smart,
                ensure_keyframes_ordered_for_ai,
                build_ai_vision_images_from_phase_keyframes,
            )
            from services.phase_analysis_gate import (
                build_phase_alignment_fail_detail,
                should_run_phase_analysis_strict,
            )

            region = "CN" if request.headers.get("CF-IPCountry", "").upper() == "CN" else "global"
            swing_phases = detect_swing_phases(poses)
            for i, pose in enumerate(poses):
                if i < len(swing_phases):
                    pose["phase_data"] = swing_phases[i]

            phase_keyframes_map = dict(get_phase_keyframes(swing_phases, poses))
            phase_validation = validate_phase_keyframes(phase_keyframes_map, poses, source="kinematic")
            phase_source = "kinematic"
            if not phase_validation.get("spacing_ok", True) or not phase_validation.get("passed", True):
                phase_keyframes_map = respace_phase_keyframes(dict(phase_keyframes_map), len(poses))
                phase_validation = validate_phase_keyframes(phase_keyframes_map, poses, source="kinematic")
                if not phase_validation.get("passed", True):
                    phase_source = "kinematic_degraded"

            pk_mut = dict(phase_keyframes_map)
            pk_snap = dict(pk_mut)
            smart_result = extract_keyframes_smart(
                tmp_path, poses, swing_phases, pk_mut, 280,
            )
            if isinstance(smart_result, tuple):
                keyframes, _kf_val = smart_result
            else:
                keyframes = smart_result
                _kf_val = {
                    "total_keyframes": len(keyframes),
                    "near_duplicates": 0,
                    "time_too_close": 0,
                    "all_passed": True,
                    "details": [],
                }
            keyframes, _kf_val, pk_mut, _ = ensure_keyframes_ordered_for_ai(
                tmp_path,
                poses,
                swing_phases,
                pk_snap,
                keyframes,
                _kf_val,
                pk_mut,
                280,
            )
            phase_keyframes_map = pk_mut
            if bool(_kf_val.get("reselected_top")) or bool(_kf_val.get("reselected_impact")):
                phase_validation = validate_phase_keyframes(
                    phase_keyframes_map, poses, source=f"{phase_source}_reselected",
                )

            keyframe_images = build_ai_vision_images_from_phase_keyframes(
                tmp_path, poses, keyframes, phase_keyframes_map,
            )

            sem_report = build_semantic_phase_report(
                poses,
                dict(phase_keyframes_map),
                phase_validation,
                keyframes,
                dict(_kf_val.get("final_keyframe_validation") or {}),
            )
            kf_src = str(_kf_val.get("final_keyframe_source") or "")
            _gate_ok = bool(_kf_val.get("final_keyframe_gate_pass"))
            sem_ok = bool(sem_report.get("final_phase_semantic_ok"))
            sem_ok_strict = bool(sem_report.get("final_phase_semantic_ok_strict"))
            pv_pass = bool(sem_report.get("phase_validation_passed"))
            phase_validation_for_cap = {**(phase_validation or {}), "passed": pv_pass}
            gemini_assess = assess_gemini_uniform_map_vs_final_phase_strip(
                phase_source, None, dict(phase_keyframes_map),
            )
            from services.shot_predictor import estimate_sweet_spot_robust

            hand_info_gate = detect_handedness(poses, swing_phases=swing_phases)
            hand_for_sweet = str(hand_info_gate.get("hand") or "UNKNOWN")
            imp_for_sweet = int(phase_keyframes_map.get("impact", len(poses) // 2))
            sweet_spot_bundle = estimate_sweet_spot_robust(poses, imp_for_sweet, hand=hand_for_sweet)
            strict_decision = should_run_phase_analysis_strict(
                pose_quality_bundle=pose_quality_bundle,
                sem_report=sem_report,
                kf_validation=_kf_val,
                keyframe_count=len(keyframes),
                ai_vision_count=len(keyframe_images),
                gemini_assess=gemini_assess,
                sweet_spot_bundle=sweet_spot_bundle,
            )
            if not strict_decision["pass"]:
                raise HTTPException(
                    status_code=422,
                    detail=build_phase_alignment_fail_detail(strict_decision),
                )
            phase_strip_technically_sound = compute_phase_evaluations_reliable(
                final_phase_semantic_ok=sem_ok_strict,
                phase_validation_passed=pv_pass,
                final_keyframe_source=kf_src,
                final_keyframe_gate_pass=_gate_ok,
                ai_vision_frame_count=len(keyframe_images),
                keyframe_strip_frame_count=len(keyframes),
                gemini_uniform_map_applies=False,
                gemini_map_aligned_with_final_strip=None,
            )
            phase_evaluations_reliable = False
            phase_images_reliable = phase_strip_technically_sound
            phase_boundary = build_phase_boundary_flags(
                final_keyframe_source=kf_src,
                keyframe_strip_frame_count=len(keyframes),
                ai_vision_frame_count=len(keyframe_images),
                gemini_strip_assessment=gemini_assess,
                analysis_route="lite",
                plus_grade_phase_evaluations=False,
            )
            phase_boundary["phase_strip_technically_sound"] = phase_strip_technically_sound
            phase_evaluations_warning = ""
            if not _gate_ok or kf_src == "ordered_fallback_empty":
                phase_source = "kinematic_degraded"
                phase_evaluations_warning = "keyframe_gate_failed_or_empty_fallback"
            elif kf_src == "ordered_fallback":
                phase_evaluations_warning = "ordered_fallback_semantics_untrusted"
            elif not sem_ok:
                phase_evaluations_warning = "semantic_validation_failed"
            elif not pv_pass:
                phase_evaluations_warning = "phase_validation_soft_fail"
            if len(keyframe_images) < 8 or len(keyframes) < 8:
                phase_evaluations_warning = (phase_evaluations_warning + ";incomplete_phase_vision_strip").strip(";")
            mid_idx = len(poses) // 2
            representative_pose = poses[mid_idx]

            ai_result = await asyncio.wait_for(
                analyze_swing_lite(
                    pose_data={
                        "angles": representative_pose["angles"],
                        "all_frame_angles": [p["angles"] for p in poses],
                    },
                    keyframe_images=keyframe_images,
                    region=region,
                    phase_images_reliable=phase_images_reliable,
                ),
                timeout=LITE_AI_TIMEOUT_S + 60.0,
            )

            hand_info = detect_handedness(poses, swing_phases=swing_phases)
            resolved_hand = str(hand_info.get("hand") or "UNKNOWN")

            hud_frames = []
            for pose in poses:
                hud = generate_hud_data(pose_for_skeleton_render(pose), mode="lite", hand=resolved_hand)
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
                impact_pose_idx=int(phase_keyframes_map.get("impact", len(poses) // 2)),
            )
            analysis_id = str(uuid.uuid4())

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
                phase_vision_reliable=phase_strip_technically_sound,
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
                    phase_vision_reliable=phase_strip_technically_sound,
                    sweet_spot_unstable=bool(sweet_spot_bundle.get("sweet_spot_unstable")),
                    sweet_spot_confidence=sweet_spot_bundle.get("sweet_spot_confidence"),
                )
            if isinstance(_kf_val, dict):
                ar = dict(analysis_reliability)
                reasons = list(ar.get("reasons", []))
                cc = int(ar.get("capped_confidence", 50))
                if _kf_val.get("near_duplicates", 0) > 2 or not _kf_val.get("all_passed", True):
                    reasons.append("keyframe_quality_low")
                    cc = max(20, cc - 15)
                if not bool(_kf_val.get("final_keyframe_gate_pass", True)):
                    reasons.append("keyframe_order_unsafe_uniform_vision")
                    cc = max(15, cc - 25)
                ar["reasons"] = reasons
                ar["capped_confidence"] = cc
                ar["level"] = "high" if cc >= 75 else ("medium" if cc >= 50 else "low")
                analysis_reliability = ar

            trajectory = []
            try:
                trajectory = compute_wrist_trajectory(poses)
            except Exception:
                pass

            _lite_gate_tr = strict_decision.get("gate_decision_trace") or {}
            lite_response = {
                "analysis_id": analysis_id,
                "type": "lite",
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
                    }
                    for kf in (keyframes or [])
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
                "swing_phases": swing_phases,
                "trajectory": trajectory,
                "phase_keyframes": phase_keyframes_map,
                "phase_source": phase_source,
                "phase_validation": phase_validation_for_cap,
                "keyframe_validation": _kf_val,
                "final_phase_keyframes": _kf_val.get("final_phase_keyframes") if isinstance(_kf_val, dict) else None,
                "final_keyframe_validation": _kf_val.get("final_keyframe_validation") if isinstance(_kf_val, dict) else None,
                "final_keyframe_order_ok": _kf_val.get("final_keyframe_order_ok") if isinstance(_kf_val, dict) else None,
                "final_keyframe_time_order_ok": _kf_val.get("final_keyframe_time_order_ok") if isinstance(_kf_val, dict) else None,
                "final_keyframe_source": _kf_val.get("final_keyframe_source") if isinstance(_kf_val, dict) else None,
                "final_keyframe_gate_pass": _kf_val.get("final_keyframe_gate_pass") if isinstance(_kf_val, dict) else None,
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
                "phase_pipeline_applied": True,
                "analysis_mode": "pose_lite",
                "analysis_reliability": analysis_reliability,
                "sweet_spot_warning": bool(_lite_gate_tr.get("sweet_spot_warning")),
                "sweet_spot_confidence": sweet_spot_bundle.get("sweet_spot_confidence"),
                "video_meta": {
                    "total_pose_frames": len(poses),
                    "duration_s": round(poses[-1]["timestamp"] if poses else 0, 3),
                },
            }

            # ── Lightweight quality gate for lite route ──
            rel_level = analysis_reliability.get("level", "medium") if isinstance(analysis_reliability, dict) else "medium"
            if rel_level == "low":
                lite_response["quality_warning"] = (
                    "分析可信度较低，结果仅供参考 / Analysis reliability is low, results are for reference only"
                )
            if isinstance(_kf_val, dict) and _kf_val.get("near_duplicates", 0) > 2:
                lite_response["keyframe_warning"] = "关键帧质量不佳 / Keyframe quality is poor"
            if phase_source == "kinematic_degraded" and isinstance(_kf_val, dict) and _kf_val.get("near_duplicates", 0) > 2:
                lite_response.setdefault(
                    "keyframe_warning",
                    "关键帧质量不佳，部分阶段可能重叠 / Keyframe quality is poor",
                )
            # Surface prediction assumption flags
            if prediction.get("hand_assumed"):
                lite_response["hand_warning"] = prediction.get("hand_warning", "")
            if prediction.get("club_assumed"):
                lite_response["club_warning"] = prediction.get("club_warning", "")

            log_non_finite_if_any(logger, lite_response, "analyze_lite")
            return sanitize_json_floats(lite_response)

        print("[analyze] No poses detected, falling back to AI-only analysis")
        from services.gemini_service import IMAGE_ONLY_TIMEOUT_S, analyze_with_images_only
        region = "CN" if request.headers.get("CF-IPCountry", "").upper() == "CN" else "global"
        try:
            frame_images = await asyncio.wait_for(
                loop.run_in_executor(None, _extract_keyframe_images_safe, tmp_path),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="分析步骤超时: image_frame_extraction / Image frame extraction timed out",
            )
        ai_result = await asyncio.wait_for(
            analyze_with_images_only(frame_images, region=region),
            timeout=IMAGE_ONLY_TIMEOUT_S + 30.0,
        )

        image_only_payload = {
            "analysis_id": str(uuid.uuid4()),
            "type": "lite",
            "analysis_mode": "image_only",
            "phase_pipeline_applied": False,
            "phase_evaluations_reliable": False,
            "phase_boundary": {
                "expected_phase_vision_frames": 0,
                "keyframe_strip_frame_count": 0,
                "ai_vision_frame_count": len(frame_images),
                "phase_vision_complete_strip": False,
                "phase_keyframe_extraction_label": "none",
                "phase_strip_is_monotonic_fallback_only": False,
                "gemini_uniform_map_vs_strip": {
                    "gemini_uniform_thumbnail_map_applies": False,
                    "gemini_map_aligned_with_final_strip": None,
                    "aligned": None,
                },
                "analysis_route_tier": "lite_image_only",
                "plus_grade_phase_evaluations": False,
            },
            "phase_warning_zh": "未进行姿态阶段管线；本结果为纯图像分析，不含阶段校验关键帧。",
            "phase_warning_en": "No pose phase pipeline; image-only analysis — not phase-validated keyframes.",
            "ai_provider": ai_result.get("ai_provider", "unknown"),
            "ai_key": ai_result.get("ai_key"),
            "what_i_see": ai_result.get("what_i_see", ""),
            "what_i_see_zh": ai_result.get("what_i_see_zh", ""),
            "is_golf_swing": ai_result.get("is_golf_swing", False),
            "scores": ai_result.get("scores", {}),
            "total_score": ai_result.get("total_score", 0),
            "issues": ai_result.get("issues", []),
            "issues_zh": ai_result.get("issues_zh", []),
            "suggestions": ai_result.get("suggestions", []),
            "suggestions_zh": ai_result.get("suggestions_zh", []),
            "summary": ai_result.get("summary", ""),
            "summary_zh": ai_result.get("summary_zh", ""),
            "keyframes": [],
            "skeleton_data": {
                "frames": [],
                "total_frames": 0,
                "joint_space": "analysis_frame",
                "joint_sources": {
                    "raw_detection_joints": "pose.raw_detection_joints",
                    "analysis_joints": "pose.joints",
                    "render_joints": "pose.render_joints->pose.joints",
                },
            },
            "prediction": {
                "predicted_distance": ai_result.get("prediction", {}).get("predicted_distance", 0),
                "lateral_offset": ai_result.get("prediction", {}).get("lateral_offset", 0),
                "shot_shape": ai_result.get("prediction", {}).get("shot_shape", "N/A"),
                "shot_shape_zh": ai_result.get("prediction", {}).get("shot_shape_zh", "未知"),
                "club_head_speed": ai_result.get("prediction", {}).get("club_head_speed", 0),
                "ball_speed": ai_result.get("prediction", {}).get("ball_speed", 0),
                "launch_angle": ai_result.get("prediction", {}).get("launch_angle", 0),
                "spin_rate": ai_result.get("prediction", {}).get("spin_rate", 0),
                "smash_factor": ai_result.get("prediction", {}).get("smash_factor", 0),
                "trajectory": [],
                "hand": "UNKNOWN",
                "hand_confidence": 0.0,
                "baseline_distance": 0.0,
                "technique_multiplier": 1.0,
                "strike_multiplier": 1.0,
                "speed_multiplier": 1.0,
                "distance_confidence": 0.0,
                "distance_debug": {"source": "ai_only_fallback"},
            },
        }
        log_non_finite_if_any(logger, image_only_payload, "analyze_lite_image_only")
        return sanitize_json_floats(image_only_payload)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/recalculate")
async def recalculate_prediction(
    payload: RecalculatePredictionRequest,
    current_user: Optional[dict] = Depends(get_current_user),
):
    try:
        from services.shot_predictor import predict_shot

        pose_data = payload.pose_data or {}
        all_angles = payload.all_frame_angles or []
        hand = (payload.hand or "UNKNOWN").upper().strip()
        if hand not in ("R", "L", "UNKNOWN"):
            hand = "UNKNOWN"

        pref_speed = payload.preferred_ball_speed
        if pref_speed is not None:
            ps = safe_float(pref_speed, 0.0)
            pref_speed = ps if ps > 0 else None

        prediction = predict_shot(
            pose_data=pose_data,
            swing_duration=safe_float(payload.swing_duration, 1.2),
            all_frame_angles=all_angles,
            club_type=payload.club_type,
            club_group=payload.club_group,
            hand=hand,
            hand_confidence=safe_float(payload.hand_confidence, 0.0),
            preferred_ball_speed=pref_speed,
        )
        recalc_out = {"prediction": prediction}
        log_non_finite_if_any(logger, recalc_out, "recalculate")
        return sanitize_json_floats(recalc_out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recalculate failed: {str(e)}")
