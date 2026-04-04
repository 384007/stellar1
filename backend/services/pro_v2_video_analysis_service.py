"""Pro v2 — single orchestrator (240fps → swing window → dense scan → keyframes → impact → gate → sheet → AI).

Does **not** use MMAction2 / TSN or any legacy Pro pose chain — FFmpeg + OpenCV motion + Gemini text only.
Gemini uses ``PRO_V2_REPORT_PROMPT`` (pass-2 + local fallback); full summaries; up to 3 phase-tagged issues/suggestions in API payload.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import cv2

from services.pro_contact_sheet_service import build_pro_keyframe_contact_sheet
from services.pro_v2_dense_scan_service import DenseFrame, dense_scan_swing_region
from services.pro_v2_ffmpeg_service import run_pro_v2_ffmpeg_preprocess
from services.pro_v2_impact_refine_service import refine_impact_keyframe_only
from services.pro_v2_keyframe_picker_service import pick_eight_keyframes_motion_only
from services.pro_v2_ai_routing_service import run_pro_v2_ai_routing
from services.pro_v2_keyframe_review_service import CORE_PHASE_ORDER, run_core_keyframe_review_round
from services.pro_v2_report_service import LIMITED_NOTICE_ZH, pop_pro_v2_report_meta, write_pro_v2_ai_report
from services.pro_v2_screen_preprocess_service import run_pro_v2_screen_preprocess
from services.pro_v2_simple_gate_service import run_simple_gate
from services.pro_v2_picker_tuning_service import build_pro_v2_picker_tuning, log_picker_tuning
from services.pro_v2_strategy_profiles import (
    compute_screen_relaxed_margin,
    derive_routing_execution,
    ffmpeg_vf_for_attempt,
    log_route_apply,
    resolve_screen_unsharp,
    second_pass_impact_aggressive,
)
from services.pro_v2_swing_window_service import find_swing_window_seconds

logger = logging.getLogger(__name__)


def _source_frame_count(video_path: str) -> int:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(0, n)
    except Exception:
        return 0


def _normalize_training_plan(raw: Any) -> dict[str, Any] | None:
    if not raw or not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        kl = k.strip().lower()
        if not kl.startswith("day"):
            continue
        suf = kl[3:]
        if not suf.isdigit():
            continue
        day_key = f"day{int(suf)}"
        focus = str(v.get("focus") or "").strip()
        drills_raw = v.get("drills")
        drills: list[str] = []
        if isinstance(drills_raw, list):
            drills = [str(x).strip() for x in drills_raw if str(x).strip()]
        dur = str(v.get("duration") or "20–30 min").strip()
        if focus or drills:
            out[day_key] = {
                "focus": focus or "练习重点",
                "drills": drills[:6] or ["参考改进建议完成练习"],
                "duration": dur,
            }
    for need in (f"day{i}" for i in range(1, 8)):
        if need not in out:
            return None
    return out


def _training_plan_fallback(suggestions_zh: list[str], suggestions: list[str]) -> dict[str, Any]:
    zhs = [str(s).strip() for s in suggestions_zh if str(s).strip()]
    ens = [str(s).strip() for s in suggestions if str(s).strip()]
    zh0 = zhs[0] if zhs else "按计划完成挥杆与节奏练习"
    en0 = ens[0] if ens else "Swing and tempo drills"
    out: dict[str, Any] = {}
    for i in range(7):
        zh = zhs[i] if i < len(zhs) else zh0
        en = ens[i] if i < len(ens) else en0
        out[f"day{i + 1}"] = {
            "focus": zh[:120],
            "drills": [zh[:200], en[:200]],
            "duration": "20–30 min",
        }
    return out


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
    extras: dict[str, Any] | None = None,
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
    ctx: dict[str, Any] = {
        "pipeline": "pro_v2",
        "motion_authority": "dense_opencv_only",
        "fps": fps,
        "swing_window_s": [round(swing_t0, 4), round(swing_t1, 4)],
        "dense_frame_count": len(dense),
        "keyframes": rows,
    }
    if extras:
        ctx.update(extras)
    return ctx


def _screen_motion_flag(analysis_input: str) -> bool:
    return analysis_input == "screen_cropped"


def _run_motion_pipeline_once(
    *,
    ffmpeg_input_path: str,
    work: Path,
    rough_impact_time_s: float | None,
    analysis_input: str,
    picker_tuning: dict[str, Any] | None = None,
    impact_refine_aggressive: bool = False,
    analysis_vf_prefix: str | None = None,
    dense_pose_priority: bool = False,
    dense_club_emphasis: float = 0.0,
    min_dense_frames: int = 16,
    legacy_picker_variant: int = 0,
) -> tuple[Any, float, float, list[DenseFrame], list[dict[str, Any]]]:
    ff = run_pro_v2_ffmpeg_preprocess(
        ffmpeg_input_path,
        str(work),
        rough_impact_time_s=rough_impact_time_s,
        analysis_vf_prefix=analysis_vf_prefix,
    )
    sm = _screen_motion_flag(analysis_input)
    t0, t1 = find_swing_window_seconds(
        ff.analysis_240_path,
        fps=ff.fps,
        duration_s=ff.duration_s,
        screen_mode=sm,
    )
    dense = dense_scan_swing_region(
        ff.analysis_240_path,
        fps=ff.fps,
        t_start_s=t0,
        t_end_s=t1,
        screen_mode=sm,
        pose_priority=bool(dense_pose_priority and sm),
        club_emphasis=float(dense_club_emphasis or 0.0),
    )
    if len(dense) < min_dense_frames:
        raise RuntimeError("pro_v2: swing region too short or static — record a clearer swing clip")
    tun = dict(picker_tuning or {})
    tun.setdefault("legacy_picker_variant", legacy_picker_variant)
    keyframes = pick_eight_keyframes_motion_only(
        ff.analysis_240_path,
        dense,
        screen_mode=sm,
        picker_variant=int(legacy_picker_variant),
        picker_tuning=tun,
    )
    keyframes = refine_impact_keyframe_only(
        ff.analysis_240_path,
        keyframes,
        aggressive=bool(impact_refine_aggressive),
    )
    keyframes, _, _ = run_simple_gate(
        keyframes,
        fps=ff.fps,
        analysis_video_path=ff.analysis_240_path,
        dense=dense,
    )
    return ff, t0, t1, dense, keyframes


def _trust_extras(
    *,
    route: dict[str, Any] | None,
    routing_execution: dict[str, Any] | None,
    analysis_input: str,
    analysis_trust: str,
    report_mode: str,
    review_round: int,
    core_frame_scores: dict[str, Any],
    keyframe_mismatch_notice: bool,
    retry_reasons: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "screen_mode": True,
        "analysis_input": analysis_input,
        "analysis_trust": analysis_trust,
        "report_mode": report_mode,
        "review_round": review_round,
        "core_frame_scores": core_frame_scores,
        "keyframe_mismatch_notice": keyframe_mismatch_notice,
        "retry_reasons": list(retry_reasons or [])[:12],
    }
    if route:
        out["routing_strategy"] = route
    if routing_execution:
        out["routing_execution"] = routing_execution
    return out


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
        "[PRO_V2] start analysis_id=%s chain=ffmpeg_opencv_motion_gemini mmaction2_used=false screen_mode=%s",
        analysis_id,
        "true" if screen_mode else "false",
    )

    route: dict[str, Any] | None = None
    ff: Any
    t0: float = 0.0
    t1: float = 0.0
    dense: list[DenseFrame] = []
    keyframes: list[dict[str, Any]] = []
    analysis_input = "raw"
    screen_cropped_video_path: str | None = None
    review_round_done = 0
    core_frame_scores: dict[str, Any] = {}
    retry_reasons_final: list[str] = []
    analysis_trust = "high_trust"
    report_mode = "formal"
    keyframe_mismatch_notice = False

    routing_last_pass: dict[str, Any] = {}
    routing_exec_static: dict[str, Any] | None = None
    routing_full_out: dict[str, Any] | None = None

    if screen_mode:
        route = await run_pro_v2_ai_routing(input_video_path, screen_mode_requested=True)
        rp = derive_routing_execution(route)
        log_route_apply(rp, analysis_id=analysis_id)
        routing_exec_static = {
            "quality_level": rp.quality_level,
            "use_deblur": rp.use_deblur,
            "use_heavy_club_tracking": rp.use_heavy_club_tracking,
            "pose_priority": rp.pose_priority,
            "expected_confidence_ceiling": rp.expected_confidence_ceiling,
            "min_dense_frames": rp.min_dense_frames,
            "dense_club_emphasis": rp.dense_club_emphasis,
            "screen_apply_unsharp": rp.screen_apply_unsharp,
            "ffmpeg_vf_prefix_routing": rp.ffmpeg_analysis_vf_prefix,
        }
        retry_reasons_final: list[str] = []

        for attempt in range(2):
            pick_dict = build_pro_v2_picker_tuning(
                routing=rp,
                retry_reasons=retry_reasons_final,
                attempt_index=attempt,
                screen_mode=True,
            )
            log_picker_tuning(
                pick_dict,
                analysis_id=analysis_id,
                attempt_index=attempt,
                review_round=attempt + 1,
                retry_reasons=retry_reasons_final,
                analysis_trust="pending",
            )
            relaxed = compute_screen_relaxed_margin(attempt, rp, retry_reasons_final)
            apply_unsharp, unsharp_reason, unsharp_profile = resolve_screen_unsharp(
                rp, attempt, retry_reasons_final
            )
            ff_vf = ffmpeg_vf_for_attempt(rp, attempt, retry_reasons_final)
            impact_use = (
                rp.impact_refine_aggressive
                if attempt == 0
                else second_pass_impact_aggressive(rp, retry_reasons_final, rp.impact_refine_aggressive)
            )

            logger.info(
                "[PRO_V2][SCREEN] id=%s attempt=%s routing_use_deblur=%s unsharp_applied=%s "
                "unsharp_reason=%s unsharp_profile=%s preprocess_relaxed=%.4f ffmpeg_vf=%s impact_agg=%s "
                "legacy_picker_variant=%s retry_reasons_in=%s",
                analysis_id,
                attempt,
                rp.use_deblur,
                "true" if apply_unsharp else "false",
                unsharp_reason,
                unsharp_profile,
                relaxed,
                repr((ff_vf or "")[:100]),
                impact_use,
                pick_dict.get("legacy_picker_variant", 0),
                retry_reasons_final[:10],
            )

            ffmpeg_input_path = input_video_path
            analysis_input = "raw"
            screen_try_dir = work / f"screen_try_{attempt}"
            screen_try_dir.mkdir(parents=True, exist_ok=True)
            try:
                screen = run_pro_v2_screen_preprocess(
                    input_video_path=input_video_path,
                    work_dir=str(screen_try_dir),
                    relaxed_margin=relaxed,
                    apply_unsharp=apply_unsharp,
                    unsharp_profile=unsharp_profile,
                )
                ffmpeg_input_path = str(screen.get("cropped_video_path") or input_video_path)
                if ffmpeg_input_path != input_video_path:
                    screen_cropped_video_path = ffmpeg_input_path
                    analysis_input = "screen_cropped"
            except Exception as exc:
                logger.warning(
                    "[PRO_V2][SCREEN] preprocess_fail attempt=%s relaxed=%s reason=%s",
                    attempt,
                    relaxed,
                    exc,
                )
                ffmpeg_input_path = input_video_path
                analysis_input = "raw"

            logger.info(
                "[PRO_V2][PIPELINE] analysis_input=%s review_pass=%s",
                analysis_input,
                attempt + 1,
            )
            ff, t0, t1, dense, keyframes = _run_motion_pipeline_once(
                ffmpeg_input_path=ffmpeg_input_path,
                work=work,
                rough_impact_time_s=rough_impact_time_s,
                analysis_input=analysis_input,
                picker_tuning=pick_dict,
                impact_refine_aggressive=impact_use,
                analysis_vf_prefix=ff_vf,
                dense_pose_priority=rp.pose_priority,
                dense_club_emphasis=rp.dense_club_emphasis,
                min_dense_frames=rp.min_dense_frames,
                legacy_picker_variant=int(pick_dict.get("legacy_picker_variant", 0)),
            )

            routing_last_pass = {
                "attempt": attempt,
                "analysis_input": analysis_input,
                "relaxed_margin": relaxed,
                "apply_unsharp": apply_unsharp,
                "unsharp_reason": unsharp_reason,
                "unsharp_profile": unsharp_profile,
                "ffmpeg_vf_prefix": ff_vf or "",
                "impact_refine_aggressive": impact_use,
                "picker_tuning": {k: pick_dict.get(k) for k in sorted(pick_dict.keys())},
                "dense_count": len(dense),
            }

            rev = await run_core_keyframe_review_round(
                keyframes,
                review_round=attempt + 1,
                confidence_ceiling=rp.expected_confidence_ceiling,
            )
            review_round_done = attempt + 1
            core_frame_scores = rev["core_frame_scores"]
            retry_reasons_final = list(rev.get("retry_reasons") or [])
            need_retry = bool(rev.get("retry_required"))
            logger.info(
                "[PRO_V2][KF_REVIEW] review_round=%s retry_required=%s analysis_input=%s picker_variant=%s",
                review_round_done,
                need_retry,
                analysis_input,
                pick_dict.get("legacy_picker_variant"),
            )
            if not need_retry:
                break
            logger.info(
                "[PRO_V2][RETRY] round1_failed=%s will_run_second_pass=%s reasons=%s",
                attempt == 0,
                attempt == 0,
                retry_reasons_final[:8],
            )

        all_core_pass = all(
            bool((core_frame_scores.get(k) or {}).get("pass_90")) for k in CORE_PHASE_ORDER
        )
        analysis_trust = "high_trust" if all_core_pass else "low_trust"
        report_mode = "formal" if all_core_pass else "limited"
        keyframe_mismatch_notice = not all_core_pass
        failed_phases = [k for k in CORE_PHASE_ORDER if not (core_frame_scores.get(k) or {}).get("pass_90")]
        if not all_core_pass:
            logger.warning(
                "[PRO_V2][LOW_TRUST] analysis_trust=low_trust report_mode=limited review_round=%s "
                "failed_phases=%s reasons=%s routing_ceiling=%s quality=%s use_deblur=%s heavy_club=%s "
                "pose_priority=%s analysis_input=%s picker_tuning=%s",
                review_round_done,
                failed_phases,
                retry_reasons_final[:12],
                rp.expected_confidence_ceiling,
                rp.quality_level,
                rp.use_deblur,
                rp.use_heavy_club_tracking,
                rp.pose_priority,
                routing_last_pass.get("analysis_input"),
                routing_last_pass.get("picker_tuning"),
            )
        else:
            logger.info(
                "[PRO_V2][LOW_TRUST] analysis_trust=high_trust report_mode=formal review_round=%s "
                "quality=%s ceiling=%s picker_tuning=%s",
                review_round_done,
                rp.quality_level,
                rp.expected_confidence_ceiling,
                routing_last_pass.get("picker_tuning"),
            )
        logger.info(
            "[PRO_V2][REPORT_MODE] report_mode=%s keyframe_mismatch_notice=%s",
            report_mode,
            keyframe_mismatch_notice,
        )

        routing_full = {**(routing_exec_static or {}), "last_pass": routing_last_pass}
        routing_full_out = routing_full
        motion_context = _build_motion_context(
            fps=ff.fps,
            swing_t0=t0,
            swing_t1=t1,
            dense=dense,
            keyframes=keyframes,
            extras=_trust_extras(
                route=route,
                routing_execution=routing_full,
                analysis_input=str(routing_last_pass.get("analysis_input") or analysis_input),
                analysis_trust=analysis_trust,
                report_mode=report_mode,
                review_round=review_round_done,
                core_frame_scores=core_frame_scores,
                keyframe_mismatch_notice=keyframe_mismatch_notice,
                retry_reasons=retry_reasons_final,
            ),
        )
        report = await write_pro_v2_ai_report(motion_context, region=region, report_mode=report_mode)
    else:
        ffmpeg_input_path = input_video_path
        logger.info("[PRO_V2][PIPELINE] analysis_input=raw (non-screen)")
        ff, t0, t1, dense, keyframes = _run_motion_pipeline_once(
            ffmpeg_input_path=ffmpeg_input_path,
            work=work,
            rough_impact_time_s=rough_impact_time_s,
            analysis_input="raw",
            picker_tuning=None,
            impact_refine_aggressive=False,
            analysis_vf_prefix=None,
            dense_pose_priority=False,
            dense_club_emphasis=0.0,
            min_dense_frames=16,
            legacy_picker_variant=0,
        )
        motion_context = _build_motion_context(
            fps=ff.fps,
            swing_t0=t0,
            swing_t1=t1,
            dense=dense,
            keyframes=keyframes,
        )
        report = await write_pro_v2_ai_report(motion_context, region=region, report_mode="formal")
    rmeta = pop_pro_v2_report_meta(report)
    logger.info(
        "[PRO_V2][REPORT] first_pass_weak=%s second_pass_used=%s second_pass_weak=%s fallback_used=%s",
        rmeta.get("pass1_weak"),
        rmeta.get("pass2_used"),
        rmeta.get("pass2_weak"),
        rmeta.get("fallback_used"),
    )

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

    training_plan = _normalize_training_plan(report.get("training_plan"))
    if not training_plan:
        training_plan = _training_plan_fallback(suggestions_zh, suggestions)

    src_frames = _source_frame_count(ff.analysis_240_path)

    minimal: dict[str, Any] = {
        "analysis_id": analysis_id,
        "status": "completed",
        "summary": str(report.get("summary") or ""),
        "summary_zh": str(report.get("summary_zh") or report.get("summary") or ""),
        "total_score": total_score_f,
        "keyframes": keyframes,
        "contact_sheet_url": sheet_path,
        "video_url": input_video_path,
        "original_video_url": input_video_path,
        "playback_video_url": ff.playback_path,
        "issues": issues,
        "issues_zh": issues_zh,
        "suggestions": suggestions,
        "suggestions_zh": suggestions_zh,
        "scores": report.get("scores") or {},
        "type": "pro",
        "training_plan": training_plan,
        "video_meta": {
            "fps": ff.fps,
            "duration_s": ff.duration_s,
            "source_frame_count": src_frames,
        },
        "pro_v2_screen_pipeline": bool(screen_mode),
        "screen_mode": bool(screen_mode),
        "analysis_trust": analysis_trust,
        "report_mode": report_mode,
        "review_round": review_round_done,
        "core_frame_scores": core_frame_scores if screen_mode else {},
        "retry_required": False,
        "retry_reasons": retry_reasons_final if screen_mode else [],
        "keyframe_mismatch_notice": keyframe_mismatch_notice if screen_mode else False,
        "warning": (LIMITED_NOTICE_ZH if screen_mode and keyframe_mismatch_notice else ""),
        "screen_keyframe_review_applied": bool(screen_mode),
    }
    if screen_mode and route:
        minimal["routing_strategy"] = route
    if screen_mode and routing_full_out is not None:
        minimal["routing_execution"] = routing_full_out
    if screen_cropped_video_path:
        minimal["screen_cropped_video_url"] = screen_cropped_video_path

    logger.info("[PRO_V2] done analysis_id=%s kfs=%s", analysis_id, len(keyframes))
    return minimal
