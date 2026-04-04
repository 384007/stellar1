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
from services.pro_v2_keyframe_visual_gate_service import run_keyframe_visual_diversity_gate
from services.pro_v2_final_keyframe_render_service import render_display_keyframes_from_sources
from services.pro_v2_report_service import LIMITED_NOTICE_ZH, pop_pro_v2_report_meta, write_pro_v2_ai_report
from services.pro_v2_screen_trust_gates import evaluate_dense_motion_health, evaluate_screen_roi_health
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
    return analysis_input in ("screen_cropped", "screen_clean")


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
    screen_clean_video_path: str | None = None
    review_round_done = 0
    core_frame_scores: dict[str, Any] = {}
    retry_reasons_final: list[str] = []
    analysis_trust = "high_trust"
    report_mode = "formal"
    keyframe_mismatch_notice = False

    routing_last_pass: dict[str, Any] = {}
    routing_exec_static: dict[str, Any] | None = None
    routing_full_out: dict[str, Any] | None = None
    trust_structural_ok = True
    trust_core_ai_all_pass = True
    pro_v2_debug_payload = None
    screen_keyframe_audit = None

    if screen_mode:
        route = await run_pro_v2_ai_routing(input_video_path, screen_mode_requested=True)
        rp = derive_routing_execution(route)
        log_route_apply(rp, analysis_id=analysis_id)
        logger.info("[PRO_V2][ROUTE_APPLY] analysis_ts=%s route=%s", analysis_id, route)
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
        final_screen_bundle_at_pass: dict[str, Any] | None = None

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
            iteration_screen_bundle: dict[str, Any] | None = None
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
                screen_clean_video_path = str(screen.get("clean_video_path") or "")
                ffmpeg_input_path = str(screen.get("clean_video_path") or screen.get("cropped_video_path") or input_video_path)
                if ffmpeg_input_path != input_video_path:
                    screen_cropped_video_path = str(screen.get("cropped_video_path") or "")
                    analysis_input = "screen_clean" if screen.get("clean_video_path") else "screen_cropped"
                    sz = screen.get("source_frame_size") or {}
                    fw = int(sz.get("w") or 0)
                    fh = int(sz.get("h") or 0)
                    conf = float(screen.get("confidence") or 0.0)
                    roi_ev = evaluate_screen_roi_health(screen.get("crop_box"), fw, fh, conf)
                    iteration_screen_bundle = {**screen, "roi_health_eval": roi_ev}
                    logger.info(
                        "[PRO_V2][SCREEN_DEBUG] attempt=%s crop=%s src_size=%sx%s roi_pass=%s roi_reasons=%s",
                        attempt,
                        screen.get("crop_box"),
                        fw,
                        fh,
                        roi_ev.get("passed"),
                        roi_ev.get("reason_codes"),
                    )
                    logger.info(
                        "[PRO_V2][SCREEN_CLEAN] analysis_ts=%s attempt=%s clean=%s cropped=%s",
                        analysis_id,
                        attempt,
                        screen_clean_video_path,
                        screen_cropped_video_path,
                    )
            except Exception as exc:
                logger.warning(
                    "[PRO_V2][SCREEN] preprocess_fail attempt=%s relaxed=%s reason=%s",
                    attempt,
                    relaxed,
                    exc,
                )
                ffmpeg_input_path = input_video_path
                analysis_input = "raw"
                iteration_screen_bundle = None

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
            keyframes, display_source_kind, render_missing_reasons = render_display_keyframes_from_sources(
                keyframes,
                screen_clean_video_path=screen_clean_video_path,
                screen_cropped_video_path=screen_cropped_video_path,
                raw_video_path=input_video_path,
            )
            for _k in keyframes:
                logger.info(
                    "[PRO_V2][KEYFRAME_SOURCE] analysis_ts=%s phase=%s analysis_ts_s=%.4f source_kind=%s source_ts_s=%.4f source_frame_index=%s picker_tuning=%s",
                    analysis_id,
                    _k.get("phase"),
                    float(_k.get("timestamp") or 0.0),
                    _k.get("display_source_kind"),
                    float(_k.get("display_source_timestamp") or 0.0),
                    _k.get("display_source_frame_index"),
                    routing_last_pass.get("picker_tuning"),
                )
            routing_last_pass["display_source_kind"] = display_source_kind
            if render_missing_reasons:
                retry_reasons_final = list(dict.fromkeys(list(retry_reasons_final) + render_missing_reasons))

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
                "[PRO_V2][KF_REVIEW] review_round=%s retry_required=%s analysis_input=%s display_source=%s picker_variant=%s",
                review_round_done,
                need_retry,
                analysis_input,
                display_source_kind,
                pick_dict.get("legacy_picker_variant"),
            )
            final_screen_bundle_at_pass = iteration_screen_bundle
            if not need_retry:
                break
            logger.info(
                "[PRO_V2][RETRY] round1_failed=%s will_run_second_pass=%s reasons=%s",
                attempt == 0,
                attempt == 0,
                retry_reasons_final[:8],
            )

        visual_gate = run_keyframe_visual_diversity_gate(keyframes)
        dense_health = evaluate_dense_motion_health(
            dense, t0, t1, ff.duration_s, screen_mode=True
        )
        logger.info(
            "[PRO_V2][DENSE_DEBUG] dense_count=%s passed=%s reasons=%s swing_window=(%.4f,%.4f)",
            len(dense),
            dense_health.get("passed"),
            dense_health.get("reason_codes"),
            t0,
            t1,
        )

        # Re-resolve ROI for the same pass as routing_last_pass (final keyframes)
        _ai = str(routing_last_pass.get("analysis_input") or "")
        if _ai in ("screen_cropped", "screen_clean") and final_screen_bundle_at_pass:
            roi_health: dict[str, Any] = dict(final_screen_bundle_at_pass.get("roi_health_eval") or {})
        else:
            roi_health = {
                "passed": False,
                "reason_codes": ["SCREEN_PIPELINE_NOT_CROPPED"],
                "area_ratio": 0.0,
                "detection_confidence": 0.0,
                "crop_box": None,
                "center_norm": None,
                "source_size": {"w": 0, "h": 0},
            }
            logger.info(
                "[PRO_V2][SCREEN_DEBUG] roi_health forced_fail analysis_input=%s had_bundle=%s",
                _ai,
                final_screen_bundle_at_pass is not None,
            )

        structural_ok = (
            bool(roi_health.get("passed"))
            and bool(dense_health.get("passed"))
            and bool(visual_gate.get("passed"))
            and not bool(visual_gate.get("duplicate_pairs"))
        )
        gate_reason_codes = list(
            dict.fromkeys(
                list(roi_health.get("reason_codes") or [])
                + list(dense_health.get("reason_codes") or [])
                + list(visual_gate.get("reason_codes") or [])
            )
        )

        all_core_pass = all(
            bool((core_frame_scores.get(k) or {}).get("pass_90")) for k in CORE_PHASE_ORDER
        )
        formal_allowed = bool(all_core_pass and structural_ok)
        analysis_trust = "high_trust" if formal_allowed else "low_trust"
        report_mode = "formal" if formal_allowed else "limited"
        keyframe_mismatch_notice = not formal_allowed
        retry_reasons_final = list(dict.fromkeys(list(retry_reasons_final) + gate_reason_codes))

        failed_phases = [k for k in CORE_PHASE_ORDER if not (core_frame_scores.get(k) or {}).get("pass_90")]
        if not formal_allowed:
            logger.warning(
                "[PRO_V2][LOW_TRUST] analysis_trust=%s report_mode=%s formal_allowed=%s structural_ok=%s "
                "all_core_ai_pass=%s review_round=%s failed_phases=%s gate_reasons=%s ai_reasons=%s "
                "dup_pairs=%s routing_ceiling=%s quality=%s analysis_input=%s",
                analysis_trust,
                report_mode,
                formal_allowed,
                structural_ok,
                all_core_pass,
                review_round_done,
                failed_phases,
                gate_reason_codes[:16],
                [r for r in retry_reasons_final if r not in gate_reason_codes][:8],
                (visual_gate.get("duplicate_pairs") or [])[:6],
                rp.expected_confidence_ceiling,
                rp.quality_level,
                routing_last_pass.get("analysis_input"),
            )
        else:
            logger.info(
                "[PRO_V2][LOW_TRUST] analysis_trust=high_trust formal_allowed=true structural_ok=true "
                "review_round=%s quality=%s ceiling=%s",
                review_round_done,
                rp.quality_level,
                rp.expected_confidence_ceiling,
            )

        screen_keyframe_audit = {
            "structural_gates_passed": structural_ok,
            "all_core_ai_pass_90": all_core_pass,
            "roi_passed": bool(roi_health.get("passed")),
            "dense_motion_passed": bool(dense_health.get("passed")),
            "visual_gate_passed": bool(visual_gate.get("passed")),
            "formal_report_allowed": formal_allowed,
            "reason_codes": gate_reason_codes,
            "duplicate_pairs": visual_gate.get("duplicate_pairs") or [],
            "summary_zh": (
                "关键帧审核未通过，结论受限。"
                if not formal_allowed
                else "关键帧结构与视觉校验通过，且核心帧 AI 达门槛。"
            ),
            "summary_en": (
                "Keyframe audit failed; conclusions are limited."
                if not formal_allowed
                else "Structural and visual keyframe checks passed."
            ),
        }
        pro_v2_debug_payload = {
            "analysis_240_path": ff.analysis_240_path,
            "playback_path": ff.playback_path,
            "swing_window_s": [round(t0, 4), round(t1, 4)],
            "analysis_input": routing_last_pass.get("analysis_input"),
            "screen_preprocess": {
                "crop_box": (final_screen_bundle_at_pass or {}).get("crop_box"),
                "confidence": (final_screen_bundle_at_pass or {}).get("confidence"),
                "source_frame_size": (final_screen_bundle_at_pass or {}).get("source_frame_size"),
                "screen_clean_video_path": screen_clean_video_path,
                "roi_health": roi_health,
            },
            "dense_motion_health": dense_health,
            "keyframe_visual_gate": visual_gate,
            "keyframes_lineup": [
                {
                    "phase": k.get("phase"),
                    "analysis_timestamp": k.get("timestamp"),
                    "frame_index": k.get("frame_index"),
                    "display_source_kind": k.get("display_source_kind"),
                    "display_source_timestamp": k.get("display_source_timestamp"),
                    "display_source_frame_index": k.get("display_source_frame_index"),
                    "display_render_ok": k.get("display_render_ok"),
                    "display_render_error": k.get("display_render_error"),
                }
                for k in keyframes
            ],
            "picker_tuning": routing_last_pass.get("picker_tuning"),
            "trust_gate_reason_codes": gate_reason_codes,
        }
        trust_structural_ok = structural_ok
        trust_core_ai_all_pass = all_core_pass
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
        motion_context["structural_gates_passed"] = structural_ok
        motion_context["structural_gate_reason_codes"] = gate_reason_codes[:12]
        motion_context["analysis_keyframes"] = [
            {
                "phase": k.get("phase"),
                "analysis_timestamp": float(k.get("timestamp") or 0.0),
                "analysis_frame_index": int(k.get("frame_index") or 0),
            }
            for k in keyframes
        ]
        motion_context["display_keyframes"] = [
            {
                "phase": k.get("phase"),
                "display_source_kind": k.get("display_source_kind"),
                "display_source_timestamp": float(k.get("display_source_timestamp") or 0.0),
                "display_source_frame_index": int(k.get("display_source_frame_index") or -1),
                "display_render_ok": bool(k.get("display_render_ok")),
                "display_render_error": str(k.get("display_render_error") or ""),
            }
            for k in keyframes
        ]
        hard_stop = bool(not formal_allowed and (
            any(str(c).startswith("SCREEN_ROI_") or str(c).startswith("DENSE_") for c in gate_reason_codes)
            or "KEYFRAME_VISUAL_DUPLICATE" in gate_reason_codes
            or "LATE_STRIP_COLLAPSED" in gate_reason_codes
            or "KEYFRAME_INDEX_COLLAPSE" in gate_reason_codes
            or bool(visual_gate.get("duplicate_pairs"))
        ))
        if hard_stop:
            logger.warning("[PRO_V2][REPORT_HARD_STOP] analysis_id=%s reasons=%s", analysis_id, gate_reason_codes[:14])
            report = {
                "total_score": 0,
                "scores": {"grip": 0, "stance": 0, "backswing": 0, "downswing": 0, "follow_through": 0},
                "issues": ["Keyframe structural/visual validation failed; conclusions are limited."],
                "issues_zh": ["关键帧未通过结构与视觉校验，结论受限。"],
                "suggestions": ["Please re-upload a clearer clip or disable Screen Mode and retry."],
                "suggestions_zh": ["请重新上传更清晰视频，或关闭 Screen Mode 后重试。"],
                "summary": "Key frames did not pass verification; conclusions are limited.",
                "summary_zh": "关键帧不符，结论受限。关键帧未通过结构与视觉校验，结论受限。",
                "training_plan": {},
                "ai_provider": "pro_v2_hard_stop",
            }
        else:
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
        "retry_required": bool(screen_mode and (not trust_core_ai_all_pass)),
        "retry_reasons": retry_reasons_final if screen_mode else [],
        "keyframe_mismatch_notice": keyframe_mismatch_notice if screen_mode else False,
        "warning": "",
        "screen_keyframe_review_applied": bool(screen_mode),
    }
    if screen_mode and keyframe_mismatch_notice:
        wline = LIMITED_NOTICE_ZH
        if not trust_structural_ok:
            wline += "（ROI/运动曲线/视觉去重未通过）"
        elif not trust_core_ai_all_pass:
            wline += "（核心帧 AI 未全部≥90）"
        minimal["warning"] = wline
    if screen_mode and route:
        minimal["routing_strategy"] = route
    if screen_mode and routing_full_out is not None:
        minimal["routing_execution"] = routing_full_out
    if screen_cropped_video_path:
        minimal["screen_cropped_video_url"] = screen_cropped_video_path
    if screen_clean_video_path:
        minimal["screen_clean_video_url"] = screen_clean_video_path
    if screen_mode and pro_v2_debug_payload is not None:
        minimal["pro_v2_debug"] = {**pro_v2_debug_payload, "contact_sheet_path": sheet_path}
    if screen_mode and screen_keyframe_audit is not None:
        minimal["screen_keyframe_audit"] = screen_keyframe_audit

    logger.info("[PRO_V2] done analysis_id=%s kfs=%s", analysis_id, len(keyframes))
    return minimal
