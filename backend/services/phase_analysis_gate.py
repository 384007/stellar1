"""
Strict phase-AI contract: single entry for lite / plus / pro routes.

When ``pass`` is False, routers must return HTTP 422 with
``build_phase_alignment_fail_detail`` — no neutralized / degraded phase narrative.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from services.pose_strict_config import (
    POSE_DRIFT_CLAMP_FRAMES_WARN,
    SWEET_SPOT_CONFIDENCE_LOW,
)

logger = logging.getLogger(__name__)

ERROR_CODE_PHASE_ALIGNMENT = "PHASE_ALIGNMENT_NOT_RELIABLE"

_KEYFRAME_QUALITY_CONTRACT_REASONS = frozenset({
    "NEAR_DUPLICATE_PRESENT",
    "TIME_TOO_CLOSE_PRESENT",
    "DETAIL_VALIDATION_FAILED",
})


def _is_modal_runtime() -> bool:
    return any(
        os.getenv(k)
        for k in (
            "MODAL_TASK_ID",
            "MODAL_CONTAINER_ID",
            "MODAL_IS_SUBTASK",
            "MODAL_REGION",
            "MODAL_ENVIRONMENT",
        )
    )


def _modal_echo_fallback(msg: str) -> None:
    # Keep identical message mirrored to both streams (matches backend.main behavior).
    print(msg, flush=True)
    print(msg, flush=True, file=sys.stderr)


def collect_plus_route_hard_reasons(
    *,
    gate_pass_kf: bool,
    phase_source: str,
    pv_pass: bool,
    sem_ok: bool,
    phase_evaluations_reliable: bool,
    source_frame_gap_reasons: list[str] | None = None,
) -> list[str]:
    """Machine-readable codes for Plus-only pipeline blocks (merged into 422 ``detail.reasons``)."""
    out: list[str] = []
    if not gate_pass_kf:
        out.append("KEYFRAME_GATE_FAIL")
    if "degraded" in str(phase_source):
        out.append("PHASE_SOURCE_DEGRADED")
    if not pv_pass:
        out.append("PHASE_VALIDATION_FAIL")
    if not sem_ok:
        out.append("SEMANTIC_FAIL")
    if not phase_evaluations_reliable:
        out.append("PHASE_EVAL_UNRELIABLE")
    for sg in source_frame_gap_reasons or []:
        s = str(sg)
        if s.startswith("MIN_GAP_VIOLATION") and s not in out:
            out.append(s)
    return out


def should_run_phase_analysis_strict(
    *,
    pose_quality_bundle: dict[str, Any],
    sem_report: dict[str, Any],
    kf_validation: dict[str, Any],
    keyframe_count: int,
    ai_vision_count: int,
    gemini_assess: dict[str, Any] | None = None,
    sweet_spot_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Returns dict with ``pass: bool`` and populated diagnostic fields for 422 responses.
    """
    reasons: list[str] = []
    trace: dict[str, Any] = {}

    rel = str(pose_quality_bundle.get("pose_reliability_level") or "low")
    trace["pose_reliability_level"] = rel
    if rel == "low":
        reasons.append("POSE_RELIABILITY_LOW")
        trace["pose_gate"] = False
    else:
        trace["pose_gate"] = True

    strict_sem = bool(sem_report.get("final_phase_semantic_ok_strict"))
    trace["final_phase_semantic_ok_strict"] = strict_sem
    trace["align_top"] = sem_report.get("align_top")
    trace["align_impact"] = sem_report.get("align_impact")
    trace["keyframe_semantic_ok"] = sem_report.get("keyframe_semantic_ok")
    trace["phase_validation_passed"] = sem_report.get("phase_validation_passed")
    trace["final_phase_semantic_ok_strict_reasons"] = list(
        sem_report.get("final_phase_semantic_ok_strict_reasons") or [],
    )
    if not strict_sem:
        reasons.append("PHASE_SEMANTIC_OR_ALIGNMENT_STRICT_FAIL")
        trace["semantic_strict_gate"] = False
        for code in trace["final_phase_semantic_ok_strict_reasons"]:
            if code and code not in reasons:
                reasons.append(str(code))
    else:
        trace["semantic_strict_gate"] = True

    gate_pass = bool(kf_validation.get("final_keyframe_gate_pass"))
    fv = kf_validation.get("final_keyframe_validation")
    if not isinstance(fv, dict):
        fv = {}
    strict_kf = bool(fv.get("strict_contract_ok", False))
    semantic_strip_ok = bool(fv.get("semantic_strip_ok", False))
    phase_strip_semantic_reasons = list(fv.get("semantic_strip_reasons") or [])
    relabel_cnt = int(fv.get("relabel_count") or 0)
    trace["relabel_count"] = relabel_cnt
    trace["rebuild_used"] = bool(fv.get("rebuild_used")) or bool(kf_validation.get("phase_strip_repaired"))
    trace["phase_strip_repaired"] = bool(kf_validation.get("phase_strip_repaired"))
    trace["rebuild_reasons"] = list(fv.get("rebuild_reasons") or [])
    trace["top_reselected"] = bool(fv.get("top_reselected"))
    trace["impact_reselected"] = bool(fv.get("impact_reselected"))
    trace["final_keyframe_gate_pass"] = gate_pass
    trace["strict_contract_ok"] = strict_kf
    strict_contract_checks = fv.get("strict_contract_checks")
    if not isinstance(strict_contract_checks, dict):
        strict_contract_checks = {}
    strict_contract_fail_reasons = list(fv.get("strict_contract_fail_reasons") or [])
    trace["strict_contract_checks"] = dict(strict_contract_checks)
    trace["strict_contract_fail_reasons"] = list(strict_contract_fail_reasons)
    trace["near_duplicates"] = int(fv.get("near_duplicates") or 0)
    trace["time_too_close_count"] = int(fv.get("time_too_close_count") or 0)
    trace["late_strip_cleanup_applied"] = bool(fv.get("late_strip_cleanup_applied"))
    trace["late_strip_cleanup_resolved"] = bool(fv.get("late_strip_cleanup_resolved"))
    trace["remaining_near_duplicate_phases"] = list(fv.get("remaining_near_duplicate_phases") or [])
    trace["impact_preserved"] = fv.get("impact_preserved")
    trace["impact_shift_frames"] = fv.get("impact_shift_frames")
    trace["follow_shift_frames"] = fv.get("follow_shift_frames")
    trace["finish_shift_frames"] = fv.get("finish_shift_frames")
    trace["late_strip_cleanup_rounds"] = fv.get("late_strip_cleanup_rounds")
    trace["late_strip_cleanup_changed_phase_ids"] = list(fv.get("late_strip_cleanup_changed_phase_ids") or [])
    trace["late_strip_cleanup_improved"] = fv.get("late_strip_cleanup_improved")
    trace["late_strip_cleanup_reverted"] = fv.get("late_strip_cleanup_reverted")
    trace["late_strip_cleanup_pass"] = fv.get("late_strip_cleanup_pass")
    trace["late_strip_cleanup_reason"] = fv.get("late_strip_cleanup_reason")
    trace["late_strip_cleanup_accepted_by_service"] = fv.get("late_strip_cleanup_accepted_by_service")
    trace["strip_detail_any_failed"] = bool(fv.get("strip_detail_any_failed"))
    trace["source_frame_gaps_ok"] = bool(fv.get("source_frame_gaps_ok", True))
    trace["source_frame_gap_reasons"] = list(fv.get("source_frame_gap_reasons") or [])
    trace["adjacent_strip_hard_dup_reasons"] = list(
        fv.get("adjacent_strip_hard_dup_reasons") or [],
    )
    fr_set = frozenset(str(x) for x in strict_contract_fail_reasons if x)
    if fr_set and fr_set <= _KEYFRAME_QUALITY_CONTRACT_REASONS and not strict_kf:
        trace["keyframe_quality_repair_attempted"] = bool(
            kf_validation.get("keyframe_quality_repair_attempted"),
        )
        trace["keyframe_quality_repair_success"] = bool(
            kf_validation.get("keyframe_quality_repair_success"),
        )
    trace["phase_strip_semantic_ok"] = semantic_strip_ok
    trace["phase_strip_semantic_reasons"] = phase_strip_semantic_reasons
    if relabel_cnt > 1:
        reasons.append("PHASE_STRIP_REPAIR_FAILED_RELABEL_FORBIDDEN")
    if not strict_kf:
        reasons.append("KEYFRAME_STRICT_CONTRACT_FAIL")
        for code in strict_contract_fail_reasons:
            if code and str(code) not in reasons:
                reasons.append(str(code))
    if not semantic_strip_ok:
        reasons.append("PHASE_STRIP_SEMANTIC_ORDER_FAIL")
    trace["keyframe_gate"] = bool(
        strict_kf and semantic_strip_ok and gate_pass and relabel_cnt <= 1
    )

    exp = 8
    if keyframe_count < exp or ai_vision_count < exp:
        reasons.append("INCOMPLETE_PHASE_STRIP")
        trace["strip_complete"] = False
    else:
        trace["strip_complete"] = True

    src = str(kf_validation.get("final_keyframe_source") or "")
    trace["final_keyframe_source"] = src
    if src == "ordered_fallback":
        reasons.append("ORDERED_FALLBACK_BLOCKED")
        trace["source_allowed"] = False
    else:
        trace["source_allowed"] = True

    ga = gemini_assess or {}
    if bool(ga.get("gemini_uniform_thumbnail_map_applies")):
        aligned = ga.get("gemini_map_aligned_with_final_strip")
        trace["gemini_uniform_map_applies"] = True
        trace["gemini_map_aligned_with_final_strip"] = aligned
        if aligned is not True:
            reasons.append("GEMINI_THUMB_MAP_DIVERGED")
    else:
        trace["gemini_uniform_map_applies"] = False

    if bool(sem_report.get("phase_reselection_failed")):
        reasons.append("PHASE_RESELECTION_FAILED")
        trace["phase_reselection_failed"] = True
    else:
        trace["phase_reselection_failed"] = False

    rel_codes = list(pose_quality_bundle.get("reliability_reason_codes") or [])
    post_rep = pose_quality_bundle.get("pose_quality_report_post") or {}
    clamps = int(post_rep.get("wrist_jump_clamped_count") or 0)
    trace["wrist_jump_clamped_count"] = clamps
    if "SMOOTHING_LAG_HIGH" in rel_codes or clamps >= POSE_DRIFT_CLAMP_FRAMES_WARN:
        reasons.append("POSE_TRACK_DRIFT_HIGH")
        trace["pose_track_drift_flag"] = True
    else:
        trace["pose_track_drift_flag"] = False

    ss = sweet_spot_bundle or {}
    ss_conf = float(ss.get("sweet_spot_confidence") or 1.0)
    ss_unstable = bool(ss.get("sweet_spot_unstable"))
    ss_reasons = list(ss.get("sweet_spot_reasons") or [])
    trace["sweet_spot_confidence"] = ss_conf
    trace["sweet_spot_unstable"] = ss_unstable
    trace["sweet_spot_reasons"] = ss_reasons
    # Soft signal only: sweet-spot quality lowers analysis_reliability via cap_confidence, not 422.
    trace["sweet_spot_warning"] = bool(
        ss_unstable
        or ss_conf < SWEET_SPOT_CONFIDENCE_LOW
        or any("SWEET_SPOT_UNSTABLE" in str(r) for r in ss_reasons)
    )

    def _add_reason_code(code: str) -> None:
        if code and code not in reasons:
            reasons.append(code)

    if bool(sem_report.get("dt_axis_invalid")):
        _add_reason_code("DT_AXIS_INVALID")
    if bool(sem_report.get("non_finite_kinematics")):
        _add_reason_code("NON_FINITE_KINEMATICS")
    for surf in phase_strip_semantic_reasons:
        strip_code = str(surf)
        if strip_code in (
            "PHASE_POSE_INDEX_NOT_INCREASING",
            "PHASE_STRIP_PHASE_ORDER_MISMATCH_AFTER_POSE_SORT",
            "DT_AXIS_INVALID",
            "NON_FINITE_KINEMATICS",
        ) or strip_code.startswith("MIN_GAP_VIOLATION") or strip_code.startswith("ADJACENT_STRIP_"):
            _add_reason_code(strip_code)

    trace["phase_strip_repaired"] = bool(fv.get("phase_strip_repaired"))
    if src in {"ordered_fallback_repaired", "smart_repaired", "smart_repaired_best_effort", "smart_rebuilt_best_effort"}:
        trace["rebuild_used"] = True
        trace["phase_strip_repaired"] = True
    trace["reselected_top"] = bool(fv.get("reselected_top"))
    trace["reselected_impact"] = bool(fv.get("reselected_impact"))
    trace["dt_fixed_count"] = int(fv.get("dt_fixed_count") or 0)
    trace["repair_log"] = list(fv.get("repair_log") or [])

    if not semantic_strip_ok and (
        bool(fv.get("phase_strip_repaired")) or fv.get("enforce_ok") is False
    ):
        _add_reason_code("PHASE_STRIP_REPAIR_FAILED")

    phase_alignment_metrics = {
        "align_tol": sem_report.get("align_tol"),
        "top_abs_err": sem_report.get("top_abs_err"),
        "impact_abs_err": sem_report.get("impact_abs_err"),
        "top_abs_err_after": sem_report.get("top_abs_err_after"),
        "impact_abs_err_after": sem_report.get("impact_abs_err_after"),
        "final_phase_semantic_ok_strict": strict_sem,
        "keyframe_semantic_ok": sem_report.get("keyframe_semantic_ok"),
        "semantic_validation": sem_report.get("semantic_validation"),
        "phase_reselection_failed": sem_report.get("phase_reselection_failed"),
        "rebuild_used": sem_report.get("rebuild_used"),
        "fail_code": sem_report.get("fail_code"),
        "align_top": sem_report.get("align_top"),
        "align_impact": sem_report.get("align_impact"),
        "phase_validation_passed": sem_report.get("phase_validation_passed"),
        "final_phase_semantic_ok_strict_reasons": list(
            sem_report.get("final_phase_semantic_ok_strict_reasons") or [],
        ),
        "sweet_spot_confidence": ss_conf,
        "sweet_spot_valid_frames": ss.get("sweet_spot_valid_frames"),
        "sweet_spot_warning": bool(trace.get("sweet_spot_warning")),
        "phase_strip_semantic_ok": bool(semantic_strip_ok),
        "phase_strip_semantic_reasons": list(phase_strip_semantic_reasons),
        "dt_axis_invalid": bool(sem_report.get("dt_axis_invalid")),
        "non_finite_kinematics": bool(sem_report.get("non_finite_kinematics")),
    }

    pose_quality_report = pose_quality_bundle.get("pose_quality_report")
    post_rep_out = pose_quality_bundle.get("pose_quality_report_post")

    passed = len(reasons) == 0
    # Keep a concise summary line for every request so Modal logs always show gate activity.
    logger.info(
        "[phase_strict_gate] %s strict_contract_ok=%s semantic_strip_ok=%s reasons=%s",
        "PASS" if passed else "FAIL",
        strict_kf,
        semantic_strip_ok,
        reasons if not passed else [],
    )
    if not passed:
        logger.warning(
            "[phase_strict_gate] FAIL reasons=%s trace=%s",
            reasons,
            trace,
        )
        if _is_modal_runtime():
            _modal_echo_fallback(
                f"[phase_strict_gate] FAIL reasons={reasons} strict_contract_ok={strict_kf} semantic_strip_ok={semantic_strip_ok}",
            )

    return {
        "pass": passed,
        "error_code": None if passed else ERROR_CODE_PHASE_ALIGNMENT,
        "reasons": list(reasons),
        "pose_quality_report": pose_quality_report,
        "pose_quality_report_post": post_rep_out,
        "pose_reliability_level": rel,
        "reliability_reason_codes": list(pose_quality_bundle.get("reliability_reason_codes") or []),
        "gate_decision_trace": trace,
        "phase_alignment_metrics": phase_alignment_metrics,
        "sweet_spot_bundle": sweet_spot_bundle,
        "phase_strip_semantic_ok": bool(semantic_strip_ok),
        "phase_strip_semantic_reasons": list(phase_strip_semantic_reasons),
    }


def build_phase_alignment_fail_detail(decision: dict[str, Any]) -> dict[str, Any]:
    """Stable JSON body for HTTP 422 ``detail`` (machine-readable)."""
    tr = decision.get("gate_decision_trace") or {}
    return {
        "error": True,
        "error_code": decision.get("error_code") or ERROR_CODE_PHASE_ALIGNMENT,
        "reasons": list(decision.get("reasons") or []),
        "pose_quality_report": decision.get("pose_quality_report"),
        "pose_quality_report_post": decision.get("pose_quality_report_post"),
        "pose_reliability_level": decision.get("pose_reliability_level"),
        "reliability_reason_codes": decision.get("reliability_reason_codes"),
        "gate_decision_trace": decision.get("gate_decision_trace"),
        "phase_alignment_metrics": decision.get("phase_alignment_metrics"),
        "sweet_spot_bundle": decision.get("sweet_spot_bundle"),
        "phase_strip_semantic_ok": decision.get("phase_strip_semantic_ok"),
        "phase_strip_semantic_reasons": list(decision.get("phase_strip_semantic_reasons") or []),
        "repair_log": list(tr.get("repair_log") or []),
        "relabel_count": tr.get("relabel_count"),
        "rebuild_used": tr.get("rebuild_used"),
        "rebuild_reasons": list(tr.get("rebuild_reasons") or []),
        "top_reselected": tr.get("top_reselected"),
        "impact_reselected": tr.get("impact_reselected"),
        "strict_contract_checks": tr.get("strict_contract_checks"),
        "strict_contract_fail_reasons": list(tr.get("strict_contract_fail_reasons") or []),
        "near_duplicates": tr.get("near_duplicates"),
        "time_too_close_count": tr.get("time_too_close_count"),
        "late_strip_cleanup_applied": tr.get("late_strip_cleanup_applied"),
        "late_strip_cleanup_resolved": tr.get("late_strip_cleanup_resolved"),
        "remaining_near_duplicate_phases": list(tr.get("remaining_near_duplicate_phases") or []),
        "impact_preserved": tr.get("impact_preserved"),
        "impact_shift_frames": tr.get("impact_shift_frames"),
        "follow_shift_frames": tr.get("follow_shift_frames"),
        "finish_shift_frames": tr.get("finish_shift_frames"),
        "late_strip_cleanup_rounds": tr.get("late_strip_cleanup_rounds"),
        "late_strip_cleanup_changed_phase_ids": list(tr.get("late_strip_cleanup_changed_phase_ids") or []),
        "late_strip_cleanup_improved": tr.get("late_strip_cleanup_improved"),
        "late_strip_cleanup_reverted": tr.get("late_strip_cleanup_reverted"),
        "late_strip_cleanup_pass": tr.get("late_strip_cleanup_pass"),
        "late_strip_cleanup_reason": tr.get("late_strip_cleanup_reason"),
        "late_strip_cleanup_accepted_by_service": tr.get("late_strip_cleanup_accepted_by_service"),
        "strip_detail_any_failed": tr.get("strip_detail_any_failed"),
        "source_frame_gaps_ok": tr.get("source_frame_gaps_ok"),
        "source_frame_gap_reasons": list(tr.get("source_frame_gap_reasons") or []),
        "adjacent_strip_hard_dup_reasons": list(tr.get("adjacent_strip_hard_dup_reasons") or []),
    }
