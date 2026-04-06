"""Bridge: run the Prov3 true-240 A/B keyframe product chain for Plus (no HTTP to /pro-v3).

Plus keeps its own pose/HUD/skeleton path; this module is the **authoritative** source for
product keyframes, trust/semantic gate, and formal-score eligibility. The legacy Plus smart
keyframe pipeline in ``keyframe_service`` is compatibility-only and must not drive formal output.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from lib.prov3.keyframes.decode_spacing import spread_keyframes_for_preview_strip
from services.internal.prov3_ffmpeg import ffprobe_video_meta
from services.pro_prov3_analyze_service import (
    ANALYSIS_FPS,
    _build_ui_keyframes,
    _semantic_acceptance_gate,
)
from services.prov3_keyframe_orchestrator_service import run_keyframe_analyze

logger = logging.getLogger(__name__)

PHASE_ORDER = (
    "address",
    "takeaway",
    "backswing",
    "top",
    "downswing",
    "impact",
    "follow_through",
    "finish",
)


def _nearest_pose_for_timestamp(poses: list[dict], t_s: float) -> tuple[int, int]:
    """Return (pose_index, source_frame_index) closest to ``t_s`` seconds."""
    if not poses:
        return 0, 0
    best_i = 0
    best_d = 1e18
    for i, p in enumerate(poses):
        pt = float(p.get("timestamp") or 0.0)
        d = abs(pt - t_s)
        if d < best_d:
            best_d = d
            best_i = i
    sfi = int(poses[best_i].get("frame_index", best_i))
    return best_i, sfi


def _plus_rows_from_ui_strip(
    ui_rows: list[dict[str, Any]],
    poses: list[dict],
    *,
    preview_fallback: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in ui_rows:
        phase = str(row.get("phase") or "")
        ts = float(row.get("timestamp") or 0.0)
        spi, sfi = _nearest_pose_for_timestamp(poses, ts)
        pose_snap = poses[spi] if 0 <= spi < len(poses) else None
        b64 = str(row.get("image_base64") or "").strip()
        out.append(
            {
                "phase": phase,
                "label_en": row.get("label_en", ""),
                "label_zh": row.get("label_zh", ""),
                "frame_index": sfi,
                "timestamp": ts,
                "confidence": row.get("confidence"),
                "selection_reason": "prov3_preview_strip" if preview_fallback else "prov3_true240",
                "fallback_used": bool(preview_fallback),
                "image_base64": b64,
                "pose_snapshot": pose_snap,
                "width": 320,
                "height": 320,
                "source_pose_idx": spi,
                "source_frame_index": sfi,
            }
        )
    return out


def _phase_map_from_plus_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    m: dict[str, int] = {}
    for r in rows:
        ph = str(r.get("phase") or "").strip()
        if not ph:
            continue
        spi = r.get("source_pose_idx")
        if isinstance(spi, int) and spi >= 0:
            m[ph] = spi
    return m


def _span_timeline_frames(raw_kfs: list[dict[str, Any]], av_path: str) -> int:
    fi_max = max((int(k.get("frame_index") or 0) for k in raw_kfs), default=0)
    try:
        meta = ffprobe_video_meta(av_path)
        nb_pf = int(meta.get("nb_frames") or 0)
        dur_pf = float(meta.get("duration_s") or 0.0)
        fps_pf = float(meta.get("fps") or 240.0)
        if nb_pf <= 0 and dur_pf > 0 and fps_pf > 1e-6:
            nb_pf = max(1, int(round(dur_pf * fps_pf)))
        return max(nb_pf, fi_max + 1, 1)
    except Exception:
        return max(fi_max + 1, 1)


def run_plus_prov3_keyframe_bridge(
    input_video_path: str,
    work_dir: str,
    poses: list[dict],
    *,
    screen_mode: bool = False,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Sync: full Prov3 preprocess + A/B + true-240 thumbnails; normalize for Plus router."""
    work_dir = str(work_dir).strip()
    if not work_dir:
        raise RuntimeError("plus_prov3_bridge:empty_work_dir")

    prov3 = run_keyframe_analyze(
        input_video_path,
        work_dir,
        screen_mode=screen_mode,
        cancel_check=cancel_check,
    )
    dumped = prov3.model_dump(exclude={"analysis_video", "analysis_fps", "source_fps"})
    raw_kfs = [dict(x) for x in (dumped.get("keyframes") or [])]

    av_path = str(prov3.analysis_video or "").strip()
    if not av_path:
        raise RuntimeError("analysis_video_missing: true240 analysis video is required")
    if not os.path.isfile(av_path):
        raise RuntimeError("analysis_video_missing: true240 analysis video is required")
    analysis_fps = float(prov3.analysis_fps or ANALYSIS_FPS)

    trust_raw = str(dumped.get("trust_level") or "low")
    final_status = str(dumped.get("status") or "low_trust")
    fail_reasons = sorted(set(str(x) for x in (dumped.get("fail_reasons") or [])))

    ui_official = _build_ui_keyframes(raw_kfs, av_path, analysis_fps)
    span = _span_timeline_frames(raw_kfs, av_path)
    preview_raw = spread_keyframes_for_preview_strip(raw_kfs, span)
    ui_preview = _build_ui_keyframes(preview_raw, av_path, analysis_fps)

    semantic_ok, semantic_fails = _semantic_acceptance_gate(ui_official)
    logger.info(
        "[PLUS_PROV3_BRIDGE][semantic_gate] ok=%s fails=%s",
        int(semantic_ok),
        semantic_fails,
    )
    if not semantic_ok:
        final_status = "low_trust"
        trust_raw = "low"
        fail_reasons = sorted(set([*fail_reasons, *semantic_fails]))

    analysis_trust = {
        "high": "high_trust",
        "medium": "medium_trust",
        "low": "low_trust",
    }.get(trust_raw, "low_trust")

    low_trust_preview_only = (final_status != "pass") or (analysis_trust == "low_trust")
    if low_trust_preview_only and "low_trust_preview_only" not in fail_reasons:
        fail_reasons = [*fail_reasons, "low_trust_preview_only"]

    strip_ui = list(ui_official) if not low_trust_preview_only else list(ui_preview)
    preview_fallback = bool(low_trust_preview_only)

    plus_keyframes = _plus_rows_from_ui_strip(strip_ui, poses, preview_fallback=preview_fallback)
    plus_keyframes.sort(key=lambda r: PHASE_ORDER.index(r["phase"]) if r["phase"] in PHASE_ORDER else 99)

    phase_keyframes_pose = _phase_map_from_plus_rows(plus_keyframes)
    ai_vision = [str(k.get("image_base64") or "") for k in plus_keyframes][:8]

    missing_b64 = sum(1 for x in ai_vision if not str(x).strip())
    formal_scoring_allowed = (
        (final_status == "pass")
        and bool(semantic_ok)
        and len(plus_keyframes) == 8
        and missing_b64 == 0
        and not preview_fallback
    )

    gate_pass = bool(formal_scoring_allowed)
    fv_strict_reasons: list[str] = [] if gate_pass else list(fail_reasons)

    kf_validation: dict[str, Any] = {
        "total_keyframes": len(plus_keyframes),
        "near_duplicates": 0,
        "time_too_close": 0,
        "all_passed": gate_pass,
        "details": [],
        "final_phase_keyframes": dict(phase_keyframes_pose),
        "final_keyframe_validation": {
            "pass": gate_pass,
            "strict_contract_ok": gate_pass,
            "semantic_strip_ok": gate_pass,
            "strict_contract_fail_reasons": fv_strict_reasons,
            "near_duplicates": 0,
            "time_too_close_count": 0,
            "source_frame_gaps_ok": True,
            "source_frame_gap_reasons": [],
            "final_keyframe_order_ok": len(plus_keyframes) == 8,
            "final_keyframe_time_order_ok": len(plus_keyframes) == 8,
            "final_phase_keyframes_sync_ok": len(phase_keyframes_pose) >= 8,
            "negative_time_gap_in_details": False,
            "relabel_count": 0,
            "rebuild_used": False,
            "top_reselected": False,
            "impact_reselected": False,
            "prov3_authoritative": True,
        },
        "final_keyframe_order_ok": len(plus_keyframes) == 8,
        "final_keyframe_time_order_ok": len(plus_keyframes) == 8,
        "final_phase_keyframes_sync_ok": len(phase_keyframes_pose) >= 8,
        "negative_time_gap_in_details": False,
        "final_keyframe_gate_pass": gate_pass,
        "final_validation_failed": not gate_pass,
        "final_keyframe_source": "prov3_pass" if gate_pass else "prov3_low_trust",
        "phase_strip_repaired": False,
        "display_rebuild_applied": False,
        "final_phase_semantic_pass": bool(semantic_ok),
        "final_phase_semantic_fail_reasons": [] if semantic_ok else list(semantic_fails or fail_reasons),
    }

    sem_ok = bool(semantic_ok)
    pv_pass = bool(gate_pass)
    sem_report: dict[str, Any] = {
        "phase_detector_version": "prov3_true240",
        "phase_detector_confidence": None,
        "top_candidate_debug": None,
        "impact_candidate_debug": None,
        "top_keyframe_vs_event": None,
        "impact_keyframe_vs_event": None,
        "top_semantic_at_keyframe": None,
        "impact_semantic_at_keyframe": None,
        "top_semantic_ok": sem_ok,
        "impact_semantic_ok": sem_ok,
        "semantic_validation": {"source": "prov3", "ok": sem_ok},
        "final_phase_semantic_ok": sem_ok,
        "final_phase_semantic_ok_strict": bool(formal_scoring_allowed),
        "final_phase_semantic_ok_strict_reasons": [] if formal_scoring_allowed else list(fail_reasons),
        "phase_validation_passed": pv_pass,
        "phase_validation_soft_fail": not pv_pass,
        "keyframe_semantic_ok": sem_ok,
        "align_top": True,
        "align_impact": True,
        "align_tol": None,
        "top_abs_err": None,
        "impact_abs_err": None,
        "phase_reselection_failed": False,
        "dt_axis_invalid": False,
        "non_finite_kinematics": False,
        "fail_code": None if formal_scoring_allowed else "PROV3_TRUST_OR_SEMANTIC_FAIL",
        "rebuild_used": False,
    }

    phase_validation_soft = {
        "passed": pv_pass,
        "source": "prov3_mapped_to_pose_stream",
    }

    return {
        "plus_keyframes": plus_keyframes,
        "ai_vision_base64_list": ai_vision,
        "phase_keyframes_pose": phase_keyframes_pose,
        "kf_validation": kf_validation,
        "sem_report": sem_report,
        "phase_validation_soft": phase_validation_soft,
        "phase_source": "prov3_true240",
        "formal_scoring_allowed": formal_scoring_allowed,
        "product_ready": gate_pass,
        "prov3": {
            "analysis_id": str(dumped.get("analysis_id") or ""),
            "final_status": final_status,
            "trust_level": trust_raw,
            "analysis_trust": analysis_trust,
            "fail_reasons": fail_reasons,
            "low_trust_preview_only": low_trust_preview_only,
            "semantic_acceptance_ok": semantic_ok,
            "semantic_fail_reasons": list(semantic_fails),
        },
        "official_ui_rows": list(ui_official),
        "preview_ui_rows": list(ui_preview),
        "_prov3_motion": {
            "analysis_video": av_path,
            "analysis_fps": float(analysis_fps),
            "source_fps": float(prov3.source_fps or 30.0),
            "screen_mode": bool(screen_mode),
        },
    }
