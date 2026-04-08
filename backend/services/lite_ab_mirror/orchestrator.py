"""Lite single-stage keyframe pipeline: SwingNet infer → optional local refine → A quality gate."""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Set, Tuple

from services.golfdb_swingnet_service import clear_swingnet_ctx
from services.lite_ab_mirror.a_extract import run_lite_a_infer_only
from services.lite_ab_mirror.a_gate import run_lite_a_gate
from services.lite_ab_mirror.b_layer import refine_with_lite_b_layer
from services.lite_ab_mirror.constants import TRUST_HIGH, TRUST_LOW
from services.lite_ab_mirror.scoring import per_event_confidence
from services.provider_registry import role_log

logger = logging.getLogger(__name__)
_LOG = "[lite_ab]"
_KF = "[LITE_KF_DECIDE]"

_REFINE_ALLOWED_FAILS: Set[str] = {
    "event_order_invalid",
    "top_impact_gap_invalid",
    "low_overall_confidence",
    "top_confidence_low",
    "impact_confidence_low",
    "finish_confidence_low",
}


def _tif_conf_sum(keyframes: List[dict]) -> float:
    d = {str(r.get("event_name") or ""): float(r.get("confidence", 0.0)) for r in keyframes}
    return float(d.get("Top", 0.0) + d.get("Impact", 0.0) + d.get("Finish", 0.0))


def _choose_better_keyframes(
    raw_rows: List[dict],
    raw_status: str,
    raw_reasons: List[str],
    refined_rows: List[dict],
    refined_status: str,
    refined_reasons: List[str],
) -> tuple[List[dict], str, List[str], str]:
    """
    Prefer refined only when clearly better; else raw A.
    Returns (rows, gate_status, gate_reasons, picked_label).
    """
    raw_pass = raw_status == "pass"
    ref_pass = refined_status == "pass"
    if ref_pass and not raw_pass:
        return refined_rows, refined_status, refined_reasons, "refined"
    if raw_pass and not ref_pass:
        return raw_rows, raw_status, raw_reasons, "raw"
    if raw_pass and ref_pass:
        if _tif_conf_sum(refined_rows) > _tif_conf_sum(raw_rows) + 1e-9:
            return refined_rows, refined_status, refined_reasons, "refined"
        return raw_rows, raw_status, raw_reasons, "raw"
    if len(refined_reasons) < len(raw_reasons):
        return refined_rows, refined_status, refined_reasons, "refined"
    if len(raw_reasons) < len(refined_reasons):
        return raw_rows, raw_status, raw_reasons, "raw"
    if _tif_conf_sum(refined_rows) > _tif_conf_sum(raw_rows) + 1e-9:
        return refined_rows, refined_status, refined_reasons, "refined"
    return raw_rows, raw_status, raw_reasons, "raw"


def run_lite_ab_after_preprocess(
    pre: Dict[str, Any],
    *,
    cancel_check: Callable[[], None] | None = None,
    plus_fast_b: bool = False,
) -> Tuple[List[dict], str, bool, List[str]]:
    """Returns ``(final_keyframe_rows, trust_tier, phase_passed, fail_reasons)``."""
    analysis_id = str(pre["analysis_id"])
    analysis_video = str(pre["analysis_video_path"])
    preprocess_meta = {
        "source_fps": float(pre.get("source_fps", 30.0)),
        "analysis_fps": int(round(float(pre.get("analysis_fps", 30)))),
        "stabilized": True,
        "denoised": True,
        "cropped_single_swing": True,
        "screen_mode_corrected": bool(pre.get("screen_mode_corrected", False)),
    }
    analysis_frames: List[dict] = list(pre.get("analysis_frames") or [])
    enhanced_local_frames: List[dict] = list(pre.get("enhanced_local_frames") or [])

    try:
        if cancel_check:
            cancel_check()

        role_log("[ROLE=LITE_PIPELINE] lite_keyframes_infer_start (SwingNet)")
        infer = run_lite_a_infer_only(
            analysis_id=analysis_id,
            analysis_video=analysis_video,
            preprocess_meta=preprocess_meta,
            analysis_frames=analysis_frames,
        )
        logger.info("%s %s infer_done status=%s reasons=%s", _LOG, _KF, infer.a_status, infer.fail_reasons)

        if infer.a_status == "fail":
            return infer.keyframes, TRUST_LOW, False, list(infer.fail_reasons)

        raw_rows = copy.deepcopy(infer.keyframes)
        raw_status, raw_fail_reasons = run_lite_a_gate(raw_rows)
        logger.info(
            "%s %s raw_a_gate status=%s reasons=%s",
            _LOG,
            _KF,
            raw_status,
            raw_fail_reasons,
        )

        allowed_hits = set(raw_fail_reasons) & _REFINE_ALLOWED_FAILS
        should_refine = raw_status != "pass" and bool(allowed_hits)
        logger.info(
            "%s %s refine_decision should_refine=%s allowed_hits=%s",
            _LOG,
            _KF,
            should_refine,
            sorted(allowed_hits),
        )

        if not should_refine:
            if raw_status == "pass":
                logger.info("%s %s final_pick=raw_a (no_refine, gate_pass)", _LOG, _KF)
                return raw_rows, TRUST_HIGH, True, []
            logger.info("%s %s final_pick=raw_a (no_refine, gate_fail)", _LOG, _KF)
            return raw_rows, TRUST_LOW, False, list(raw_fail_reasons)

        if cancel_check:
            cancel_check()

        role_log("[ROLE=LITE_PIPELINE] lite_keyframes_local_refine_start")
        conf = per_event_confidence(raw_rows)
        refined = refine_with_lite_b_layer(
            copy.deepcopy(raw_rows),
            enhanced_local_frames,
            analysis_id=analysis_id,
            analysis_video=analysis_video,
            preprocess_meta=preprocess_meta,
            analysis_frames=analysis_frames,
            confidence=conf,
            fail_reasons=list(raw_fail_reasons),
            recovery_pass=False,
            plus_fast=plus_fast_b,
        )

        max_idx = max((int(x.get("frame_index", 0)) for x in analysis_frames), default=-1)
        if max_idx >= 0:
            for row in refined:
                fi = int(row.get("frame_index", 0))
                row["frame_index"] = max(0, min(fi, max_idx))

        if cancel_check:
            cancel_check()

        refined_status, refined_fail_reasons = run_lite_a_gate(refined)
        logger.info(
            "%s %s refined_gate status=%s reasons=%s",
            _LOG,
            _KF,
            refined_status,
            refined_fail_reasons,
        )

        final_rows, final_status, final_reasons, picked = _choose_better_keyframes(
            raw_rows,
            raw_status,
            raw_fail_reasons,
            refined,
            refined_status,
            refined_fail_reasons,
        )
        logger.info(
            "%s %s final_pick=%s gate_status=%s reasons=%s",
            _LOG,
            _KF,
            picked,
            final_status,
            final_reasons,
        )

        phase_pass = final_status == "pass"
        trust = TRUST_HIGH if phase_pass else TRUST_LOW
        out_reasons: List[str] = [] if phase_pass else list(final_reasons)
        return final_rows, trust, phase_pass, out_reasons
    finally:
        clear_swingnet_ctx(analysis_id)
