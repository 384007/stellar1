"""A→B chain — mirror of ``services.prov3_keyframe_orchestrator_service._run_ab_after_preprocess``."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from services.golfdb_swingnet_service import clear_swingnet_ctx
from services.lite_ab_mirror.a_extract import run_lite_a_extract
from services.lite_ab_mirror.b_refine import run_lite_b_refine
from services.lite_ab_mirror.constants import TRUST_HIGH, TRUST_LOW, TRUST_MEDIUM
from services.lite_ab_mirror.scoring import per_event_confidence
from services.provider_registry import role_log

logger = logging.getLogger(__name__)
_LOG = "[lite_ab]"


def run_lite_ab_after_preprocess(
    pre: Dict[str, Any],
    *,
    cancel_check: Callable[[], None] | None = None,
    plus_fast_b: bool = False,
) -> Tuple[List[dict], str, bool, List[str]]:
    """Returns ``(final_keyframe_rows, trust_tier, phase_passed, ab_fail_reasons)``."""
    analysis_id = str(pre["analysis_id"])
    analysis_video = str(pre["analysis_video_path"])
    preprocess_meta = {
        "source_fps": float(pre.get("source_fps", 30.0)),
        "analysis_fps": int(round(float(pre.get("analysis_fps", 240)))),
        "stabilized": True,
        "denoised": True,
        "cropped_single_swing": True,
        "screen_mode_corrected": bool(pre.get("screen_mode_corrected", False)),
    }
    analysis_frames: List[dict] = list(pre.get("analysis_frames") or [])
    enhanced_local_frames: List[dict] = list(pre.get("enhanced_local_frames") or [])

    if int(preprocess_meta["analysis_fps"]) != 240:
        logger.warning(
            "%s analysis_fps=%s (expected 240 fake-CFR for Lite SwingNet alignment)",
            _LOG,
            preprocess_meta["analysis_fps"],
        )

    if cancel_check:
        cancel_check()

    role_log("[ROLE=LITE_PIPELINE] lite_ab_A_extract_start (SwingNet infer + gate)")
    a_result = run_lite_a_extract(
        analysis_id=analysis_id,
        analysis_video=analysis_video,
        preprocess_meta=preprocess_meta,
        analysis_frames=analysis_frames,
    )
    logger.info("%s [A] %s reasons=%s", _LOG, a_result.a_status, a_result.fail_reasons)

    if a_result.a_status == "pass":
        clear_swingnet_ctx(analysis_id)
        return a_result.keyframes, TRUST_HIGH, True, []

    if cancel_check:
        cancel_check()

    role_log("[ROLE=LITE_PIPELINE] lite_ab_A_need_B_refine starting_B_path")
    conf = per_event_confidence(a_result.keyframes)
    b_result = run_lite_b_refine(
        analysis_id=analysis_id,
        analysis_video=analysis_video,
        preprocess_meta=preprocess_meta,
        analysis_frames=analysis_frames,
        enhanced_local_frames=enhanced_local_frames,
        keyframes=a_result.keyframes,
        confidence=conf,
        fail_reasons=a_result.fail_reasons,
        plus_fast=plus_fast_b,
    )
    logger.info("%s [B] %s reasons=%s", _LOG, b_result.b_status, b_result.fail_reasons)

    if b_result.b_status == "pass":
        return b_result.refined_keyframes, TRUST_MEDIUM, True, list(b_result.fail_reasons)

    return (
        b_result.refined_keyframes,
        TRUST_LOW,
        False,
        list(b_result.fail_reasons),
    )
