"""Lite single-stage keyframe pipeline: SwingNet infer → local refine → A quality gate."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from services.golfdb_swingnet_service import clear_swingnet_ctx
from services.lite_ab_mirror.a_extract import run_lite_a_infer_only
from services.lite_ab_mirror.a_gate import run_lite_a_gate
from services.lite_ab_mirror.b_layer import refine_with_lite_b_layer
from services.lite_ab_mirror.constants import TRUST_HIGH, TRUST_LOW
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
        logger.info("%s [infer] %s reasons=%s", _LOG, infer.a_status, infer.fail_reasons)

        if infer.a_status == "fail":
            return infer.keyframes, TRUST_LOW, False, list(infer.fail_reasons)

        if cancel_check:
            cancel_check()

        role_log("[ROLE=LITE_PIPELINE] lite_keyframes_local_refine_start")
        conf = per_event_confidence(infer.keyframes)
        refined = refine_with_lite_b_layer(
            infer.keyframes,
            enhanced_local_frames,
            analysis_id=analysis_id,
            analysis_video=analysis_video,
            preprocess_meta=preprocess_meta,
            analysis_frames=analysis_frames,
            confidence=conf,
            fail_reasons=[],
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

        role_log("[ROLE=LITE_PIPELINE] lite_keyframes_quality_gate_start")
        a_status, fail_reasons = run_lite_a_gate(refined)
        logger.info("%s [gate] %s reasons=%s", _LOG, a_status, fail_reasons)

        phase_pass = a_status == "pass"
        trust = TRUST_HIGH if phase_pass else TRUST_LOW
        return refined, trust, phase_pass, list(fail_reasons)
    finally:
        clear_swingnet_ctx(analysis_id)
