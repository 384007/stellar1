"""Lite keyframe pipeline: SwingNet A infer → A quality gate only (no B / local refine)."""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Tuple

from services.golfdb_swingnet_service import clear_swingnet_ctx
from services.lite_ab_mirror.a_extract import run_lite_a_infer_only
from services.lite_ab_mirror.a_gate import run_lite_a_gate
from services.lite_ab_mirror.constants import TRUST_HIGH, TRUST_LOW
from services.provider_registry import role_log

logger = logging.getLogger(__name__)
_LOG = "[lite_ab]"
_KF = "[LITE_KF_DECIDE]"


def run_lite_ab_after_preprocess(
    pre: Dict[str, Any],
    *,
    cancel_check: Callable[[], None] | None = None,
    plus_fast_b: bool = False,
) -> Tuple[List[dict], str, bool, List[str]]:
    """Returns ``(final_keyframe_rows, trust_tier, phase_passed, fail_reasons)``."""
    _ = plus_fast_b  # API compatibility; B refine removed
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
        role_log(
            f"[ROLE=LITE_PIPELINE] {_KF} infer_done status={infer.a_status!r} "
            f"reasons={list(infer.fail_reasons)!r}"
        )

        if infer.a_status == "fail":
            return infer.keyframes, TRUST_LOW, False, list(infer.fail_reasons)

        if cancel_check:
            cancel_check()

        rows = copy.deepcopy(infer.keyframes)
        max_idx = max((int(x.get("frame_index", 0)) for x in analysis_frames), default=-1)
        if max_idx >= 0:
            for row in rows:
                fi = int(row.get("frame_index", 0))
                row["frame_index"] = max(0, min(fi, max_idx))

        status, fail_reasons = run_lite_a_gate(rows)
        logger.info("%s %s a_gate status=%s reasons=%s", _LOG, _KF, status, fail_reasons)
        role_log(
            f"[ROLE=LITE_PIPELINE] {_KF} a_gate status={status!r} reasons={list(fail_reasons)!r}"
        )

        phase_pass = status == "pass"
        trust = TRUST_HIGH if phase_pass else TRUST_LOW
        out_reasons: List[str] = [] if phase_pass else list(fail_reasons)
        return rows, trust, phase_pass, out_reasons
    finally:
        clear_swingnet_ctx(analysis_id)
