"""Lite keyframe pipeline: SwingNet A infer → Mid-downswing refine (pose/motion) → A quality gate (A-only, no B)."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Callable, Dict, List, Tuple

from services.golfdb_swingnet_service import clear_swingnet_ctx
from services.lite_ab_mirror.a_extract import run_lite_a_infer_only
from services.lite_ab_mirror.a_gate import run_lite_a_gate
from services.lite_ab_mirror.constants import TRUST_HIGH, TRUST_LOW
from services.lite_ab_mirror.downswing_refine import refine_mid_downswing_with_pose_motion
from services.lite_timeline_motion import (
    lite_build_uniform_timeline,
    lite_impact_hint_from_timeline,
    lite_motion_along_timeline,
)
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
    total_frames = int(pre.get("total_frames") or 0)
    max_from_total = max(0, total_frames - 1) if total_frames > 0 else -1
    max_from_lattice = max((int(x.get("frame_index", 0)) for x in analysis_frames), default=-1)
    max_idx = max(max_from_lattice, max_from_total)
    if max_idx < 0:
        max_idx = 0
    preprocess_meta["max_frame_index"] = max_idx

    vfps = float(pre.get("analysis_fps") or 30.0)
    duration_s = float(pre.get("duration_s") or 0.0)
    timeline: List[dict] = []
    motions: List[float] = []
    impact_hint_fi: int | None = None
    if total_frames > 0 and analysis_video and os.path.isfile(analysis_video):
        timeline = lite_build_uniform_timeline(total_frames, vfps)
        indices = [int(t.get("frame_index", 0)) for t in timeline]
        if indices:
            motions = lite_motion_along_timeline(analysis_video, indices)
            hp = lite_impact_hint_from_timeline(indices, motions, vfps, duration_s or total_frames / max(vfps, 1e-6))
            impact_hint_fi = int(round(float(hp.get("impact_hint_s") or 0.0) * vfps))
            impact_hint_fi = max(0, min(impact_hint_fi, max_idx))

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

        rows = copy.deepcopy(infer.keyframes)
        refined_rows, refine_fail_reasons, refine_dbg = refine_mid_downswing_with_pose_motion(
            rows,
            analysis_frames=analysis_frames,
            preprocess_meta=preprocess_meta,
            poses=list(pre.get("poses") or []),
            timeline=timeline,
            motions=motions,
            impact_hint_frame_index=impact_hint_fi,
            max_frame_index=max_idx,
        )
        logger.info(
            "%s %s downswing_refine action=%s reasons=%s dbg=%s",
            _LOG,
            _KF,
            refine_dbg.get("action"),
            refine_fail_reasons,
            {k: refine_dbg.get(k) for k in ("module", "action", "original_mds", "final_mds", "touched")},
        )
        role_log(
            f"[ROLE=LITE_PIPELINE] {_KF} downswing_refine_done action={refine_dbg.get('action')!r} "
            f"reasons={list(refine_fail_reasons)!r}"
        )

        if cancel_check:
            cancel_check()

        for row in refined_rows:
            fi = int(row.get("frame_index", 0))
            row["frame_index"] = max(0, min(fi, max_idx))

        gate_extra_reasons = list(refine_fail_reasons)
        if infer.a_status == "fail":
            for r in infer.fail_reasons:
                if r and r not in gate_extra_reasons:
                    gate_extra_reasons.append(r)

        status, fail_reasons = run_lite_a_gate(refined_rows, semantic_fail_reasons=gate_extra_reasons)
        logger.info("%s %s a_gate status=%s reasons=%s", _LOG, _KF, status, fail_reasons)
        role_log(
            f"[ROLE=LITE_PIPELINE] {_KF} a_gate status={status!r} reasons={list(fail_reasons)!r}"
        )

        phase_pass = status == "pass"
        trust = TRUST_HIGH if phase_pass else TRUST_LOW
        out_reasons: List[str] = [] if phase_pass else list(fail_reasons)
        return refined_rows, trust, phase_pass, out_reasons
    finally:
        clear_swingnet_ctx(analysis_id)
