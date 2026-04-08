"""Legacy B-style refine wrapper (single pass). Lite product path uses ``orchestrator.run_lite_ab_after_preprocess``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from services.golfdb_swingnet_service import clear_swingnet_ctx
from services.lite_ab_mirror.b_gate import run_lite_b_gate
from services.lite_ab_mirror.b_layer import refine_with_lite_b_layer
from services.lite_ab_mirror.scoring import per_event_confidence


@dataclass
class LiteRefineResult:
    analysis_id: str
    refined_keyframes: List[dict]
    b_status: str
    fail_reasons: List[str] = field(default_factory=list)


def run_lite_b_refine(
    *,
    analysis_id: str,
    analysis_video: str,
    preprocess_meta: Dict[str, object],
    analysis_frames: List[dict],
    enhanced_local_frames: List[dict],
    keyframes: List[dict],
    confidence: Dict[str, float],
    fail_reasons: List[str],
    plus_fast: bool = False,
) -> LiteRefineResult:
    try:
        refined = refine_with_lite_b_layer(
            keyframes,
            enhanced_local_frames,
            analysis_id=analysis_id,
            analysis_video=analysis_video,
            preprocess_meta=preprocess_meta,
            analysis_frames=analysis_frames,
            confidence=confidence,
            fail_reasons=fail_reasons,
            recovery_pass=False,
            plus_fast=plus_fast,
        )
        max_idx = max((int(x.get("frame_index", 0)) for x in analysis_frames), default=-1)
        if max_idx >= 0:
            for row in refined:
                fi = int(row.get("frame_index", 0))
                row["frame_index"] = max(0, min(fi, max_idx))
        b_status, b_fail_reasons = run_lite_b_gate(refined, fail_reasons)
        return LiteRefineResult(
            analysis_id=analysis_id,
            refined_keyframes=refined,
            b_status=b_status,
            fail_reasons=b_fail_reasons,
        )
    finally:
        clear_swingnet_ctx(analysis_id)
