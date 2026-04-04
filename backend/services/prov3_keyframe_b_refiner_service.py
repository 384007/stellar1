from __future__ import annotations

from typing import Dict, List

from lib.prov3.keyframes.types import RefineResult
from services.internal.b_adapter_service import refine_with_b_layer
from services.internal.b_gate_service import run_b_gate


def run_b_refine(
    *,
    analysis_id: str,
    analysis_video: str,
    preprocess_meta: Dict[str, object],
    analysis_frames: List[dict],
    enhanced_local_frames: List[dict],
    keyframes: List[dict],
    confidence: Dict[str, float],
    fail_reasons: List[str],
) -> RefineResult:
    _ = (analysis_video, preprocess_meta, analysis_frames, confidence)
    refined = refine_with_b_layer(keyframes, enhanced_local_frames)
    b_status, b_fail_reasons = run_b_gate(refined, fail_reasons)
    return RefineResult(
        analysis_id=analysis_id,
        refined_keyframes=refined,
        b_status=b_status,
        fail_reasons=b_fail_reasons,
    )
