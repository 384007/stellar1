from __future__ import annotations

from typing import Dict, List

from lib.prov3.keyframes.types import ExtractResult
from services.internal.a_adapter_service import infer_a_candidates
from services.internal.a_gate_service import run_a_gate


def run_a_extract(
    *,
    analysis_id: str,
    analysis_video: str,
    preprocess_meta: Dict[str, object],
    analysis_frames: List[dict],
) -> ExtractResult:
    _ = (analysis_video, preprocess_meta)
    keyframes = infer_a_candidates(analysis_frames)
    a_status, fail_reasons = run_a_gate(keyframes)
    return ExtractResult(
        analysis_id=analysis_id,
        keyframes=keyframes,
        a_status=a_status,
        fail_reasons=fail_reasons,
    )
