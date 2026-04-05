from __future__ import annotations

from typing import Dict, List

from lib.prov3.keyframes.constants import EVENT_SEQUENCE
from lib.prov3.keyframes.types import ExtractResult
from services.internal.a_adapter_service import infer_a_candidates
from services.internal.a_gate_service import run_a_gate


_A_INCOMPLETE_REASON = "a_inference_unavailable_or_incomplete"


def run_a_extract(
    *,
    analysis_id: str,
    analysis_video: str,
    preprocess_meta: Dict[str, object],
    analysis_frames: List[dict],
) -> ExtractResult:
    keyframes = infer_a_candidates(
        analysis_frames,
        analysis_video=analysis_video,
        analysis_id=analysis_id,
        preprocess_meta=preprocess_meta,
    )

    if len(keyframes) != len(EVENT_SEQUENCE):
        return ExtractResult(
            analysis_id=analysis_id,
            keyframes=keyframes,
            a_status="fail",
            fail_reasons=[_A_INCOMPLETE_REASON],
        )

    a_status, fail_reasons = run_a_gate(keyframes)
    if a_status != "pass" and _A_INCOMPLETE_REASON not in fail_reasons:
        # Preserve a canonical reason for downstream B focus when A is not fully usable.
        fail_reasons = [*fail_reasons, _A_INCOMPLETE_REASON]

    return ExtractResult(
        analysis_id=analysis_id,
        keyframes=keyframes,
        a_status=a_status,
        fail_reasons=fail_reasons,
    )
