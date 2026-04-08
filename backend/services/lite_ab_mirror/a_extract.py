"""A-path extract — mirror of ``services.prov3_keyframe_a_extractor_service``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from services.lite_ab_mirror.a_adapter import infer_lite_a_candidates
from services.lite_ab_mirror.a_gate import run_lite_a_gate
from services.lite_ab_mirror.constants import EVENT_SEQUENCE

_A_INCOMPLETE_REASON = "a_inference_unavailable_or_incomplete"


@dataclass
class LiteExtractResult:
    analysis_id: str
    keyframes: List[dict]
    a_status: str
    fail_reasons: List[str] = field(default_factory=list)


def run_lite_a_extract(
    *,
    analysis_id: str,
    analysis_video: str,
    preprocess_meta: Dict[str, object],
    analysis_frames: List[dict],
) -> LiteExtractResult:
    keyframes = infer_lite_a_candidates(
        analysis_frames,
        analysis_video=analysis_video,
        analysis_id=analysis_id,
        preprocess_meta=preprocess_meta,
    )

    if len(keyframes) != len(EVENT_SEQUENCE):
        return LiteExtractResult(
            analysis_id=analysis_id,
            keyframes=keyframes,
            a_status="fail",
            fail_reasons=[_A_INCOMPLETE_REASON],
        )

    a_status, fail_reasons = run_lite_a_gate(keyframes)
    if a_status != "pass" and _A_INCOMPLETE_REASON not in fail_reasons:
        fail_reasons = [*fail_reasons, _A_INCOMPLETE_REASON]

    return LiteExtractResult(
        analysis_id=analysis_id,
        keyframes=keyframes,
        a_status=a_status,
        fail_reasons=fail_reasons,
    )
