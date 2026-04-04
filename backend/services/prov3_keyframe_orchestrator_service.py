from __future__ import annotations

from typing import Dict

from lib.prov3.keyframes.constants import TRUST_HIGH, TRUST_LOW, TRUST_MEDIUM
from lib.prov3.keyframes.scoring import per_event_confidence
from lib.prov3.keyframes.types import AnalyzeResponse
from services.internal.low_trust_service import build_low_trust_result
from services.prov3_keyframe_a_extractor_service import run_a_extract
from services.prov3_keyframe_b_refiner_service import run_b_refine
from services.prov3_keyframe_preprocess_service import run_preprocess


def run_keyframe_analyze(input_video: str, work_dir: str, *, screen_mode: bool = False) -> AnalyzeResponse:
    pre = run_preprocess(input_video, work_dir, screen_mode=screen_mode)

    a_result = run_a_extract(
        analysis_id=pre.analysis_id,
        analysis_video=pre.analysis_video,
        preprocess_meta=pre.preprocess_meta.model_dump(),
        analysis_frames=pre.analysis_frames,
    )

    if a_result.a_status == "pass":
        return AnalyzeResponse(
            analysis_id=pre.analysis_id,
            status="pass",
            trust_level=TRUST_HIGH,
            keyframes=a_result.keyframes,
            fail_reasons=[],
        )

    b_result = run_b_refine(
        analysis_id=pre.analysis_id,
        analysis_video=pre.analysis_video,
        preprocess_meta=pre.preprocess_meta.model_dump(),
        analysis_frames=pre.analysis_frames,
        enhanced_local_frames=pre.enhanced_local_frames,
        keyframes=[item.model_dump() for item in a_result.keyframes],
        confidence=per_event_confidence([item.model_dump() for item in a_result.keyframes]),
        fail_reasons=a_result.fail_reasons,
    )

    if b_result.b_status == "pass":
        return AnalyzeResponse(
            analysis_id=pre.analysis_id,
            status="pass",
            trust_level=TRUST_MEDIUM,
            keyframes=b_result.refined_keyframes,
            fail_reasons=[],
        )

    low_trust = build_low_trust_result(
        analysis_id=pre.analysis_id,
        keyframes=[item.model_dump() for item in b_result.refined_keyframes],
        fail_reasons=b_result.fail_reasons,
    )
    return AnalyzeResponse(**low_trust)
