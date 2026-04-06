from __future__ import annotations

import logging
from typing import Callable, Dict

from lib.prov3.keyframes.constants import TRUST_HIGH, TRUST_LOW, TRUST_MEDIUM
from lib.prov3.keyframes.scoring import per_event_confidence
from lib.prov3.keyframes.types import AnalyzeResponse
from services.internal.low_trust_service import build_low_trust_result
from services.golfdb_swingnet_service import clear_swingnet_ctx
from services.prov3_keyframe_a_extractor_service import run_a_extract
from services.prov3_keyframe_b_refiner_service import run_b_refine
from services.prov3_keyframe_preprocess_service import run_preprocess

logger = logging.getLogger(__name__)


def run_keyframe_analyze(
    input_video: str,
    work_dir: str,
    *,
    screen_mode: bool = False,
    cancel_check: Callable[[], None] | None = None,
    plus_fast_b: bool = False,
) -> AnalyzeResponse:
    if cancel_check:
        cancel_check()
    pre = run_preprocess(input_video, work_dir, screen_mode=screen_mode, cancel_check=cancel_check)
    if int(pre.preprocess_meta.analysis_fps) != 240:
        raise RuntimeError(f"true240_required: analysis_fps={pre.preprocess_meta.analysis_fps}")

    logger.info("[prov3][A] start analysis_id=%s", pre.analysis_id)
    if cancel_check:
        cancel_check()
    a_result = run_a_extract(
        analysis_id=pre.analysis_id,
        analysis_video=pre.analysis_video,
        preprocess_meta=pre.preprocess_meta.model_dump(),
        analysis_frames=pre.analysis_frames,
    )
    logger.info(
        "[prov3][A] %s reasons=%s",
        a_result.a_status,
        a_result.fail_reasons,
    )

    if a_result.a_status == "pass":
        clear_swingnet_ctx(pre.analysis_id)
        for row in a_result.keyframes:
            fi = int(row.frame_index)
            if fi < 0:
                raise RuntimeError(f"analysis_timeline_index_invalid:{fi}")
        return AnalyzeResponse(
            analysis_id=pre.analysis_id,
            status="pass",
            trust_level=TRUST_HIGH,
            keyframes=a_result.keyframes,
            fail_reasons=[],
            analysis_video=pre.analysis_video,
            analysis_fps=int(pre.preprocess_meta.analysis_fps),
            source_fps=float(pre.preprocess_meta.source_fps),
        )

    logger.info(
        "[prov3][B] start analysis_id=%s incoming_reasons=%s",
        pre.analysis_id,
        a_result.fail_reasons,
    )
    if cancel_check:
        cancel_check()
    b_result = run_b_refine(
        analysis_id=pre.analysis_id,
        analysis_video=pre.analysis_video,
        preprocess_meta=pre.preprocess_meta.model_dump(),
        analysis_frames=pre.analysis_frames,
        enhanced_local_frames=pre.enhanced_local_frames,
        keyframes=[item.model_dump() for item in a_result.keyframes],
        confidence=per_event_confidence([item.model_dump() for item in a_result.keyframes]),
        fail_reasons=a_result.fail_reasons,
        plus_fast=plus_fast_b,
    )
    logger.info(
        "[prov3][B] %s reasons=%s",
        b_result.b_status,
        b_result.fail_reasons,
    )

    if b_result.b_status == "pass":
        clear_swingnet_ctx(pre.analysis_id)
        for row in b_result.refined_keyframes:
            fi = int(row.frame_index)
            if fi < 0:
                raise RuntimeError(f"analysis_timeline_index_invalid:{fi}")
        return AnalyzeResponse(
            analysis_id=pre.analysis_id,
            status="pass",
            trust_level=TRUST_MEDIUM,
            keyframes=b_result.refined_keyframes,
            fail_reasons=[],
            analysis_video=pre.analysis_video,
            analysis_fps=int(pre.preprocess_meta.analysis_fps),
            source_fps=float(pre.preprocess_meta.source_fps),
        )

    low_trust = build_low_trust_result(
        analysis_id=pre.analysis_id,
        keyframes=[item.model_dump() for item in b_result.refined_keyframes],
        fail_reasons=b_result.fail_reasons,
    )
    for row in b_result.refined_keyframes:
        fi = int(row.frame_index)
        if fi < 0:
            raise RuntimeError(f"analysis_timeline_index_invalid:{fi}")
    clear_swingnet_ctx(pre.analysis_id)
    return AnalyzeResponse(
        **low_trust,
        analysis_video=pre.analysis_video,
        analysis_fps=int(pre.preprocess_meta.analysis_fps),
        source_fps=float(pre.preprocess_meta.source_fps),
    )
