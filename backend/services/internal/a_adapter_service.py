from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from lib.prov3.keyframes.constants import EVENT_SEQUENCE, TOP_K
from services.golfdb_swingnet_service import (
    PROV3_A_ENGINE_ID,
    run_swingnet_extract,
    swingnet_checkpoint_path,
    swingnet_enabled,
)

logger = logging.getLogger(__name__)


def _heuristic_a_candidates(analysis_frames: List[dict]) -> List[Dict[str, object]]:
    if not analysis_frames:
        return []

    max_idx = max(int(frame.get("frame_index", 0)) for frame in analysis_frames)
    stride = max(6, max_idx // max(1, len(EVENT_SEQUENCE) + 2))

    outputs: List[Dict[str, object]] = []
    for i, event_name in enumerate(EVENT_SEQUENCE):
        base_idx = min(max_idx, (i + 1) * stride)
        center_conf = max(0.35, 0.84 - (i * 0.03))
        top_k = []
        for k in range(TOP_K):
            top_k.append(
                {
                    "event_name": event_name,
                    "frame_index": max(0, base_idx + (k - 1) * 2),
                    "confidence": round(max(0.2, center_conf - abs(k - 1) * 0.07), 3),
                }
            )
        outputs.append(
            {
                "event_name": event_name,
                "frame_index": top_k[1]["frame_index"],
                "confidence": top_k[1]["confidence"],
                "top_k_candidates": top_k,
            }
        )
    return outputs


def infer_a_candidates(
    analysis_frames: List[dict],
    *,
    analysis_video: Optional[str] = None,
    analysis_id: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Pro v3 **A** = **wmcnally/golfdb SwingNet** when weights resolve; else heuristic only.

    Weights: ``STELLAR_SWINGNET_CHECKPOINT`` or ``backend/models/swingnet_1800.pth.tar``.
    """
    _ = preprocess_meta
    if (
        swingnet_enabled()
        and analysis_video
        and analysis_id
        and os.path.isfile(analysis_video)
    ):
        logger.info(
            "[prov3][A] engine=%s checkpoint=%s",
            PROV3_A_ENGINE_ID,
            swingnet_checkpoint_path(),
        )
        kfs = run_swingnet_extract(analysis_video, analysis_id=analysis_id)
        if kfs:
            return kfs
        logger.warning(
            "[prov3][A] %s inference failed for this clip — heuristic A-path",
            PROV3_A_ENGINE_ID,
        )
    return _heuristic_a_candidates(analysis_frames)
