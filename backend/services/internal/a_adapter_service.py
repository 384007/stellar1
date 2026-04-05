from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from services.golfdb_swingnet_service import (
    PROV3_A_ENGINE_ID,
    run_swingnet_extract,
    swingnet_checkpoint_path,
    swingnet_enabled,
)

logger = logging.getLogger(__name__)


def infer_a_candidates(
    analysis_frames: List[dict],
    *,
    analysis_video: Optional[str] = None,
    analysis_id: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Pro v3 A-path keyframes from SwingNet only.

    If SwingNet is unavailable or inference fails, return empty list instead of
    heuristic/fake keyframes.
    """
    _ = (analysis_frames, preprocess_meta)
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
        logger.warning("[prov3][A] %s inference failed — returning empty keyframes", PROV3_A_ENGINE_ID)
        return []

    logger.warning(
        "[prov3][A] SwingNet unavailable or input missing (enabled=%s, video=%s, analysis_id=%s)",
        swingnet_enabled(),
        bool(analysis_video and os.path.isfile(analysis_video)),
        bool(analysis_id),
    )
    return []
