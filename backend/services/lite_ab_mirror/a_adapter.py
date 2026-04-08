"""A-path SwingNet extract — same as ``services.internal.a_adapter_service`` (Lite mirror)."""

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


def infer_lite_a_candidates(
    analysis_frames: List[dict],
    *,
    analysis_video: Optional[str] = None,
    analysis_id: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    _ = (analysis_frames, preprocess_meta)
    if (
        swingnet_enabled()
        and analysis_video
        and analysis_id
        and os.path.isfile(analysis_video)
    ):
        logger.info(
            "[lite_ab][A] engine=%s checkpoint=%s",
            PROV3_A_ENGINE_ID,
            swingnet_checkpoint_path(),
        )
        afps = float((preprocess_meta or {}).get("analysis_fps") or 240)
        kfs = run_swingnet_extract(
            analysis_video,
            analysis_id=analysis_id,
            analysis_fps=afps,
        )
        if kfs:
            return kfs
        logger.warning("[lite_ab][A] %s inference failed — returning empty keyframes", PROV3_A_ENGINE_ID)
        return []

    logger.warning(
        "[lite_ab][A] SwingNet unavailable or input missing (enabled=%s, video=%s, analysis_id=%s)",
        swingnet_enabled(),
        bool(analysis_video and os.path.isfile(analysis_video)),
        bool(analysis_id),
    )
    return []
