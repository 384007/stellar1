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

_LITE_SWINGNET_DEFAULT_FRAMES = 1200
_LITE_SWINGNET_MAX_FRAMES = 2400
_LITE_SWINGNET_MIN_ACCURATE_FRAMES = 480


def infer_lite_a_candidates(
    analysis_frames: List[dict],
    *,
    analysis_video: Optional[str] = None,
    analysis_id: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    _ = (analysis_frames, preprocess_meta)
    if not swingnet_enabled():
        raise RuntimeError("lite_swingnet_checkpoint_missing")
    if not analysis_video or not os.path.isfile(analysis_video):
        raise RuntimeError("lite_swingnet_analysis_video_missing")
    if not analysis_id:
        raise RuntimeError("lite_swingnet_analysis_id_missing")

    logger.info(
        "[lite_ab][A] engine=%s checkpoint=%s",
        PROV3_A_ENGINE_ID,
        swingnet_checkpoint_path(),
    )
    afps = float((preprocess_meta or {}).get("analysis_fps") or 240)
    lite_cap_raw = (os.getenv("STELLAR_SWINGNET_LITE_MAX_FRAMES") or str(_LITE_SWINGNET_DEFAULT_FRAMES)).strip()
    try:
        lite_cap = int(lite_cap_raw)
    except ValueError:
        lite_cap = _LITE_SWINGNET_DEFAULT_FRAMES
    if lite_cap < _LITE_SWINGNET_MIN_ACCURATE_FRAMES:
        logger.warning(
            "[lite_ab][A] STELLAR_SWINGNET_LITE_MAX_FRAMES=%s below accuracy floor=%s; using floor",
            lite_cap,
            _LITE_SWINGNET_MIN_ACCURATE_FRAMES,
        )
    lite_cap = max(_LITE_SWINGNET_MIN_ACCURATE_FRAMES, min(lite_cap, _LITE_SWINGNET_MAX_FRAMES))
    kfs = run_swingnet_extract(
        analysis_video,
        analysis_id=analysis_id,
        analysis_fps=afps,
        max_extract_frames=lite_cap,
    )
    if not kfs:
        raise RuntimeError("lite_swingnet_inference_empty")
    return kfs
