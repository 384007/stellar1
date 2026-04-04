"""Pro v2 — single entry to build picker_tuning from routing + retry (orchestrator-facing)."""

from __future__ import annotations

import logging
from typing import Any

from services.pro_v2_strategy_profiles import (
    RoutingDerivedParams,
    base_picker_tuning_dict,
    merge_retry_reasons_into_tuning,
)

logger = logging.getLogger(__name__)


def build_pro_v2_picker_tuning(
    *,
    routing: RoutingDerivedParams,
    retry_reasons: list[str],
    attempt_index: int,
    screen_mode: bool,
) -> dict[str, Any]:
    """Merge AI routing base tuning with round-2 retry_reasons when ``attempt_index >= 1``.

    ``screen_mode`` reserved for future non-screen branches; routing already encodes pipeline.
    """
    _ = screen_mode
    base = base_picker_tuning_dict(routing)
    if attempt_index <= 0:
        return dict(base)
    return merge_retry_reasons_into_tuning(dict(base), list(retry_reasons or []), routing=routing)


def log_picker_tuning(
    tuning: dict[str, Any],
    *,
    analysis_id: str,
    attempt_index: int,
    review_round: int,
    retry_reasons: list[str],
    analysis_trust: str | None = None,
) -> None:
    """Structured log: retry_reasons → concrete picker knobs (visible in Modal)."""
    keys = sorted(tuning.keys())
    snap = {k: tuning.get(k) for k in keys}
    logger.info(
        "[PRO_V2][PICKER_TUNING] id=%s attempt=%s review_round=%s analysis_trust=%s retry_reasons=%s tuning=%s",
        analysis_id or "n/a",
        attempt_index,
        review_round,
        analysis_trust or "n/a",
        list(retry_reasons or [])[:14],
        snap,
    )
