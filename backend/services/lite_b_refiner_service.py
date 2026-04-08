"""Lite B refiner: recovery path after A fails (no prov3)."""

from __future__ import annotations

import logging
import statistics
from typing import Any

from services.lite_keyframe_candidate_b import lite_build_candidate_b_rows
from services.lite_keyframe_constants import LITE_EVENT_SEQUENCE
from services.lite_keyframe_heuristic import (
    lite_enforce_monotonic_frame_indices,
    lite_refine_impact_row,
)

logger = logging.getLogger(__name__)
_LOG = "[lite_b]"

_B_PASS_MIN_AVG = 0.58
_B_MIN_CORE = 0.50
_B_CORE = frozenset({"Top", "Impact"})
_B_TOP_IMPACT_GAP = 3
_B_MAX_IMPACT_HINT_DEV = 72
_B_MOTION_WINDOW = 5


def _validate_order(keyframes: list[dict[str, Any]]) -> bool:
    event_to_idx = {str(item.get("event_name")): int(item.get("frame_index", -1)) for item in keyframes}
    ordered = [event_to_idx.get(name, -1) for name in LITE_EVENT_SEQUENCE]
    if any(idx < 0 for idx in ordered):
        return False
    return all(left < right for left, right in zip(ordered, ordered[1:]))


def _core_gap(keyframes: list[dict[str, Any]], min_gap: int) -> bool:
    event_to_idx = {str(item.get("event_name")): int(item.get("frame_index", -1)) for item in keyframes}
    top_idx = event_to_idx.get("Top", -1)
    impact_idx = event_to_idx.get("Impact", -1)
    return top_idx >= 0 and impact_idx >= 0 and (impact_idx - top_idx) >= min_gap


def _avg_conf(keyframes: list[dict[str, Any]]) -> float:
    values = [float(item.get("confidence", 0.0)) for item in keyframes]
    return float(statistics.mean(values)) if values else 0.0


def _impact_fi(keyframes: list[dict[str, Any]]) -> int:
    for item in keyframes:
        if str(item.get("event_name")) == "Impact":
            return int(item.get("frame_index", -1))
    return -1


def _impact_timeline_index(frame_indices: list[int], impact_fi: int) -> int:
    best_i = 0
    best_d = 10**9
    for i, fi in enumerate(frame_indices):
        d = abs(int(fi) - int(impact_fi))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _motion_jitter_high(
    motions: list[float],
    impact_hint_frame_index: int,
    frame_indices: list[int],
) -> bool:
    if len(motions) < 4 or not frame_indices:
        return False
    k = _impact_timeline_index(frame_indices, impact_hint_frame_index)
    lo = max(1, k - _B_MOTION_WINDOW)
    hi = min(len(motions) - 1, k + _B_MOTION_WINDOW)
    window = [float(motions[i]) for i in range(lo, hi + 1) if i < len(motions)]
    if len(window) < 3:
        return False
    mean_v = statistics.mean(window)
    if mean_v <= 1e-6:
        return False
    try:
        st = statistics.pstdev(window)
    except statistics.StatisticsError:
        return False
    return (st / mean_v) > 2.8


def _b_fail_reasons(
    keyframes: list[dict[str, Any]],
    *,
    impact_hint_frame_index: int,
    frame_indices: list[int],
    motions: list[float],
) -> list[str]:
    reasons: list[str] = []
    if not _validate_order(keyframes):
        reasons.append("event_order_invalid")
    if not _core_gap(keyframes, _B_TOP_IMPACT_GAP):
        reasons.append("top_impact_gap_invalid")
    if _avg_conf(keyframes) < _B_PASS_MIN_AVG:
        reasons.append("low_overall_confidence")
    event_conf = {str(item.get("event_name")): float(item.get("confidence", 0.0)) for item in keyframes}
    for core_event in _B_CORE:
        if event_conf.get(core_event, 0.0) < _B_MIN_CORE:
            reasons.append(f"{core_event.lower()}_confidence_low")
    if any(float(item.get("confidence", 0.0)) < 0.25 for item in keyframes):
        reasons.append("possible_club_visibility_issue")
    imp = _impact_fi(keyframes)
    if imp >= 0 and abs(imp - int(impact_hint_frame_index)) > _B_MAX_IMPACT_HINT_DEV:
        reasons.append("impact_hint_mismatch")
    if _motion_jitter_high(motions, impact_hint_frame_index, frame_indices):
        reasons.append("motion_window_unstable")
    return reasons


def run_lite_b_refine(
    preprocess: dict[str, Any],
    a_bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    Build B rows from the same preprocess timeline (recovery geometry).
    Uses A bundle only for logging context; rows are not a trivial copy of A.
    """
    analysis_path = str(preprocess["analysis_video_path"])
    indices = list(a_bundle["indices"])
    motions = list(a_bundle["motions"])
    hint_fi = int(a_bundle["impact_hint_frame_index"])
    max_fi = int(a_bundle["max_fi"])

    rows0 = lite_build_candidate_b_rows(indices, motions)
    rows0 = lite_refine_impact_row(rows0, hint_fi)
    rows = lite_enforce_monotonic_frame_indices(rows0, max_fi)

    reasons = _b_fail_reasons(
        rows,
        impact_hint_frame_index=hint_fi,
        frame_indices=indices,
        motions=motions,
    )
    b_pass = len(reasons) == 0

    logger.info("%s refine b_pass=%s reasons=%s (a_kf_pass=%s)", _LOG, b_pass, reasons, a_bundle.get("kf_pass"))

    return {
        "rows": rows,
        "b_pass": b_pass,
        "fail_reasons": reasons,
    }
