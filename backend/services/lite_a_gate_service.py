"""Lite-only A gate: stricter validation for the standard path (no ``lib.prov3``)."""

from __future__ import annotations

import statistics
from typing import Any

from services.lite_keyframe_constants import LITE_EVENT_SEQUENCE

LITE_A_PASS_MIN_AVG_CONFIDENCE = 0.62
LITE_A_MIN_CORE_FRAME_CONFIDENCE = 0.55
LITE_A_CORE_EVENTS = frozenset({"Top", "Impact"})
LITE_A_TOP_IMPACT_MIN_GAP_FRAMES = 4
LITE_A_MAX_IMPACT_HINT_DEVIATION_FRAMES = 38


def _validate_event_order(keyframes: list[dict[str, Any]]) -> bool:
    event_to_idx = {str(item.get("event_name")): int(item.get("frame_index", -1)) for item in keyframes}
    ordered = [event_to_idx.get(name, -1) for name in LITE_EVENT_SEQUENCE]
    if any(idx < 0 for idx in ordered):
        return False
    return all(left < right for left, right in zip(ordered, ordered[1:]))


def _core_frame_gap_ok(keyframes: list[dict[str, Any]], min_gap: int) -> bool:
    event_to_idx = {str(item.get("event_name")): int(item.get("frame_index", -1)) for item in keyframes}
    top_idx = event_to_idx.get("Top", -1)
    impact_idx = event_to_idx.get("Impact", -1)
    return top_idx >= 0 and impact_idx >= 0 and (impact_idx - top_idx) >= min_gap


def _average_confidence(keyframes: list[dict[str, Any]]) -> float:
    values = [float(item.get("confidence", 0.0)) for item in keyframes]
    return float(statistics.mean(values)) if values else 0.0


def _impact_frame_index(keyframes: list[dict[str, Any]]) -> int:
    for item in keyframes:
        if str(item.get("event_name")) == "Impact":
            return int(item.get("frame_index", -1))
    return -1


def run_lite_a_gate(
    keyframes: list[dict[str, Any]],
    *,
    impact_hint_frame_index: int,
) -> tuple[str, list[str]]:
    fail_reasons: list[str] = []

    if not _validate_event_order(keyframes):
        fail_reasons.append("event_order_invalid")

    if not _core_frame_gap_ok(keyframes, LITE_A_TOP_IMPACT_MIN_GAP_FRAMES):
        fail_reasons.append("top_impact_gap_invalid")

    avg_conf = _average_confidence(keyframes)
    if avg_conf < LITE_A_PASS_MIN_AVG_CONFIDENCE:
        fail_reasons.append("low_overall_confidence")

    event_conf = {str(item.get("event_name")): float(item.get("confidence", 0.0)) for item in keyframes}
    for core_event in LITE_A_CORE_EVENTS:
        if event_conf.get(core_event, 0.0) < LITE_A_MIN_CORE_FRAME_CONFIDENCE:
            fail_reasons.append(f"{core_event.lower()}_confidence_low")

    if any(float(item.get("confidence", 0.0)) < 0.35 for item in keyframes):
        fail_reasons.append("possible_club_visibility_issue")

    imp = _impact_frame_index(keyframes)
    if imp >= 0 and abs(imp - int(impact_hint_frame_index)) > LITE_A_MAX_IMPACT_HINT_DEVIATION_FRAMES:
        fail_reasons.append("impact_hint_mismatch")

    return ("fail", fail_reasons) if fail_reasons else ("pass", [])
