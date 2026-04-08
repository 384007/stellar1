"""Copied from ``services.internal.a_gate_service`` (Lite mirror)."""

from __future__ import annotations

from typing import List, Tuple

from services.lite_ab_mirror.constants import (
    A_CORE_EVENTS,
    A_MIN_CORE_FRAME_CONFIDENCE,
    A_PASS_MIN_AVG_CONFIDENCE,
)
from services.lite_ab_mirror.event_rules import core_frame_gap_ok, validate_event_order
from services.lite_ab_mirror.scoring import average_confidence


def run_lite_a_gate(keyframes: List[dict]) -> Tuple[str, List[str]]:
    fail_reasons: List[str] = []

    if not validate_event_order(keyframes):
        fail_reasons.append("event_order_invalid")

    if not core_frame_gap_ok(keyframes):
        fail_reasons.append("top_impact_gap_invalid")

    avg_conf = average_confidence(keyframes)
    if avg_conf < A_PASS_MIN_AVG_CONFIDENCE:
        fail_reasons.append("low_overall_confidence")

    event_conf = {item.get("event_name"): float(item.get("confidence", 0.0)) for item in keyframes}
    for core_event in A_CORE_EVENTS:
        if event_conf.get(core_event, 0.0) < A_MIN_CORE_FRAME_CONFIDENCE:
            fail_reasons.append(f"{core_event.lower()}_confidence_low")

    if any(float(item.get("confidence", 0.0)) < 0.35 for item in keyframes):
        fail_reasons.append("possible_club_visibility_issue")

    return ("fail", fail_reasons) if fail_reasons else ("pass", [])
