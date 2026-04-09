"""Copied from ``services.internal.a_gate_service`` (Lite mirror)."""

from __future__ import annotations

from typing import List, Tuple

from services.lite_ab_mirror.constants import A_PASS_MIN_AVG_CONFIDENCE
from services.lite_ab_mirror.event_rules import core_frame_gap_ok, validate_event_order
from services.lite_ab_mirror.scoring import average_confidence

# Stricter than legacy 0.55 for Top / Impact to cut semantic false passes (minimal delta).
_A_TOP_IMPACT_MIN_CONFIDENCE = 0.58
_A_FINISH_MIN_CONFIDENCE = 0.55


def run_lite_a_gate(
    keyframes: List[dict],
    *,
    semantic_fail_reasons: List[str] | None = None,
) -> Tuple[str, List[str]]:
    fail_reasons: List[str] = []

    if semantic_fail_reasons:
        for r in semantic_fail_reasons:
            if r and r not in fail_reasons:
                fail_reasons.append(r)

    if not validate_event_order(keyframes):
        fail_reasons.append("event_order_invalid")

    if not core_frame_gap_ok(keyframes):
        fail_reasons.append("top_impact_gap_invalid")

    avg_conf = average_confidence(keyframes)
    if avg_conf < A_PASS_MIN_AVG_CONFIDENCE:
        fail_reasons.append("low_overall_confidence")

    event_conf = {item.get("event_name"): float(item.get("confidence", 0.0)) for item in keyframes}
    if event_conf.get("Top", 0.0) < _A_TOP_IMPACT_MIN_CONFIDENCE:
        fail_reasons.append("top_confidence_low")
    if event_conf.get("Impact", 0.0) < _A_TOP_IMPACT_MIN_CONFIDENCE:
        fail_reasons.append("impact_confidence_low")
    if event_conf.get("Finish", 0.0) < _A_FINISH_MIN_CONFIDENCE:
        fail_reasons.append("finish_confidence_low")

    if any(float(item.get("confidence", 0.0)) < 0.35 for item in keyframes):
        fail_reasons.append("possible_club_visibility_issue")

    return ("fail", fail_reasons) if fail_reasons else ("pass", [])
