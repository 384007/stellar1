"""Copied from ``services.internal.a_gate_service`` (Lite mirror)."""

from __future__ import annotations

from typing import List, Tuple

from services.lite_ab_mirror.constants import A_PASS_MIN_AVG_CONFIDENCE
from services.lite_ab_mirror.event_rules import core_frame_gap_ok, validate_event_order
from services.lite_ab_mirror.scoring import average_confidence

# Stricter than legacy 0.55 for Top / Impact to cut semantic false passes (minimal delta).
_A_TOP_IMPACT_MIN_CONFIDENCE = 0.58
_A_FINISH_MIN_CONFIDENCE = 0.55

# Mid-downswing corridor along Top→Impact (aligned with ``downswing_refine``)
_MID_DS_REL_MIN = 0.12
_MID_DS_REL_MAX = 0.88


def _mid_downswing_gate_reasons(keyframes: List[dict]) -> List[str]:
    """Low-trust hints when Mid-downswing is outside a plausible downswing band."""
    by = {str(k.get("event_name")): k for k in keyframes}
    need = ("Mid-downswing", "Top", "Impact")
    if not all(e in by for e in need):
        return []
    top = int(by["Top"]["frame_index"])
    mds = int(by["Mid-downswing"]["frame_index"])
    imp = int(by["Impact"]["frame_index"])
    reasons: List[str] = []
    if not (top < mds < imp):
        reasons.append("mid_downswing_semantic_invalid")
        return reasons
    corridor = imp - top
    if corridor < 2:
        return reasons
    rel = (mds - top) / float(corridor)
    if rel < _MID_DS_REL_MIN:
        reasons.append("mid_downswing_too_close_to_top")
    if rel > _MID_DS_REL_MAX:
        reasons.append("mid_downswing_too_close_to_impact")
    return reasons


def run_lite_a_gate(
    keyframes: List[dict],
    *,
    semantic_fail_reasons: List[str] | None = None,
) -> Tuple[str, List[str]]:
    fail_reasons: List[str] = []

    order_ok = validate_event_order(keyframes)
    gap_ok = core_frame_gap_ok(keyframes)

    if not order_ok:
        fail_reasons.append("event_order_invalid")

    if not gap_ok:
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

    for r in _mid_downswing_gate_reasons(keyframes):
        if r not in fail_reasons:
            fail_reasons.append(r)

    if semantic_fail_reasons:
        for r in semantic_fail_reasons:
            if not r or r in fail_reasons:
                continue
            fail_reasons.append(r)

    return ("fail", fail_reasons) if fail_reasons else ("pass", [])
