"""Copied from ``services.internal.b_gate_service`` (Lite mirror)."""

from __future__ import annotations

from typing import List, Tuple

from services.lite_ab_mirror.constants import B_MIN_CORE_FRAME_CONFIDENCE, B_PASS_MIN_AVG_CONFIDENCE
from services.lite_ab_mirror.event_rules import core_frame_gap_ok, validate_event_order
from services.lite_ab_mirror.scoring import average_confidence


def _high_risk_gap_ok(refined_keyframes: List[dict]) -> bool:
    idx = {str(x.get("event_name") or ""): int(x.get("frame_index", -1)) for x in refined_keyframes}
    top = idx.get("Top", -1)
    mid_down = idx.get("Mid-downswing", -1)
    impact = idx.get("Impact", -1)
    finish = idx.get("Finish", -1)
    if min(top, mid_down, impact, finish) < 0:
        return False
    return (mid_down - top) >= 4 and (impact - mid_down) >= 4 and (finish - impact) >= 6


def _core_semantic_ok(refined_keyframes: List[dict]) -> bool:
    rows = {str(x.get("event_name") or ""): x for x in refined_keyframes}
    top = rows.get("Top", {})
    mid = rows.get("Mid-downswing", {})
    impact = rows.get("Impact", {})
    if not top or not mid or not impact:
        return False
    top_i = int(top.get("frame_index", -1))
    mid_i = int(mid.get("frame_index", -1))
    imp_i = int(impact.get("frame_index", -1))
    top_c = float(top.get("confidence", 0.0))
    imp_c = float(impact.get("confidence", 0.0))
    if min(top_i, mid_i, imp_i) < 0:
        return False
    if not (top_i < mid_i < imp_i):
        return False
    if (mid_i - top_i) < 4:
        return False
    if (imp_i - mid_i) < 4:
        return False
    if (imp_i - top_i) < 9:
        return False
    if top_c < (B_MIN_CORE_FRAME_CONFIDENCE - 0.03):
        return False
    if imp_c < (B_MIN_CORE_FRAME_CONFIDENCE - 0.03):
        return False
    return True


def run_lite_b_gate(refined_keyframes: List[dict], incoming_fail_reasons: List[str]) -> Tuple[str, List[str]]:
    fail_reasons: List[str] = list(incoming_fail_reasons)

    if not validate_event_order(refined_keyframes):
        fail_reasons.append("event_order_not_recovered")

    if not core_frame_gap_ok(refined_keyframes):
        fail_reasons.append("top_impact_relation_unstable")
    if not _high_risk_gap_ok(refined_keyframes):
        fail_reasons.append("high_risk_event_spacing_unstable")
    if not _core_semantic_ok(refined_keyframes):
        fail_reasons.append("core_event_semantic_unstable")

    if average_confidence(refined_keyframes) < B_PASS_MIN_AVG_CONFIDENCE:
        fail_reasons.append("confidence_below_refine_threshold")

    event_conf = {item.get("event_name"): float(item.get("confidence", 0.0)) for item in refined_keyframes}
    if event_conf.get("Top", 0.0) < B_MIN_CORE_FRAME_CONFIDENCE:
        fail_reasons.append("top_not_reliable")
    if event_conf.get("Impact", 0.0) < B_MIN_CORE_FRAME_CONFIDENCE:
        fail_reasons.append("impact_not_reliable")

    return ("low_trust", sorted(set(fail_reasons))) if fail_reasons else ("pass", [])
