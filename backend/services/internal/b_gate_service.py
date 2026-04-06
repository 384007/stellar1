from __future__ import annotations

from typing import List, Tuple

from lib.prov3.keyframes.constants import B_MIN_CORE_FRAME_CONFIDENCE, B_PASS_MIN_AVG_CONFIDENCE
from lib.prov3.keyframes.event_rules import core_frame_gap_ok, validate_event_order
from lib.prov3.keyframes.scoring import average_confidence


def _high_risk_gap_ok(refined_keyframes: List[dict]) -> bool:
    idx = {str(x.get("event_name") or ""): int(x.get("frame_index", -1)) for x in refined_keyframes}
    top = idx.get("Top", -1)
    mid_down = idx.get("Mid-downswing", -1)
    impact = idx.get("Impact", -1)
    finish = idx.get("Finish", -1)
    if min(top, mid_down, impact, finish) < 0:
        return False
    return (mid_down - top) >= 4 and (impact - mid_down) >= 4 and (finish - impact) >= 6


def run_b_gate(refined_keyframes: List[dict], incoming_fail_reasons: List[str]) -> Tuple[str, List[str]]:
    fail_reasons: List[str] = list(incoming_fail_reasons)

    if not validate_event_order(refined_keyframes):
        fail_reasons.append("event_order_not_recovered")

    if not core_frame_gap_ok(refined_keyframes):
        fail_reasons.append("top_impact_relation_unstable")
    if not _high_risk_gap_ok(refined_keyframes):
        fail_reasons.append("high_risk_event_spacing_unstable")

    if average_confidence(refined_keyframes) < B_PASS_MIN_AVG_CONFIDENCE:
        fail_reasons.append("confidence_below_refine_threshold")

    event_conf = {item.get("event_name"): float(item.get("confidence", 0.0)) for item in refined_keyframes}
    if event_conf.get("Top", 0.0) < B_MIN_CORE_FRAME_CONFIDENCE:
        fail_reasons.append("top_not_reliable")
    if event_conf.get("Impact", 0.0) < B_MIN_CORE_FRAME_CONFIDENCE:
        fail_reasons.append("impact_not_reliable")

    return ("low_trust", sorted(set(fail_reasons))) if fail_reasons else ("pass", [])
