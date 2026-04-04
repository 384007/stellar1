from __future__ import annotations

from typing import List

from .constants import EVENT_SEQUENCE


def validate_event_order(keyframes: List[dict]) -> bool:
    """Return True only when all event frame indices are non-decreasing in expected order."""
    event_to_idx = {item.get("event_name"): int(item.get("frame_index", -1)) for item in keyframes}
    ordered = [event_to_idx.get(name, -1) for name in EVENT_SEQUENCE]
    if any(idx < 0 for idx in ordered):
        return False
    return all(left <= right for left, right in zip(ordered, ordered[1:]))


def core_frame_gap_ok(keyframes: List[dict], min_gap: int = 4) -> bool:
    event_to_idx = {item.get("event_name"): int(item.get("frame_index", -1)) for item in keyframes}
    top_idx = event_to_idx.get("Top", -1)
    impact_idx = event_to_idx.get("Impact", -1)
    return top_idx >= 0 and impact_idx >= 0 and (impact_idx - top_idx) >= min_gap
