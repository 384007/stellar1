"""Prov3 keyframe business constants (frontend-safe naming only)."""

from __future__ import annotations

EVENT_SEQUENCE = [
    "Address",
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
    "Finish",
]

TOP_K = 3

A_PASS_MIN_AVG_CONFIDENCE = 0.62
A_MIN_CORE_FRAME_CONFIDENCE = 0.55
A_CORE_EVENTS = {"Top", "Impact"}

B_PASS_MIN_AVG_CONFIDENCE = 0.58
B_MIN_CORE_FRAME_CONFIDENCE = 0.50

TRUST_HIGH = "high"
TRUST_MEDIUM = "medium"
TRUST_LOW = "low"

A_STATUS_PASS = "pass"
A_STATUS_FAIL = "fail"
B_STATUS_PASS = "pass"
B_STATUS_LOW_TRUST = "low_trust"
