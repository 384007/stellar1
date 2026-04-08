"""Copied from ``lib.prov3.keyframes.constants`` for Lite A/B mirror (no prov3 package import)."""

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

TOP_K = 10

A_PASS_MIN_AVG_CONFIDENCE = 0.62
A_MIN_CORE_FRAME_CONFIDENCE = 0.55
A_CORE_EVENTS = {"Top", "Impact"}

B_PASS_MIN_AVG_CONFIDENCE = 0.58
B_MIN_CORE_FRAME_CONFIDENCE = 0.50

TRUST_HIGH = "high"
TRUST_MEDIUM = "medium"
TRUST_LOW = "low"
