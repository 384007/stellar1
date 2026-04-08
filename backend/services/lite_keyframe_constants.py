"""Lite keyframe phase order — local copy; do not import ``lib.prov3`` from Lite services."""

from __future__ import annotations

LITE_EVENT_SEQUENCE: tuple[str, ...] = (
    "Address",
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
    "Finish",
)
