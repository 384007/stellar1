"""Copied from ``lib.prov3.keyframes.scoring`` (Lite mirror)."""

from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List


def average_confidence(keyframes: List[dict]) -> float:
    values = [float(item.get("confidence", 0.0)) for item in keyframes]
    return float(mean(values)) if values else 0.0


def per_event_confidence(keyframes: Iterable[dict]) -> Dict[str, float]:
    return {str(item.get("event_name")): float(item.get("confidence", 0.0)) for item in keyframes}
