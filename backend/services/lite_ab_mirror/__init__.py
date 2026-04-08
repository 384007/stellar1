"""Lite-only mirror of Pro v3 A/B keyframe logic (no ``lib.prov3`` / ``services.prov3_*`` imports)."""

from __future__ import annotations

from services.lite_ab_mirror.orchestrator import run_lite_ab_after_preprocess

__all__ = ["run_lite_ab_after_preprocess"]
