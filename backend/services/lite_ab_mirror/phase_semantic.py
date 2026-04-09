"""Compatibility shim: full multi-phase semantic reorder was removed.

Lite A-only path now corrects **Mid-downswing** only via
``downswing_refine`` (poses / timeline / motions / impact_hint).

Import from ``downswing_refine`` directly in new code; this module
re-exports the same symbols for older import paths.
"""

from __future__ import annotations

from services.lite_ab_mirror.downswing_refine import (
    ensure_eight_keyframe_rows,
    refine_lite_a_rows_with_phase_semantics,
    refine_mid_downswing_with_pose_motion,
)

__all__ = [
    "ensure_eight_keyframe_rows",
    "refine_lite_a_rows_with_phase_semantics",
    "refine_mid_downswing_with_pose_motion",
]
