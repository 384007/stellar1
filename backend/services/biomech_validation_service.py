"""Biomechanical semantic validation facade."""

from __future__ import annotations

from typing import Any

from services.provider_registry import role_log

PHASE_IDS = ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]


def validate_phase_chain_hard(
    poses: list[dict],
    phase_keyframes: dict[str, int],
) -> dict[str, Any]:
    n = len(poses)
    reasons: list[str] = []
    idx = [int(phase_keyframes.get(pid, -1)) for pid in PHASE_IDS]
    if any(i < 0 or i >= n for i in idx):
        reasons.append("index_out_of_range")
    if any(idx[i] >= idx[i + 1] for i in range(len(idx) - 1)):
        reasons.append("phase_order_not_strictly_increasing")
    top_i = int(phase_keyframes.get("top", -1))
    impact_i = int(phase_keyframes.get("impact", -1))
    follow_i = int(phase_keyframes.get("follow_through", -1))
    finish_i = int(phase_keyframes.get("finish", -1))
    if n > 0 and (top_i < int(0.20 * n) or top_i > int(0.75 * n)):
        reasons.append("top_out_of_range")
    if impact_i >= 0 and top_i >= 0 and impact_i <= top_i:
        reasons.append("impact_not_after_top")
    if follow_i >= 0 and finish_i >= 0 and (finish_i - follow_i) < max(2, n // 25):
        reasons.append("follow_finish_too_close")
    if impact_i >= 1 and impact_i < n - 1:
        t0 = float(poses[impact_i - 1].get("timestamp", impact_i - 1))
        t1 = float(poses[impact_i].get("timestamp", impact_i))
        t2 = float(poses[impact_i + 1].get("timestamp", impact_i + 1))
        if not (t0 < t1 < t2):
            reasons.append("impact_time_axis_invalid")
    out = {"passed": len(reasons) == 0, "reasons": reasons}
    role_log(f"[ROLE=BIOMECH] passed={out['passed']} reasons={reasons}")
    return out


def evaluate_phase_strip_semantics(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    keyframes: list[dict] | None = None,
    final_keyframe_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from services.swing_flow_utils import build_semantic_phase_report

    return build_semantic_phase_report(
        poses,
        dict(phase_keyframes),
        phase_validation={"passed": True},
        keyframes=keyframes,
        final_keyframe_validation=final_keyframe_validation,
    )
