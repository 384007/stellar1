from __future__ import annotations

from typing import Optional


def normalize_hand(hand: str | None) -> str:
    value = str(hand or "UNKNOWN").upper().strip()
    return value if value in ("R", "L") else "UNKNOWN"


def get_lead_trail_sides(hand: str | None) -> dict[str, str]:
    norm = normalize_hand(hand)
    if norm == "L":
        return {"lead_side": "right", "trail_side": "left"}
    return {"lead_side": "left", "trail_side": "right"}


def enrich_angles_with_lead_trail(angles: dict | None, hand: str | None) -> dict:
    base = dict(angles or {})
    sides = get_lead_trail_sides(hand)
    lead_side = sides["lead_side"]
    trail_side = sides["trail_side"]

    def _angle(name: str, fallback: float = 0.0) -> float:
        value = base.get(name, fallback)
        try:
            return float(value)
        except Exception:
            return float(fallback)

    base["hand"] = normalize_hand(hand)
    base["lead_side"] = lead_side
    base["trail_side"] = trail_side
    base["lead_elbow"] = _angle(f"{lead_side}_elbow")
    base["trail_elbow"] = _angle(f"{trail_side}_elbow")
    base["lead_knee"] = _angle(f"{lead_side}_knee")
    base["trail_knee"] = _angle(f"{trail_side}_knee")
    base["lead_shoulder"] = _angle(f"{lead_side}_shoulder")
    base["trail_shoulder"] = _angle(f"{trail_side}_shoulder")
    return base


def _get_joint(pose: dict, name: str) -> Optional[dict]:
    for joint in pose.get("joints", []):
        if joint.get("name") == name:
            return joint
    return None


def _joint_x(pose: dict, name: str) -> Optional[float]:
    joint = _get_joint(pose, name)
    if not joint:
        return None
    norm = joint.get("normalized", {})
    x = norm.get("x")
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _center_x(pose: dict, a: str, b: str) -> Optional[float]:
    ax = _joint_x(pose, a)
    bx = _joint_x(pose, b)
    if ax is None or bx is None:
        return None
    return (ax + bx) * 0.5


def _phase_pose_map(poses: list[dict], swing_phases: list[dict] | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not poses:
        return out
    if not swing_phases:
        out["address"] = poses[0]
        out["top"] = poses[max(0, len(poses) // 2 - 1)]
        out["impact"] = poses[min(len(poses) - 1, len(poses) // 2 + 1)]
        return out

    for idx, phase in enumerate(swing_phases):
        pid = phase.get("phase_id")
        if pid and pid not in out and idx < len(poses):
            out[pid] = poses[idx]
    return out


def detect_handedness(
    poses: list[dict],
    swing_phases: list[dict] | None = None,
) -> dict:
    """
    Detect golfer handedness from multi-frame, multi-phase evidence.

    Returns:
      { "hand": "R" | "L" | "UNKNOWN", "confidence": float, "reason": str }
    """
    if not poses:
        return {"hand": "UNKNOWN", "confidence": 0.0, "reason": "no_pose_frames", "fallback_applied": True, "fallback_hand": "R"}

    phase_map = _phase_pose_map(poses, swing_phases)
    votes = {"R": 0.0, "L": 0.0}
    evidence: list[str] = []
    used = 0

    def vote_from_club_side(pose: dict, phase_name: str, weight: float) -> None:
        nonlocal used
        shoulder_mid = _center_x(pose, "left_shoulder", "right_shoulder")
        wrist_l = _joint_x(pose, "left_wrist")
        wrist_r = _joint_x(pose, "right_wrist")
        if shoulder_mid is None or wrist_l is None or wrist_r is None:
            return
        used += 1
        left_dist = abs(wrist_l - shoulder_mid)
        right_dist = abs(wrist_r - shoulder_mid)
        if abs(left_dist - right_dist) < 0.01:
            return
        # Right-handed: lead hand is left -> typically farther from body center.
        if left_dist > right_dist:
            votes["R"] += weight
            evidence.append(f"{phase_name}:left_wrist_outer")
        else:
            votes["L"] += weight
            evidence.append(f"{phase_name}:right_wrist_outer")

    for phase_name, wt in (("address", 1.0), ("takeaway", 1.1), ("top", 1.2), ("impact", 1.1)):
        pose = phase_map.get(phase_name)
        if pose:
            vote_from_club_side(pose, phase_name, wt)

    top_pose = phase_map.get("top")
    impact_pose = phase_map.get("impact")
    if top_pose and impact_pose:
        top_l = _joint_x(top_pose, "left_wrist")
        top_r = _joint_x(top_pose, "right_wrist")
        imp_l = _joint_x(impact_pose, "left_wrist")
        imp_r = _joint_x(impact_pose, "right_wrist")
        if None not in (top_l, top_r, imp_l, imp_r):
            used += 1
            move_l = abs(float(imp_l) - float(top_l))
            move_r = abs(float(imp_r) - float(top_r))
            if abs(move_l - move_r) >= 0.01:
                # Trail arm tends to move more through transition.
                if move_r > move_l:
                    votes["R"] += 0.8
                    evidence.append("transition:right_wrist_more_motion")
                else:
                    votes["L"] += 0.8
                    evidence.append("transition:left_wrist_more_motion")

    total = votes["R"] + votes["L"]
    if used == 0 or total <= 0.2:
        return {"hand": "UNKNOWN", "confidence": 0.2, "reason": "insufficient_multi_phase_evidence", "fallback_applied": True, "fallback_hand": "R"}

    if votes["R"] > votes["L"]:
        hand = "R"
        margin = votes["R"] - votes["L"]
    else:
        hand = "L"
        margin = votes["L"] - votes["R"]

    confidence = max(0.0, min(1.0, 0.45 + margin / max(total, 1e-6) * 0.5))
    if confidence < 0.58:
        return {
            "hand": "UNKNOWN",
            "confidence": round(confidence, 3),
            "reason": "low_confidence:" + "|".join(evidence[:4]),
            "fallback_applied": True,
            "fallback_hand": "R",
        }

    return {
        "hand": hand,
        "confidence": round(confidence, 3),
        "reason": "multi_phase:" + "|".join(evidence[:5]),
    }
