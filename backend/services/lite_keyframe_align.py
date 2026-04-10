"""
Lite keyframe export: unified translation alignment from pose anchors (no re-pick frames).

Uses hip midpoint → shoulder midpoint → upper-body visible joints; falls back to image center.
Target center = median of the eight per-frame anchors (pixel space, current frame size).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_MIN_VIS = 0.22
_MIN_VIS_STRICT = 0.35


def _joint(joints: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for j in joints:
        if str(j.get("name") or "") == name:
            return j
    return None


def _to_px(j: dict[str, Any], fw: int, fh: int) -> tuple[float, float]:
    n = j.get("normalized")
    if isinstance(n, dict) and n.get("x") is not None and n.get("y") is not None:
        return float(n["x"]) * fw, float(n["y"]) * fh
    return float(j.get("x", 0.0)), float(j.get("y", 0.0))


def anchor_xy_from_joints(
    joints: list[dict[str, Any]] | None,
    fw: int,
    fh: int,
) -> tuple[float, float, bool]:
    """
    Returns (cx, cy, anchor_is_confident) in pixel coordinates.
    Confident = hip or shoulder mid from reasonably visible points.
    """
    if not joints or fw <= 0 or fh <= 0:
        return float(fw) * 0.5, float(fh) * 0.5, False

    lh = _joint(joints, "left_hip")
    rh = _joint(joints, "right_hip")
    ls = _joint(joints, "left_shoulder")
    rs = _joint(joints, "right_shoulder")

    def ok(j: dict[str, Any] | None) -> bool:
        return j is not None and float(j.get("visibility", 0.0)) >= _MIN_VIS

    # 1) Hip midpoint (preferred)
    if ok(lh) and ok(rh):
        x1, y1 = _to_px(lh, fw, fh)
        x2, y2 = _to_px(rh, fw, fh)
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5, True
    if ok(lh) or ok(rh):
        j = lh if ok(lh) else rh
        assert j is not None
        x, y = _to_px(j, fw, fh)
        return x, y, True

    # 2) Shoulder midpoint
    if ok(ls) and ok(rs):
        x1, y1 = _to_px(ls, fw, fh)
        x2, y2 = _to_px(rs, fw, fh)
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5, True
    if ok(ls) or ok(rs):
        j = ls if ok(ls) else rs
        assert j is not None
        x, y = _to_px(j, fw, fh)
        return x, y, True

    # 3) Torso: average visible upper-body landmarks
    upper = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_elbow", "right_elbow")
    pts: list[tuple[float, float]] = []
    for name in upper:
        j = _joint(joints, name)
        if j and float(j.get("visibility", 0.0)) >= _MIN_VIS:
            pts.append(_to_px(j, fw, fh))
    if len(pts) >= 2:
        return (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            False,
        )

    # 4) Any joint with minimal visibility
    loose: list[tuple[float, float]] = []
    for j in joints:
        if float(j.get("visibility", 0.0)) >= _MIN_VIS_STRICT:
            loose.append(_to_px(j, fw, fh))
    if loose:
        return (
            sum(p[0] for p in loose) / len(loose),
            sum(p[1] for p in loose) / len(loose),
            False,
        )

    return float(fw) * 0.5, float(fh) * 0.5, False


def closest_pose_for_frame(poses: list[dict[str, Any]], frame_index: int) -> dict[str, Any] | None:
    if not poses:
        return None
    best: dict[str, Any] | None = None
    best_d = 1e18
    for p in poses:
        if not isinstance(p, dict):
            continue
        pfi = int(p.get("frame_index", -10_000_000))
        d = abs(pfi - int(frame_index))
        if d < best_d:
            best_d = d
            best = p
    return best


def compensate_with_neighbors(
    anchors: list[tuple[float, float]],
    confident: list[bool],
) -> list[tuple[float, float]]:
    """Replace non-confident anchors with neighbor average where possible."""
    out = list(anchors)
    n = len(out)
    for i in range(n):
        if confident[i]:
            continue
        prev_a = out[i - 1] if i > 0 else None
        next_a = out[i + 1] if i + 1 < n else None
        if prev_a is not None and next_a is not None:
            out[i] = ((prev_a[0] + next_a[0]) * 0.5, (prev_a[1] + next_a[1]) * 0.5)
        elif prev_a is not None:
            out[i] = prev_a
        elif next_a is not None:
            out[i] = next_a
    return out


def median_target_center(anchors: list[tuple[float, float]]) -> tuple[float, float]:
    xs = sorted(a[0] for a in anchors)
    ys = sorted(a[1] for a in anchors)
    mid = len(anchors) // 2
    if len(anchors) % 2 == 1:
        return xs[mid], ys[mid]
    return (xs[mid - 1] + xs[mid]) * 0.5, (ys[mid - 1] + ys[mid]) * 0.5


def translate_frame(
    frame_bgr: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    m = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    return cv2.warpAffine(
        frame_bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def clamp_shift(dx: float, dy: float, fw: int, fh: int, max_frac: float = 0.22) -> tuple[float, float]:
    lim = max_frac * float(min(fw, fh))
    return float(np.clip(dx, -lim, lim)), float(np.clip(dy, -lim, lim))
