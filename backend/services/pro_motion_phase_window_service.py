"""Pro Stage 3: feature-driven eight phase windows (pose indices) — not proportional hard cuts."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from services.swing_flow_utils import detect_phase_events_agnostic

logger = logging.getLogger(__name__)

_PHASE_ORDER = [
    "address",
    "takeaway",
    "backswing",
    "top",
    "downswing",
    "impact",
    "follow_through",
    "finish",
]


def _arr(features: dict[str, Any], key: str, n: int) -> np.ndarray:
    raw = features.get(key) or []
    a = np.asarray(list(raw), dtype=np.float64)
    if a.size < n:
        a = np.pad(a, (0, n - a.size))
    return a[:n]


def _legacy_proportional_edges(n: int, top_i: int, imp_i: int) -> list[int]:
    """Previous fixed-ratio cutter — logged only for old/new comparison."""
    w_top = max(2, n // 35)
    w_imp = max(2, n // 45)
    e0 = 0
    e1 = max(1, n // 10)
    e2 = max(e1 + 1, int(top_i * 0.55))
    e3 = max(e2 + 1, top_i - w_top)
    e4 = min(n - 1, top_i + w_top)
    e5 = max(e4 + 1, imp_i - w_imp)
    e6 = min(n - 1, imp_i + w_imp)
    e7 = max(e6 + 1, min(n - 1, e6 + max(3, n // 12)))
    e8 = n - 1
    edges = [e0, e1, e2, e3, e4, e5, e6, e7, e8]
    for i in range(1, len(edges)):
        edges[i] = max(edges[i], edges[i - 1] + 1)
    edges[-1] = n - 1
    if edges[5] >= imp_i:
        edges[5] = max(edges[4] + 1, imp_i - 1)
    if edges[6] < imp_i:
        edges[6] = min(n - 1, max(imp_i, edges[5] + 1))
    return edges


def _motion_onset_idx(spd: np.ndarray, n: int) -> int:
    pre = max(8, min(n // 5, 40))
    thr = float(np.percentile(spd[:pre], 68)) if pre > 0 else 0.01
    thr = max(thr, float(np.max(spd[:pre]) * 0.35 + 1e-9))
    for i in range(2, min(n - 12, pre + max(15, n // 8))):
        if spd[i] > thr and spd[i + 1] > thr:
            return int(i)
    return max(2, min(n // 12, n - 10))


def _finish_stable_start(
    spd: np.ndarray,
    vy: np.ndarray,
    imp_hi: int,
    n: int,
    min_gap: int,
) -> int:
    """First sustained low-speed / low vertical-velocity region after impact window."""
    j0 = min(n - 2, imp_hi + max(2, min_gap))
    ref_lo = max(0, imp_hi - 4)
    baseline = float(np.percentile(spd[ref_lo : imp_hi + 5], 35)) if imp_hi + 5 > ref_lo else 0.02
    baseline = max(baseline, 1e-6)
    avy = np.abs(vy)
    vy_med = float(np.percentile(avy[max(1, imp_hi - 5) : min(n, imp_hi + 15)], 50))
    run = 0
    for j in range(j0, n - 2):
        if spd[j] < baseline * 0.9 and avy[j] < max(vy_med * 1.2, 0.008):
            run += 1
            if run >= 4:
                return int(max(j - 3, j0))
        else:
            run = 0
    return int(min(n - 2, imp_hi + max(min_gap * 2, n // 18)))


def _takeaway_end_idx(
    a_end: int,
    top_lo: int,
    sr: np.ndarray,
    wy: np.ndarray,
    spd: np.ndarray,
) -> int:
    if top_lo <= a_end + 3:
        return a_end + 2
    lo = a_end + 1
    hi = max(lo + 1, top_lo - 2)
    tgt = float(sr[min(hi, len(sr) - 1)] - sr[lo])
    if tgt < 1e-6:
        return min(hi, lo + max(2, (top_lo - lo) // 3))
    for i in range(lo, hi):
        if float(sr[i] - sr[lo]) >= 0.28 * tgt and spd[i] > float(np.percentile(spd[lo:hi + 1], 45)):
            return int(max(lo, i - 1))
    return int(min(hi, lo + max(2, (top_lo - lo) // 3)))


def build_motion_phase_windows(
    poses: list[dict],
    features: dict[str, Any],
    *,
    rough_impact_pose_idx: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build 8 monotonic windows from wrist/speed/rotation/spine signals + event anchors."""
    n = len(poses)
    if n < 8:
        raise ValueError("need at least 8 poses for motion windows")

    ev = detect_phase_events_agnostic(poses)
    top_ref = int(ev.get("top_pose_idx", max(2, n // 2)))
    imp_ref = int(ev.get("impact_pose_idx", min(n - 2, int(n * 0.72))))
    apex_ref = int(ev.get("excursion_apex_idx", top_ref))
    top_ref = max(2, min(top_ref, n - 6))
    imp_ref = max(top_ref + 2, min(imp_ref, n - 2))
    apex_ref = max(0, min(apex_ref, imp_ref))

    wy = _arr(features, "wrist_mid_y", n)
    vy = _arr(features, "wrist_vy", n)
    spd = _arr(features, "hand_speed", n)
    sr = _arr(features, "shoulder_rotation", n)
    hr = _arr(features, "hip_rotation", n)
    st = _arr(features, "spine_tilt_y", n)

    min_gap = max(2, n // 50)
    w_top = max(2, min(8, n // 32))
    w_imp = max(2, min(8, n // 38))

    # Impact focus: peak hand speed near detector + optional rough hint
    hint = int(rough_impact_pose_idx) if rough_impact_pose_idx is not None else imp_ref
    hint = max(top_ref + 1, min(n - 2, hint))
    lo_s = max(top_ref, hint - 12)
    hi_s = min(n - 1, hint + 12)
    local = spd[lo_s : hi_s + 1]
    imp_focus = int(lo_s + int(np.argmax(local))) if local.size else imp_ref
    imp_focus = max(top_ref + 1, min(n - 2, imp_focus))

    imp_lo = max(0, imp_focus - w_imp)
    imp_hi = min(n - 1, imp_focus + w_imp)

    # Top center: blend event top, apex, and wrist highest point (min y)
    apex_y = int(np.argmin(wy[: imp_focus + 1]))
    top_c = int(round(0.42 * top_ref + 0.33 * apex_ref + 0.25 * apex_y))
    top_c = max(2, min(top_c, imp_focus - min_gap * 3))

    top_lo = max(1, top_c - w_top)
    top_hi = min(n - 2, top_c + w_top)
    if top_hi >= imp_lo - min_gap:
        top_hi = max(top_lo + 1, imp_lo - min_gap - 1)
    if top_lo >= top_hi:
        top_lo = max(0, top_hi - max(2, w_top))

    # Downswing must sit between top and impact
    if top_hi + 1 > imp_lo - 1:
        mid = (top_hi + imp_lo) // 2
        top_hi = max(top_lo + 1, mid - 1)
        imp_lo = min(n - 2, mid + 1)

    a_end = _motion_onset_idx(spd, n) - 1
    a_end = max(0, min(a_end, top_lo - 4))
    if a_end < 1:
        a_end = 1

    t_end = _takeaway_end_idx(a_end, top_lo, sr, wy, spd)
    t_end = max(a_end + 1, min(t_end, top_lo - 2))

    f_start = _finish_stable_start(spd, vy, imp_hi, n, min_gap)
    f_start = max(imp_hi + min_gap + 1, min(f_start, n - 2))
    if f_start <= imp_hi + 1:
        f_start = min(n - 2, imp_hi + max(min_gap + 1, n // 20))

    # Exclusive cuts: phase k is [cuts[k], cuts[k+1]-1], cuts[8] == n
    cuts = [
        0,
        a_end + 1,
        t_end + 1,
        top_lo,
        top_hi + 1,
        imp_lo,
        imp_hi + 1,
        f_start,
        n,
    ]
    for i in range(1, 8):
        cuts[i] = max(cuts[i], cuts[i - 1] + 1)
    cuts[8] = n
    for i in range(7, 0, -1):
        cuts[i] = min(cuts[i], cuts[i + 1] - 1)
    for i in range(1, 8):
        cuts[i] = max(cuts[i], cuts[i - 1] + 1)

    windows: list[dict[str, Any]] = []
    base_conf = float(ev.get("phase_detector_confidence", 0.55))
    old_edges = _legacy_proportional_edges(n, top_ref, imp_ref)
    old_summary = [(_PHASE_ORDER[k], old_edges[k], old_edges[k + 1]) for k in range(8)]

    for k, phase in enumerate(_PHASE_ORDER):
        a = int(cuts[k])
        b = int(cuts[k + 1]) - 1
        a = max(0, min(a, n - 1))
        b = max(0, min(b, n - 1))
        if b < a:
            b = min(n - 1, a + 1)
        center = (a + b) // 2
        c = base_conf + (0.14 if phase in ("top", "impact") else 0.0)
        c = min(0.95, c + (0.06 if phase in ("follow_through", "finish") else 0.0))
        windows.append({
            "phase": phase,
            "start_pose_idx": a,
            "end_pose_idx": b,
            "center_pose_idx": center,
            "confidence": round(float(c), 4),
            "source": "motion_window",
        })

    new_summary = [(w["phase"], w["start_pose_idx"], w["end_pose_idx"]) for w in windows]

    logger.info(
        "[STELLAR_PRO][PHASE_WINDOW] anchors top_ref=%s imp_ref=%s imp_focus=%s top_c=%s f_start=%s",
        top_ref,
        imp_ref,
        imp_focus,
        top_c,
        f_start,
    )
    logger.info("[STELLAR_PRO][PHASE_WINDOW] old_edges_summary=%s", old_summary)
    logger.info("[STELLAR_PRO][PHASE_WINDOW] new_edges_summary=%s", new_summary)

    return windows, ev
