"""Per-phase pose index — deterministic multi-signal scores + anchor refinement (no AI)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _series(features: dict[str, Any], key: str, n: int) -> np.ndarray:
    raw = features.get(key) or []
    arr = np.asarray(list(raw), dtype=np.float64)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size))
    return arr[:n]


def _nz_window(idxs: np.ndarray, vals: np.ndarray) -> np.ndarray:
    v = vals[idxs]
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-9:
        return np.zeros_like(v, dtype=np.float64)
    return (v - lo) / (hi - lo)


def _anchor_band(j: np.ndarray, c: int, a: int, e: int, half: int = 5) -> np.ndarray:
    cc = float(max(a, min(e, c)))
    dist = np.abs(j.astype(np.float64) - cc)
    return np.clip(1.0 - dist / float(max(half, 1)), 0.0, 1.0)


def _bell_rel(r: np.ndarray, lo: float = 0.35, hi: float = 0.72) -> np.ndarray:
    mid = 0.5 * (lo + hi)
    w = max(hi - lo, 0.08)
    return np.exp(-0.5 * ((r - mid) / w) ** 2)


def pick_phase_pose_index_from_window(
    phase: str,
    start_idx: int,
    end_idx: int,
    *,
    poses: list[dict],
    features: dict[str, Any],
    events: dict[str, Any],
    pick_meta_out: list[dict[str, Any]] | None = None,
) -> int:
    n = len(poses)
    a = max(0, min(int(start_idx), n - 1))
    e = max(a, min(int(end_idx), n - 1))
    if a >= e:
        if pick_meta_out is not None:
            pick_meta_out.append({"phase": phase, "idx": a, "score": 0.0, "note": "singleton_window"})
        return a

    wy = _series(features, "wrist_mid_y", n)
    vy = _series(features, "wrist_vy", n)
    spd = _series(features, "hand_speed", n)
    sr = _series(features, "shoulder_rotation", n)
    hr = _series(features, "hip_rotation", n)
    xf = _series(features, "x_factor_delta", n)
    st = _series(features, "spine_tilt_y", n)

    d_sr = np.zeros(n)
    d_sr[1:] = np.diff(sr)
    d_hr = np.zeros(n)
    d_hr[1:] = np.diff(hr)
    dst = np.zeros(n)
    dst[1:] = np.diff(st)

    idxs = np.arange(a, e + 1, dtype=np.int64)
    j = idxs.astype(np.float64)
    early = (e - idxs) / max(e - a, 1)
    late = (idxs - a) / max(e - a, 1)
    rel = late
    nz_spd = _nz_window(idxs, spd)
    nz_vy = _nz_window(idxs, np.abs(vy))
    nz_sr = _nz_window(idxs, sr)
    nz_wy = _nz_window(idxs, wy)
    nz_neg_wy = _nz_window(idxs, -wy)
    nz_xf = _nz_window(idxs, xf)
    nz_dsr_abs = _nz_window(idxs, np.abs(d_sr))
    nz_neg_vy = _nz_window(idxs, np.maximum(0.0, -vy))
    nz_pos_vy = _nz_window(idxs, np.maximum(0.0, vy))
    nz_dhr = _nz_window(idxs, np.abs(d_hr))
    nz_dst = _nz_window(idxs, np.abs(dst))

    top_ev = int(events.get("top_pose_idx", (a + e) // 2))
    imp_ev = int(events.get("impact_pose_idx", (a + e) // 2))

    s = np.zeros(len(idxs), dtype=np.float64)
    detail: dict[str, Any] = {}

    if phase == "top":
        band = _anchor_band(idxs, top_ev, a, e, 6)
        s = (
            0.24 * (1.0 - np.minimum(nz_vy, 1.0))
            + 0.22 * band
            + 0.18 * nz_sr
            + 0.16 * nz_neg_wy
            + 0.12 * nz_xf
            + 0.08 * (1.0 - nz_spd)
        )
        detail = {"anchor_top": top_ev, "blend": "coil+slow+xf+band"}
    elif phase == "impact":
        band = _anchor_band(idxs, imp_ev, a, e, 7)
        s = (
            0.34 * nz_spd
            + 0.22 * band
            + 0.18 * nz_neg_vy
            + 0.14 * nz_dsr_abs
            + 0.12 * nz_xf
        )
        detail = {"anchor_impact": imp_ev, "blend": "speed+down_vy+band"}
    elif phase == "address":
        s = (
            -0.38 * nz_spd
            -0.28 * nz_vy
            -0.18 * nz_dsr_abs
            -0.08 * nz_dst
            + 0.28 * early
        )
        detail = {"blend": "stillness+early"}
    elif phase == "takeaway":
        dsr_pos = _nz_window(idxs, np.maximum(0.0, d_sr))
        s = 0.32 * nz_spd + 0.28 * nz_pos_vy + 0.22 * dsr_pos + 0.18 * early
        detail = {"blend": "speed+up_vy+sr_rise+early"}
    elif phase == "backswing":
        s = 0.32 * nz_sr + 0.28 * nz_wy + 0.18 * nz_xf + 0.14 * late + 0.08 * nz_spd
        detail = {"blend": "rotation+height+xf+late"}
    elif phase == "downswing":
        s = 0.36 * nz_spd + 0.28 * nz_neg_vy + 0.18 * nz_dsr_abs + 0.18 * late
        detail = {"blend": "speed+down_vy+dsr+late"}
    elif phase == "follow_through":
        bell = _bell_rel(rel, 0.28, 0.78)
        s = (
            0.30 * nz_spd
            + 0.22 * nz_vy
            + 0.18 * nz_dhr
            + 0.16 * bell
            + 0.14 * (1.0 - early)
        )
        detail = {"blend": "speed+vy+hip+bell_midlate"}
    elif phase == "finish":
        s = (
            -0.36 * nz_spd
            -0.24 * nz_vy
            -0.18 * nz_dst
            -0.10 * nz_dsr_abs
            + 0.42 * late
        )
        detail = {"blend": "quiet_spine+late"}
    else:
        s = np.zeros(len(idxs), dtype=np.float64)

    j_best = int(np.argmax(s))
    picked = int(idxs[j_best])
    best = float(s[j_best])

    rec = {
        "phase": phase,
        "idx": picked,
        "score": round(best, 4),
        "window": [a, e],
        "detail": detail,
    }
    if pick_meta_out is not None:
        pick_meta_out.append(rec)

    logger.debug(
        "[STELLAR_PRO][MOTION_PICK] phase=%s window=[%s,%s] idx=%s score=%.4f",
        phase,
        a,
        e,
        picked,
        best,
    )
    return picked


def log_motion_pick_summary(meta: list[dict[str, Any]]) -> None:
    if not meta:
        return
    line = "; ".join(
        f"{m.get('phase')}→{m.get('idx')}(s={m.get('score')})"
        for m in meta
    )
    logger.info("[STELLAR_PRO][MOTION_PICK] per_phase_summary %s", line)
