"""Pro v2 — eight phase keyframes via per-phase motion helpers (deterministic, no AI)."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.pro_v2_dense_scan_service import DenseFrame
from services.pro_v2_frame_read import read_frames_bgr_at_indices

logger = logging.getLogger(__name__)

PHASE_ORDER: tuple[str, ...] = (
    "address",
    "takeaway",
    "backswing",
    "top",
    "downswing",
    "impact",
    "follow_through",
    "finish",
)

PHASE_LABELS: dict[str, tuple[str, str]] = {
    "address": ("Address", "站姿"),
    "takeaway": ("Takeaway", "起杆"),
    "backswing": ("Backswing", "上杆"),
    "top": ("Top", "顶点"),
    "downswing": ("Downswing", "下杆"),
    "impact": ("Impact", "触球"),
    "follow_through": ("Follow-through", "送杆"),
    "finish": ("Finish", "收杆"),
}


def _jpeg_b64(frame_bgr: Any, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _E(dense: list[DenseFrame]) -> np.ndarray:
    return np.array([d.motion_energy_smooth for d in dense], dtype=np.float64)


def _pick_address(dense: list[DenseFrame], E: np.ndarray, n: int) -> int:
    """Stable low-motion near the front; prefer local valley."""
    hi = min(max(3, int(n * 0.12)), n - 1)
    region = list(range(0, hi + 1))
    valleys = [i for i in region if dense[i].is_local_valley]
    if valleys:
        return int(valleys[int(np.argmin(E[valleys]))])
    return int(np.argmin(E[0 : hi + 1]))


def _pick_takeaway(
    dense: list[DenseFrame],
    E: np.ndarray,
    address_idx: int,
    n: int,
    *,
    late_shift: int = 0,
) -> int:
    """First sustained rise after address."""
    start = min(address_idx + 1 + max(0, int(late_shift)), n - 4)
    base = float(np.mean(E[max(0, address_idx - 1) : min(address_idx + 2, n)]) + 1e-9)
    for i in range(start, n - 2):
        if E[i] < base * 1.12:
            continue
        win = E[i : min(i + 4, n)]
        if float(np.mean(win)) > base * 1.28:
            return int(i)
    return min(address_idx + max(2, n // 40), n - 6)


def _strike_burst_range(
    E: np.ndarray,
    takeaway_idx: int,
    n: int,
    *,
    percentile: int = 80,
) -> tuple[int, int]:
    """High-energy contiguous segment (strike band), not whole clip."""
    if takeaway_idx >= n - 2:
        return max(0, n - 4), n - 1
    pct = max(55, min(92, int(percentile)))
    tail = E[takeaway_idx:]
    thr = float(np.percentile(tail, pct))
    active = np.zeros(n, dtype=bool)
    active[takeaway_idx:] = E[takeaway_idx:] >= thr
    runs: list[tuple[int, int]] = []
    i = takeaway_idx
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    if not runs:
        k = takeaway_idx + int(np.argmax(E[takeaway_idx:]))
        return max(takeaway_idx, k - 2), min(n - 1, k + 3)
    best = max(runs, key=lambda ab: float(np.sum(E[ab[0] : ab[1] + 1])))
    return best[0], best[1]


def _pick_impact_rough(E: np.ndarray, b0: int, b1: int, n: int, *, variant: int = 0) -> int:
    """Strongest frame inside strike burst (local to band, not global argmax over clip)."""
    b0 = max(0, min(b0, n - 1))
    b1 = max(b0, min(b1, n - 1))
    seg = E[b0 : b1 + 1]
    if seg.size <= 0:
        return b0
    if variant > 0:
        order = np.argsort(seg)
        k = max(0, min(seg.size - 1, seg.size - 1 - min(variant, seg.size - 1)))
        rel = int(order[k])
        return b0 + rel
    rel = int(np.argmax(seg))
    return b0 + rel


def _pick_top(
    dense: list[DenseFrame],
    E: np.ndarray,
    takeaway_idx: int,
    impact_idx: int,
    n: int,
    *,
    use_second_valley: bool = False,
) -> int:
    """Valley / direction-change pocket before downswing burst."""
    lo = min(takeaway_idx + 1, n - 3)
    hi = max(lo + 1, impact_idx - 1)
    if hi <= lo:
        return min(lo, n - 2)
    region = list(range(lo, hi))
    valleys = [i for i in region if dense[i].is_local_valley]
    if valleys:
        order = sorted(valleys, key=lambda i: float(E[i]))
        if use_second_valley and len(order) >= 2:
            return int(order[1])
        return int(order[0])
    return int(lo + np.argmin(E[lo:hi]))


def _pick_backswing(
    E: np.ndarray,
    takeaway_idx: int,
    top_idx: int,
    *,
    median_mode: bool = False,
) -> int:
    """Representative rising motion before top."""
    lo = min(takeaway_idx + 1, top_idx - 1)
    hi = max(lo + 1, top_idx)
    if hi <= lo + 1:
        return min(takeaway_idx + 1, top_idx - 1) if top_idx > takeaway_idx + 1 else takeaway_idx + 1
    inner_lo = lo + max(1, (hi - lo) // 8)
    inner_hi = hi - max(1, (hi - lo) // 8)
    if inner_hi <= inner_lo:
        inner_lo, inner_hi = lo, hi - 1
    seg = E[inner_lo : inner_hi + 1]
    if median_mode and seg.size >= 3:
        order = np.argsort(seg)
        mid = int(order[len(order) // 2])
        return int(inner_lo + mid)
    return int(inner_lo + np.argmax(seg))


def _pick_downswing(
    E: np.ndarray,
    top_idx: int,
    impact_idx: int,
    *,
    dense_delta: int = 0,
) -> int:
    """Strong motion after top, strictly before impact."""
    lo = min(top_idx + 1, impact_idx - 2)
    hi = max(lo + 1, impact_idx - 1)
    if hi <= lo:
        base = max(top_idx + 1, impact_idx - 2)
    else:
        base = int(lo + np.argmax(E[lo : hi + 1]))
    dd = int(dense_delta)
    return max(top_idx + 1, min(base + dd, max(top_idx + 1, impact_idx - 2)))


def _pick_follow_through(
    dense: list[DenseFrame],
    E: np.ndarray,
    impact_idx: int,
    n: int,
    *,
    min_gap_delta: int = 0,
) -> int:
    """Pick post-impact release frame: first clear local peak with minimum spacing from impact."""
    min_gap = max(2, max(3, min(6, n // 40)) + int(min_gap_delta))
    start = min(impact_idx + min_gap, n - 3)
    if start >= n - 1:
        return min(n - 2, max(impact_idx + 1, n - 2))

    post_tail = E[min(impact_idx + 1, n - 1) :]
    post_p75 = float(np.percentile(post_tail, 75)) if len(post_tail) > 0 else float(E[impact_idx])
    imp_e = float(E[impact_idx]) + 1e-9
    y_scale = max((dense[j].motion_y for j in range(min(impact_idx + 1, n - 1), n)), default=1e-6) + 1e-9
    peak_floor = max(post_p75 * 0.88, imp_e * 0.45)

    # Preferred: first strong local peak that is clearly after strike but still high motion (release).
    for i in range(start, n - 1):
        if i <= 0 or i >= n - 1:
            continue
        if not (E[i] >= E[i - 1] and E[i] >= E[i + 1]):
            continue
        if E[i] < peak_floor:
            continue
        if E[i] > imp_e * 1.02:  # avoid relabeling strike-like spike as follow-through
            continue
        y_norm = dense[i].motion_y / y_scale
        if y_norm >= 0.22 or (E[i] >= post_p75 and E[i] >= E[i - 1] * 1.03):
            return int(i)

    # Fallback: first high-motion point after sufficient gap.
    for i in range(start, n - 1):
        if E[i] >= post_p75 * 0.9 and E[i] <= imp_e * 1.08:
            return int(i)

    return min(n - 2, impact_idx + min_gap + 1)


def _pick_finish(dense: list[DenseFrame], E: np.ndarray, follow_idx: int, n: int) -> int:
    """Pick stable tail frame after release: low motion + plateau, not defaulting to final frame."""
    min_gap = max(4, min(8, n // 28))
    lo = min(follow_idx + min_gap, n - 2)
    hi = n - 2
    if lo > hi:
        return min(n - 1, follow_idx + max(2, min_gap - 1))

    tail = E[lo : hi + 1]
    low_thr = float(np.percentile(tail, 35)) if len(tail) > 0 else float(E[min(lo, n - 1)])
    best_i = lo
    best_s = float("inf")
    for i in range(lo, hi + 1):
        if i <= 0 or i >= n - 1:
            continue
        local_span = E[max(lo, i - 2) : min(hi + 1, i + 3)]
        plateau = float(np.std(local_span)) if len(local_span) >= 2 else 0.0
        valley_bonus = 0.0 if dense[i].is_local_valley else 0.02
        tail_bias = 0.02 * ((i - lo) / max(1, hi - lo))  # prefer later stable hold, but not extreme end
        score = float(E[i]) + 0.55 * plateau + valley_bonus + tail_bias
        if E[i] <= low_thr * 1.15 and score < best_s:
            best_s = score
            best_i = i

    if best_s < float("inf"):
        return int(best_i)
    return int(lo + np.argmin(E[lo : hi + 1]))


def _late_strip_spacing_pass(order_idx: list[int], n: int) -> list[int]:
    """Keep downswing < impact < follow_through < finish with minimum dense gaps."""
    o = list(order_idx)
    min_step = max(2, min(10, n // 35))
    for _ in range(32):
        changed = False
        if o[4] <= o[3]:
            o[4] = min(n - 1, o[3] + min_step)
            changed = True
        if o[5] <= o[4]:
            o[5] = min(n - 1, o[4] + min_step)
            changed = True
        if o[6] <= o[5]:
            o[6] = min(n - 1, o[5] + min_step)
            changed = True
        if o[7] <= o[6]:
            o[7] = min(n - 1, o[6] + min_step)
            changed = True
        for k in range(1, 8):
            if o[k] <= o[k - 1]:
                o[k] = min(n - 1, o[k - 1] + 1)
                changed = True
        if not changed:
            break
    return o


def _late_strip_gap_dense(o: list[int]) -> tuple[int, int]:
    return int(o[6] - o[5]), int(o[7] - o[6])


def pick_eight_keyframes_motion_only(
    analysis_video_path: str,
    dense: list[DenseFrame],
    *,
    screen_mode: bool = False,
    picker_variant: int = 0,
    picker_tuning: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Per-phase motion picks; v2 is frame-driven (no pose indices).

    picker_tuning merges routing + retry_reasons (see pro_v2_strategy_profiles).
    picker_variant is legacy; when picker_tuning is None, pv still nudges late-strip.
    """
    if len(dense) < 16:
        raise RuntimeError("pro_v2: insufficient dense frames for 8-phase pick")

    n = len(dense)
    E = _E(dense)
    tun = dict(picker_tuning or {})
    pv = max(0, int(tun.get("legacy_picker_variant", picker_variant)))
    strike_pct = int(tun.get("strike_percentile", 80))
    imp_var = int(tun.get("impact_variant", min(2, pv)))

    addr = _pick_address(dense, E, n)
    tw = _pick_takeaway(dense, E, addr, n, late_shift=int(tun.get("takeaway_late_shift", 0)))
    tw = max(addr + 1, min(tw, n - 5))

    b0, b1 = _strike_burst_range(E, tw, n, percentile=strike_pct)
    imp = _pick_impact_rough(E, b0, b1, n, variant=min(3, max(0, imp_var)))

    top = _pick_top(
        dense,
        E,
        tw,
        imp,
        n,
        use_second_valley=bool(tun.get("top_use_second_valley", False)),
    )
    top = max(tw + 1, min(top + int(tun.get("top_dense_delta", 0)), imp - 1))
    if int(tun.get("top_dense_delta", 0)) == 0 and pv > 0:
        nudge = min(4, max(1, (top - tw) // 5))
        top = max(tw + 1, top - nudge)

    bs = _pick_backswing(E, tw, top, median_mode=bool(tun.get("backswing_median_mode", False)))
    bs = max(tw + 1, min(bs, top - 1)) if top > tw + 2 else min(tw + 1, top - 1)

    ds = _pick_downswing(E, top, imp, dense_delta=int(tun.get("downswing_dense_delta", 0)))
    ds = max(top + 1, min(ds, imp - 1)) if imp > top + 2 else top + 1

    imp = max(ds + 1, min(imp, n - 3))
    ft = _pick_follow_through(
        dense,
        E,
        imp,
        n,
        min_gap_delta=int(tun.get("follow_min_gap_delta", 0)),
    )
    ft = max(imp + 1, min(ft + int(tun.get("release_dense_shift", 0)), n - 2))

    fin = _pick_finish(dense, E, ft, n)
    fin = max(ft + 1, min(fin, n - 1))

    order_idx = [addr, tw, bs, top, ds, imp, ft, fin]
    order_idx = _late_strip_spacing_pass(order_idx, n)
    late_min_gap = max(4, min(9, n // 30)) if screen_mode else max(3, min(7, n // 34))
    late_min_gap += int(tun.get("late_min_gap_extra", 0))
    late_min_gap += min(4, pv * 2)
    gap_if, gap_ff = _late_strip_gap_dense(order_idx)
    if gap_if < late_min_gap:
        repick_ft = _pick_follow_through(dense, E, int(order_idx[5]), n)
        order_idx[6] = max(int(order_idx[5]) + late_min_gap, min(repick_ft + (1 if screen_mode else 0), n - 2))
    gap_if, gap_ff = _late_strip_gap_dense(order_idx)
    if gap_ff < late_min_gap:
        repick_fin = _pick_finish(dense, E, int(order_idx[6]), n)
        order_idx[7] = max(int(order_idx[6]) + late_min_gap, min(repick_fin + (1 if screen_mode else 0), n - 1))
    order_idx = _late_strip_spacing_pass(order_idx, n)
    for i in range(1, 8):
        if order_idx[i] <= order_idx[i - 1]:
            order_idx[i] = min(n - 1, order_idx[i - 1] + 1)

    path = str(Path(analysis_video_path))
    frame_indices: list[int] = []
    phase_rows: list[tuple[str, int, float]] = []
    for phase, di in zip(PHASE_ORDER, order_idx, strict=True):
        di = max(0, min(int(di), n - 1))
        d = dense[di]
        f_idx = int(d.frame_index)
        t_s = float(d.timestamp_s)
        frame_indices.append(f_idx)
        phase_rows.append((phase, f_idx, t_s))

    bgr_by_idx = read_frames_bgr_at_indices(path, frame_indices)
    keyframes: list[dict[str, Any]] = []
    for phase, f_idx, t_s in phase_rows:
        frame = bgr_by_idx.get(f_idx)
        if frame is None:
            raise RuntimeError(f"pro_v2: failed to read frame {f_idx}")
        en, zh = PHASE_LABELS[phase]
        keyframes.append(
            {
                "phase": phase,
                "label_en": en,
                "label_zh": zh,
                "timestamp": t_s,
                "frame_index": f_idx,
                "source_frame_index": f_idx,
                # Legacy UI key: v2 is frame-only, not pose sampling.
                "source_pose_idx": f_idx,
                "image_base64": _jpeg_b64(frame),
            }
        )

    logger.info(
        "[PRO_V2][KEYFRAMES] strike_dense=(%s,%s) impact_dense=%s follow_dense=%s finish_dense=%s late_strip_gaps=(impact_follow=%s,follow_finish=%s) frame_indices=%s",
        dense[b0].frame_index,
        dense[b1].frame_index,
        int(order_idx[5]),
        int(order_idx[6]),
        int(order_idx[7]),
        int(order_idx[6] - order_idx[5]),
        int(order_idx[7] - order_idx[6]),
        [k["frame_index"] for k in keyframes],
    )
    logger.info(
        "[PRO_V2][GATE] screen_mode=%s late_min_gap=%s impact_follow_gap=%s follow_finish_gap=%s",
        "true" if screen_mode else "false",
        late_min_gap,
        int(order_idx[6] - order_idx[5]),
        int(order_idx[7] - order_idx[6]),
    )
    if pv > 0:
        logger.info("[PRO_V2][RETRY] picker_variant=%s late_min_gap_adjusted=%s", pv, late_min_gap)
    return keyframes
