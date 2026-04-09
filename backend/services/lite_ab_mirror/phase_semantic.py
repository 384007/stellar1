"""Lite A-only: semantic checks + local window reselection for the 6 middle swing phases (no B path)."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Tuple

from services.lite_ab_mirror.constants import EVENT_SEQUENCE

logger = logging.getLogger(__name__)

_EVENT_TO_SEMANTIC_REASON: Dict[str, str] = {
    "Toe-up": "toeup_semantic_invalid",
    "Mid-backswing": "mid_backswing_semantic_invalid",
    "Top": "top_semantic_invalid",
    "Mid-downswing": "mid_downswing_semantic_invalid",
    "Impact": "impact_semantic_invalid",
    "Mid-follow-through": "mid_followthrough_semantic_invalid",
}

_MAX_CANDIDATES = 320
_PASSES = 5
_SOFT_SCORE_FLOOR = 0.12


def _rows_by_event(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in rows:
        ev = str(r.get("event_name") or "")
        if ev:
            out[ev] = r
    return out


def _ordered_rows(rows: List[dict]) -> List[dict]:
    by_e = _rows_by_event(rows)
    ordered: List[dict] = []
    for ev in EVENT_SEQUENCE:
        if ev in by_e:
            ordered.append(copy.deepcopy(by_e[ev]))
        else:
            ordered.append({"event_name": ev, "frame_index": 0, "confidence": 0.25})
    return ordered


def _enforce_strict_monotonic(rows: List[dict], max_idx: int) -> None:
    """Guarantee strictly increasing frame_index within [0, max_idx]."""
    n = len(rows)
    hi = max(0, int(max_idx))
    prev = -1
    for i in range(n):
        slack = n - 1 - i
        upper = hi - slack
        fi = int(rows[i].get("frame_index", 0))
        fi = max(prev + 1, min(fi, upper))
        rows[i]["frame_index"] = fi
        prev = fi


def _spread_even_if_needed(rows: List[dict], max_idx: int) -> None:
    hi = max(1, int(max_idx))
    n = len(rows)
    for i in range(n):
        rows[i]["frame_index"] = int(round(i * hi / max(1, n - 1)))
        rows[i]["confidence"] = float(rows[i].get("confidence") or 0.25)
    _enforce_strict_monotonic(rows, hi)


def ensure_eight_keyframe_rows(
    rows: List[dict],
    *,
    max_frame_index: int,
) -> List[dict]:
    """Always return 8 rows in ``EVENT_SEQUENCE`` order with monotonic indices."""
    ordered = _ordered_rows(rows)
    hi = max(0, int(max_frame_index))
    if hi <= 0:
        for i, row in enumerate(ordered):
            row["frame_index"] = i
        return ordered

    known = [int(r.get("frame_index", 0)) for r in ordered if float(r.get("confidence", 0) or 0) > 0.05]
    if not known or max(known) - min(known) < 2:
        _spread_even_if_needed(ordered, hi)
        return ordered

    _enforce_strict_monotonic(ordered, hi)
    return ordered


def _linspace_indices(lo: int, hi: int, n: int) -> List[int]:
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo
    if n <= 1:
        return [lo]
    span = hi - lo
    return [int(round(lo + span * i / (n - 1))) for i in range(n)]


def _candidates_in_window(
    lo: int,
    hi: int,
    analysis_frames: List[dict],
    *,
    max_frame_index: int,
    max_points: int = _MAX_CANDIDATES,
) -> List[int]:
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo
    mx = max(0, int(max_frame_index))
    lo = max(0, min(lo, mx))
    hi = max(0, min(hi, mx))
    if lo >= hi:
        return [lo]
    pool = sorted(
        {int(f.get("frame_index", -1)) for f in analysis_frames if int(f.get("frame_index", -1)) >= 0}
    )
    cand = [i for i in pool if lo <= i <= hi]
    if len(cand) > max_points:
        step = max(1, len(cand) // max_points)
        cand = cand[::step][:max_points]
    if not cand:
        span = hi - lo + 1
        if span <= max_points:
            cand = list(range(lo, hi + 1))
        else:
            cand = _linspace_indices(lo, hi, max_points)
    return sorted({int(x) for x in cand if 0 <= int(x) <= mx})


def _gaussian_score(idx: int, ideal: float, sigma: float) -> float:
    if sigma <= 1e-6:
        return 1.0 if abs(float(idx) - ideal) < 0.5 else 0.0
    d = float(idx) - ideal
    return float(math.exp(-(d * d) / (2.0 * sigma * sigma)))


def _resolve_impact_hint(
    by: Dict[str, dict],
    *,
    max_idx: int,
    explicit: int | None,
) -> int:
    if explicit is not None:
        return max(0, min(int(explicit), max_idx))
    if "Impact" in by:
        return max(0, min(int(by["Impact"].get("frame_index", 0)), max_idx))
    top = by.get("Top")
    fin = by.get("Finish")
    if top is not None and fin is not None:
        return max(0, min(int((int(top["frame_index"]) + int(fin["frame_index"])) // 2), max_idx))
    return max(0, max_idx // 2)


def _score_toeup(idx: int, addr: int, mid_bs: int) -> float:
    if idx <= addr or idx >= mid_bs:
        return 0.0
    span = max(1, mid_bs - addr)
    ideal = addr + 0.28 * span
    sigma = max(3.0, 0.12 * span)
    return _gaussian_score(idx, ideal, sigma)


def _score_mid_back(idx: int, toe: int, top: int) -> float:
    if idx <= toe or idx >= top:
        return 0.0
    span = max(1, top - toe)
    ideal = toe + 0.45 * span
    sigma = max(3.0, 0.11 * span)
    return _gaussian_score(idx, ideal, sigma)


def _score_top(idx: int, mbs: int, mds: int, cur: int, max_idx: int) -> float:
    w = max(6, min(24, (mds - mbs) // 3 or 8))
    lo = max(mbs + 1, cur - w)
    hi = min(mds - 1, cur + w)
    if idx < lo or idx > hi:
        return 0.0
    if idx <= mbs or idx >= mds:
        return 0.0
    span = max(1, hi - lo)
    ideal = float(lo + hi) / 2.0
    sigma = max(2.5, 0.08 * span)
    return _gaussian_score(idx, ideal, sigma)


def _score_mid_down(idx: int, top: int, imp: int) -> float:
    if idx <= top or idx >= imp:
        return 0.0
    span = max(1, imp - top)
    # Not at fixed offset from Impact — favor mid corridor
    ideal = top + 0.42 * span
    sigma = max(3.5, 0.12 * span)
    return _gaussian_score(idx, ideal, sigma)


def _score_impact(idx: int, hint: int, top: int, fin: int) -> float:
    w = max(5, min(30, (fin - top) // 6 or 10))
    lo = max(top + 1, hint - w)
    hi = min(fin - 1, hint + w)
    if idx < lo or idx > hi:
        return 0.0
    sigma = max(3.0, 0.06 * max(1, hi - lo))
    return _gaussian_score(idx, float(hint), sigma)


def _score_mid_follow(idx: int, imp: int, fin: int) -> float:
    if idx <= imp or idx >= fin:
        return 0.0
    span = max(1, fin - imp)
    ideal = imp + 0.38 * span
    sigma = max(3.5, 0.11 * span)
    return _gaussian_score(idx, ideal, sigma)


def _best_pick(
    candidates: List[int],
    score_fn,
    *,
    fallback: int,
) -> Tuple[int, float]:
    if not candidates:
        return fallback, 0.0
    best_i = candidates[0]
    best_s = -1.0
    for idx in candidates:
        s = float(score_fn(idx))
        if s > best_s:
            best_s = s
            best_i = idx
    return best_i, best_s


def refine_lite_a_rows_with_phase_semantics(
    rows: List[dict],
    *,
    analysis_frames: List[dict],
    preprocess_meta: dict,
    poses: List[dict] | None = None,
    timeline: List[dict] | None = None,
    motions: List[float] | None = None,
    impact_hint_frame_index: int | None = None,
    max_frame_index: int | None = None,
) -> Tuple[List[dict], List[str], Dict[str, Any]]:
    """
    Local window reselection for middle 6 phases; always returns 8 monotonic rows.

    ``poses`` / ``timeline`` / ``motions`` are accepted for API compatibility; scoring uses
    timeline indices and optional impact hint only (no full re-infer).
    """
    _ = (poses, timeline, motions)
    meta = dict(preprocess_meta or {})
    hi = max_frame_index
    if hi is None:
        hi = int(meta.get("max_frame_index", -1))
    if hi is None or hi < 0:
        hi = max((int(x.get("frame_index", 0)) for x in analysis_frames), default=-1)
    if hi < 0:
        hi = 0

    working = ensure_eight_keyframe_rows(rows, max_frame_index=hi)
    semantic_debug: Dict[str, Any] = {"passes": [], "max_frame_index": hi}

    fail_reasons: List[str] = []

    for p in range(_PASSES):
        by = _rows_by_event(working)
        hint = _resolve_impact_hint(by, max_idx=hi, explicit=impact_hint_frame_index)
        addr = int(by["Address"]["frame_index"])
        toe = int(by["Toe-up"]["frame_index"])
        mbs = int(by["Mid-backswing"]["frame_index"])
        top = int(by["Top"]["frame_index"])
        mds = int(by["Mid-downswing"]["frame_index"])
        imp = int(by["Impact"]["frame_index"])
        fin = int(by["Finish"]["frame_index"])

        pass_log: Dict[str, Any] = {"pass": p}

        # --- Toe-up: Address → Mid-backswing
        lo, hi_t = addr + 1, max(addr + 2, mbs - 1)
        cand = _candidates_in_window(lo, hi_t, analysis_frames, max_frame_index=hi)
        pick, sc = _best_pick(cand, lambda i: _score_toeup(i, addr, mbs), fallback=toe)
        by["Toe-up"]["frame_index"] = pick
        by["Toe-up"]["confidence"] = max(float(by["Toe-up"].get("confidence") or 0.0), min(0.95, 0.35 + sc))
        pass_log["Toe-up"] = {"idx": pick, "score": sc}
        if sc < _SOFT_SCORE_FLOOR and _EVENT_TO_SEMANTIC_REASON["Toe-up"] not in fail_reasons:
            fail_reasons.append(_EVENT_TO_SEMANTIC_REASON["Toe-up"])

        # --- Mid-backswing: Toe-up → Top
        toe = int(by["Toe-up"]["frame_index"])
        lo, hi_t = toe + 1, max(toe + 2, top - 1)
        cand = _candidates_in_window(lo, hi_t, analysis_frames, max_frame_index=hi)
        pick, sc = _best_pick(cand, lambda i: _score_mid_back(i, toe, top), fallback=mbs)
        by["Mid-backswing"]["frame_index"] = pick
        by["Mid-backswing"]["confidence"] = max(
            float(by["Mid-backswing"].get("confidence") or 0.0), min(0.95, 0.35 + sc)
        )
        pass_log["Mid-backswing"] = {"idx": pick, "score": sc}
        if sc < _SOFT_SCORE_FLOOR and _EVENT_TO_SEMANTIC_REASON["Mid-backswing"] not in fail_reasons:
            fail_reasons.append(_EVENT_TO_SEMANTIC_REASON["Mid-backswing"])

        mbs = int(by["Mid-backswing"]["frame_index"])
        top = int(by["Top"]["frame_index"])
        mds = int(by["Mid-downswing"]["frame_index"])

        # --- Top: corridor between Mid-bs and Mid-down, local window around current Top
        lo = max(mbs + 1, top - max(8, (hi - addr) // 25))
        hi_t = min(mds - 1, top + max(8, (hi - addr) // 25))
        cand = _candidates_in_window(lo, hi_t, analysis_frames, max_frame_index=hi)
        pick, sc = _best_pick(cand, lambda i: _score_top(i, mbs, mds, top, hi), fallback=top)
        by["Top"]["frame_index"] = pick
        by["Top"]["confidence"] = max(float(by["Top"].get("confidence") or 0.0), min(0.95, 0.4 + sc))
        pass_log["Top"] = {"idx": pick, "score": sc}
        if sc < _SOFT_SCORE_FLOOR and _EVENT_TO_SEMANTIC_REASON["Top"] not in fail_reasons:
            fail_reasons.append(_EVENT_TO_SEMANTIC_REASON["Top"])

        top = int(by["Top"]["frame_index"])
        imp = int(by["Impact"]["frame_index"])
        mds = int(by["Mid-downswing"]["frame_index"])

        # --- Mid-downswing: Top → Impact (never reuse “Impact minus fixed offset” exclusively)
        lo, hi_t = top + 1, max(top + 2, imp - 1)
        cand = _candidates_in_window(lo, hi_t, analysis_frames, max_frame_index=hi)
        pick, sc = _best_pick(cand, lambda i: _score_mid_down(i, top, imp), fallback=mds)
        by["Mid-downswing"]["frame_index"] = pick
        by["Mid-downswing"]["confidence"] = max(
            float(by["Mid-downswing"].get("confidence") or 0.0), min(0.95, 0.35 + sc)
        )
        pass_log["Mid-downswing"] = {"idx": pick, "score": sc}
        if sc < _SOFT_SCORE_FLOOR and _EVENT_TO_SEMANTIC_REASON["Mid-downswing"] not in fail_reasons:
            fail_reasons.append(_EVENT_TO_SEMANTIC_REASON["Mid-downswing"])

        mds = int(by["Mid-downswing"]["frame_index"])
        hint = _resolve_impact_hint(by, max_idx=hi, explicit=impact_hint_frame_index)
        fin = int(by["Finish"]["frame_index"])

        # --- Impact: around hint
        lo = max(top + 1, hint - max(6, (fin - top) // 8))
        hi_t = min(fin - 1, hint + max(6, (fin - top) // 8))
        cand = _candidates_in_window(lo, hi_t, analysis_frames, max_frame_index=hi)
        pick, sc = _best_pick(cand, lambda i: _score_impact(i, hint, top, fin), fallback=imp)
        by["Impact"]["frame_index"] = pick
        by["Impact"]["confidence"] = max(float(by["Impact"].get("confidence") or 0.0), min(0.95, 0.45 + sc))
        pass_log["Impact"] = {"idx": pick, "score": sc}
        if sc < _SOFT_SCORE_FLOOR and _EVENT_TO_SEMANTIC_REASON["Impact"] not in fail_reasons:
            fail_reasons.append(_EVENT_TO_SEMANTIC_REASON["Impact"])

        imp = int(by["Impact"]["frame_index"])

        # --- Mid-follow-through: Impact → Finish
        lo, hi_t = imp + 1, max(imp + 2, fin - 1)
        cand = _candidates_in_window(lo, hi_t, analysis_frames, max_frame_index=hi)
        mft = int(by["Mid-follow-through"]["frame_index"])
        pick, sc = _best_pick(cand, lambda i: _score_mid_follow(i, imp, fin), fallback=mft)
        by["Mid-follow-through"]["frame_index"] = pick
        by["Mid-follow-through"]["confidence"] = max(
            float(by["Mid-follow-through"].get("confidence") or 0.0), min(0.95, 0.35 + sc)
        )
        pass_log["Mid-follow-through"] = {"idx": pick, "score": sc}
        if sc < _SOFT_SCORE_FLOOR and _EVENT_TO_SEMANTIC_REASON["Mid-follow-through"] not in fail_reasons:
            fail_reasons.append(_EVENT_TO_SEMANTIC_REASON["Mid-follow-through"])

        working = [by[e] for e in EVENT_SEQUENCE]
        _enforce_strict_monotonic(working, hi)
        semantic_debug["passes"].append(pass_log)

    logger.info(
        "[phase_semantic] refined middle phases max_idx=%s reasons=%s",
        hi,
        fail_reasons,
    )
    return working, fail_reasons, semantic_debug


__all__ = [
    "ensure_eight_keyframe_rows",
    "refine_lite_a_rows_with_phase_semantics",
]
