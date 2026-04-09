"""Lite A-only: selective phase fixes — preserve SwingNet A rows unless a phase is clearly wrong (no B)."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Sequence, Tuple

from services.lite_ab_mirror.constants import EVENT_SEQUENCE

logger = logging.getLogger(__name__)

_MIDDLE_EVENTS = (
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
)

_MAX_CANDIDATES = 256
# Weak prior weight — must not dominate motion / boundary evidence
_WEAK_PRIOR_WEIGHT = 0.06


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


def _nearest_timeline_k(frame_index: int, timeline: List[dict]) -> int:
    if not timeline:
        return 0
    best_k = 0
    best_d = 10**9
    for k, t in enumerate(timeline):
        fi = int(t.get("frame_index", -1))
        d = abs(fi - int(frame_index))
        if d < best_d:
            best_d = d
            best_k = k
    return best_k


def motion_score_at_frame(
    frame_index: int,
    timeline: List[dict],
    motions: Sequence[float],
) -> float:
    """Map a decode index to motion proxy (0..1 scale using max on timeline)."""
    if not timeline or not motions:
        return 0.5
    k = _nearest_timeline_k(frame_index, timeline)
    if k >= len(motions):
        k = len(motions) - 1
    raw = [float(motions[i]) for i in range(min(len(motions), len(timeline)))]
    if not raw:
        return 0.5
    mx = max(raw) or 1e-6
    v = float(motions[k]) if k < len(motions) else 0.0
    return max(0.0, min(1.0, v / mx))


def _weak_gaussian_prior(idx: int, ideal: float, span: float) -> float:
    if span <= 1e-6:
        return 1.0
    sigma = max(2.5, 0.1 * span)
    d = float(idx) - ideal
    return float(math.exp(-(d * d) / (2.0 * sigma * sigma)))


def _resolve_impact_hint_frames(
    by: Dict[str, dict],
    *,
    max_idx: int,
    explicit: int | None,
    timeline: List[dict],
    motions: Sequence[float],
) -> int:
    if explicit is not None:
        return max(0, min(int(explicit), max_idx))
    if timeline and motions and len(motions) > 2:
        # Peak motion on timeline → impact proxy
        m = list(motions[1:]) if len(motions) > 1 else list(motions)
        if m:
            k_peak = int(max(range(len(m)), key=lambda i: m[i]))
            k_peak = max(0, min(k_peak, len(timeline) - 1))
            return max(0, min(int(timeline[k_peak].get("frame_index", max_idx // 2)), max_idx))
    if "Impact" in by:
        return max(0, min(int(by["Impact"].get("frame_index", 0)), max_idx))
    top = by.get("Top")
    fin = by.get("Finish")
    if top is not None and fin is not None:
        return max(0, min(int((int(top["frame_index"]) + int(fin["frame_index"])) // 2), max_idx))
    return max(0, max_idx // 2)


# --- should_refine: only clearly broken phases ---------------------------------


def _gap_top_impact(by: Dict[str, dict]) -> int:
    t = int(by["Top"]["frame_index"])
    i = int(by["Impact"]["frame_index"])
    return max(0, i - t)


def should_refine_toeup(by: Dict[str, dict]) -> bool:
    addr = int(by["Address"]["frame_index"])
    toe = int(by["Toe-up"]["frame_index"])
    mbs = int(by["Mid-backswing"]["frame_index"])
    if toe <= addr or toe >= mbs:
        return True
    if mbs - addr <= 2:
        return False
    return toe <= addr + 1 or toe >= mbs - 1


def should_refine_mid_backswing(by: Dict[str, dict]) -> bool:
    toe = int(by["Toe-up"]["frame_index"])
    mbs = int(by["Mid-backswing"]["frame_index"])
    top = int(by["Top"]["frame_index"])
    if mbs <= toe or mbs >= top:
        return True
    if top - toe <= 2:
        return False
    return mbs <= toe + 1 or mbs >= top - 1


def should_refine_top(by: Dict[str, dict]) -> bool:
    mbs = int(by["Mid-backswing"]["frame_index"])
    top = int(by["Top"]["frame_index"])
    mds = int(by["Mid-downswing"]["frame_index"])
    if top <= mbs or top >= mds:
        return True
    if mds - mbs <= 2:
        return False
    return top <= mbs + 1 or top >= mds - 1


def should_refine_mid_downswing(by: Dict[str, dict]) -> bool:
    """Conservative: only fix clear corridor violations or hugging boundaries."""
    top = int(by["Top"]["frame_index"])
    mds = int(by["Mid-downswing"]["frame_index"])
    imp = int(by["Impact"]["frame_index"])
    if mds <= top or mds >= imp:
        return True
    corridor = imp - top
    if corridor <= 3:
        return False
    # Too close to Top or Impact (not a real "mid" downswing)
    if mds <= top + max(1, corridor // 12):
        return True
    if mds >= imp - max(1, corridor // 12):
        return True
    rel = (mds - top) / float(corridor)
    if rel < 0.08 or rel > 0.92:
        return True
    return False


def should_refine_impact(
    by: Dict[str, dict],
    *,
    hint_fi: int,
    timeline: List[dict],
    motions: Sequence[float],
) -> bool:
    top = int(by["Top"]["frame_index"])
    imp = int(by["Impact"]["frame_index"])
    fin = int(by["Finish"]["frame_index"])
    if imp <= top or imp >= fin:
        return True
    # Motion peak alignment when we have evidence
    if timeline and motions and len(motions) > 2:
        w = max(5, min(28, (fin - top) // 5))
        lo = max(top + 1, hint_fi - w)
        hi_b = min(fin - 1, hint_fi + w)
        if imp < lo or imp > hi_b:
            return True
    return False


def should_refine_mid_follow(by: Dict[str, dict]) -> bool:
    imp = int(by["Impact"]["frame_index"])
    mft = int(by["Mid-follow-through"]["frame_index"])
    fin = int(by["Finish"]["frame_index"])
    if mft <= imp or mft >= fin:
        return True
    if fin - imp <= 2:
        return False
    return mft <= imp + 1 or mft >= fin - 1


_SELECTORS = {
    "Toe-up": should_refine_toeup,
    "Mid-backswing": should_refine_mid_backswing,
    "Top": should_refine_top,
    "Mid-downswing": should_refine_mid_downswing,
    "Impact": None,  # special: needs hint
    "Mid-follow-through": should_refine_mid_follow,
}


def _pick_with_motion(
    candidates: List[int],
    *,
    timeline: List[dict],
    motions: Sequence[float],
    weak_ideal: float | None,
    weak_span: float,
    fallback: int,
) -> Tuple[int, float]:
    if not candidates:
        return fallback, 0.0
    best_i = candidates[0]
    best_s = -1.0
    for idx in candidates:
        ms = motion_score_at_frame(idx, timeline, motions)
        prior = 1.0
        if weak_ideal is not None and weak_span > 0:
            prior = _weak_gaussian_prior(idx, weak_ideal, weak_span)
        s = ms + _WEAK_PRIOR_WEIGHT * prior
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
    Single-pass selective refine: keep A rows unless ``should_refine_*`` says otherwise.
    Uses timeline + motions for scoring; weak Gaussian only as tie-breaker.
    """
    _ = poses  # reserved — future pose-based checks; do not destabilize A with guesses
    meta = dict(preprocess_meta or {})
    hi = max_frame_index
    if hi is None:
        hi = int(meta.get("max_frame_index", -1))
    if hi is None or hi < 0:
        hi = max((int(x.get("frame_index", 0)) for x in analysis_frames), default=-1)
    if hi < 0:
        hi = 0

    tl = list(timeline or [])
    mo = list(motions or [])

    working = ensure_eight_keyframe_rows(rows, max_frame_index=hi)
    by = _rows_by_event(working)
    hint_fi = _resolve_impact_hint_frames(by, max_idx=hi, explicit=impact_hint_frame_index, timeline=tl, motions=mo)

    refined_events: Dict[str, Dict[str, Any]] = {}
    fail_reasons: List[str] = []

    def snapshot_original() -> Dict[str, int]:
        return {str(r.get("event_name")): int(r.get("frame_index", 0)) for r in working}

    original_idx = snapshot_original()

    # --- single pass: middle 6 only, in phase order
    for ev in _MIDDLE_EVENTS:
        row = by.get(ev)
        if row is None:
            continue
        orig_fi = int(row.get("frame_index", 0))

        need_fix = False
        if ev == "Impact":
            need_fix = should_refine_impact(by, hint_fi=hint_fi, timeline=tl, motions=mo)
        else:
            fn = _SELECTORS.get(ev)
            if fn is not None:
                need_fix = bool(fn(by))

        if not need_fix:
            refined_events[ev] = {"action": "kept", "frame_index": orig_fi}
            continue

        # 无 timeline/motion 时不强行按弱先验改帧 — 保留 A 原始结果
        if len(tl) < 2 or len(mo) < 2:
            refined_events[ev] = {"action": "kept_no_motion_evidence", "frame_index": orig_fi}
            continue

        addr = int(by["Address"]["frame_index"])
        toe = int(by["Toe-up"]["frame_index"])
        mbs = int(by["Mid-backswing"]["frame_index"])
        top = int(by["Top"]["frame_index"])
        mds = int(by["Mid-downswing"]["frame_index"])
        imp = int(by["Impact"]["frame_index"])
        fin = int(by["Finish"]["frame_index"])

        pick = orig_fi
        score = 0.0

        if ev == "Toe-up":
            lo, hi_b = addr + 1, max(addr + 2, mbs - 1)
            cand = _candidates_in_window(lo, hi_b, analysis_frames, max_frame_index=hi)
            span = max(1, mbs - addr)
            ideal = addr + 0.28 * span
            pick, score = _pick_with_motion(
                cand, timeline=tl, motions=mo, weak_ideal=ideal, weak_span=float(span), fallback=orig_fi
            )
        elif ev == "Mid-backswing":
            lo, hi_b = toe + 1, max(toe + 2, top - 1)
            cand = _candidates_in_window(lo, hi_b, analysis_frames, max_frame_index=hi)
            span = max(1, top - toe)
            ideal = toe + 0.45 * span
            pick, score = _pick_with_motion(
                cand, timeline=tl, motions=mo, weak_ideal=ideal, weak_span=float(span), fallback=orig_fi
            )
        elif ev == "Top":
            lo = max(mbs + 1, top - max(6, (hi - addr) // 30))
            hi_b = min(mds - 1, top + max(6, (hi - addr) // 30))
            cand = _candidates_in_window(lo, hi_b, analysis_frames, max_frame_index=hi)
            span = max(1, hi_b - lo)
            ideal = float(lo + hi_b) / 2.0
            pick, score = _pick_with_motion(
                cand, timeline=tl, motions=mo, weak_ideal=ideal, weak_span=float(span), fallback=orig_fi
            )
        elif ev == "Mid-downswing":
            lo, hi_b = top + 1, max(top + 2, imp - 1)
            cand = _candidates_in_window(lo, hi_b, analysis_frames, max_frame_index=hi)
            span = max(1, imp - top)
            ideal = top + 0.42 * span
            pick, score = _pick_with_motion(
                cand, timeline=tl, motions=mo, weak_ideal=ideal, weak_span=float(span), fallback=orig_fi
            )
        elif ev == "Impact":
            w = max(6, min(30, (fin - top) // 6 or 10))
            lo = max(top + 1, hint_fi - w)
            hi_b = min(fin - 1, hint_fi + w)
            cand = _candidates_in_window(lo, hi_b, analysis_frames, max_frame_index=hi)
            span = max(1, hi_b - lo)
            pick, score = _pick_with_motion(
                cand, timeline=tl, motions=mo, weak_ideal=float(hint_fi), weak_span=float(span), fallback=orig_fi
            )
        elif ev == "Mid-follow-through":
            lo, hi_b = imp + 1, max(imp + 2, fin - 1)
            cand = _candidates_in_window(lo, hi_b, analysis_frames, max_frame_index=hi)
            span = max(1, fin - imp)
            ideal = imp + 0.38 * span
            pick, score = _pick_with_motion(
                cand, timeline=tl, motions=mo, weak_ideal=ideal, weak_span=float(span), fallback=orig_fi
            )

        if pick == orig_fi and need_fix:
            # Could not move — report only if still structurally bad
            if ev == "Mid-downswing" and (mds <= top or mds >= imp):
                fail_reasons.append("mid_downswing_unresolved")
            refined_events[ev] = {"action": "refine_skipped", "frame_index": orig_fi, "score": score}
        else:
            row["frame_index"] = int(pick)
            row["confidence"] = max(float(row.get("confidence") or 0.0), min(0.95, 0.35 + min(0.5, score)))
            refined_events[ev] = {"action": "refined", "from": orig_fi, "to": int(pick), "score": score}
            by[ev] = row

    working = [by[e] for e in EVENT_SEQUENCE]
    _enforce_strict_monotonic(working, hi)

    semantic_debug: Dict[str, Any] = {
        "mode": "selective_single_pass",
        "max_frame_index": hi,
        "impact_hint_frame_index": hint_fi,
        "timeline_len": len(tl),
        "motions_len": len(mo),
        "original_frame_index": original_idx,
        "per_event": refined_events,
    }

    # Post-check: only add fail reasons if still broken (no blanket semantic_invalid)
    by2 = _rows_by_event(working)
    if _gap_top_impact(by2) < 4:
        fail_reasons.append("top_impact_gap_invalid_post_semantic")
    t2 = int(by2["Top"]["frame_index"])
    i2 = int(by2["Impact"]["frame_index"])
    md2 = int(by2["Mid-downswing"]["frame_index"])
    if not (t2 < md2 < i2):
        fail_reasons.append("mid_downswing_corridor_invalid")

    logger.info(
        "[phase_semantic] selective max_idx=%s hint=%s fails=%s events=%s",
        hi,
        hint_fi,
        fail_reasons,
        {k: v.get("action") for k, v in refined_events.items()},
    )
    return working, fail_reasons, semantic_debug


__all__ = [
    "ensure_eight_keyframe_rows",
    "refine_lite_a_rows_with_phase_semantics",
    "motion_score_at_frame",
]
