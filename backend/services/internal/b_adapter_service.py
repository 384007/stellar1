from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from lib.prov3.keyframes.constants import EVENT_SEQUENCE
from services.golfdb_swingnet_service import swingnet_b_refine

FOCUS_EVENTS_DEFAULT = {"Top", "Impact", "Finish", "Mid-downswing"}


def _frame_index_set(rows: Sequence[dict]) -> set[int]:
    out: set[int] = set()
    for row in rows:
        try:
            out.add(int(row.get("frame_index", -1)))
        except (TypeError, ValueError):
            continue
    return {x for x in out if x >= 0}


def _nearest_frame(target: int, frames: set[int], *, max_dist: int | None = None) -> int | None:
    if not frames:
        return None
    nearest = min(frames, key=lambda x: abs(x - target))
    if max_dist is not None and abs(nearest - target) > max_dist:
        return None
    return nearest


def _event_focus_set(fail_reasons: Sequence[str], confidence: Dict[str, float]) -> set[str]:
    focus = set(FOCUS_EVENTS_DEFAULT)
    reasons = "|".join(str(x).lower() for x in fail_reasons)
    if "top" in reasons:
        focus.add("Top")
    if "impact" in reasons:
        focus.add("Impact")
    if "order" in reasons or "gap" in reasons:
        focus.update({"Top", "Impact", "Mid-downswing"})

    for name, conf in confidence.items():
        if float(conf) < 0.6:
            focus.add(str(name))
    return focus


def _candidate_confidence(item: Dict[str, object], frame_idx: int) -> float:
    base = float(item.get("confidence", 0.0))
    for c in list(item.get("top_k_candidates") or []):
        try:
            if int(c.get("frame_index", -1)) == frame_idx:
                base = max(base, float(c.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
    return base


def _build_focus_candidates(item: Dict[str, object], frame_idx: int) -> set[int]:
    candidates = {frame_idx}
    for c in list(item.get("top_k_candidates") or []):
        try:
            candidates.add(int(c.get("frame_index", frame_idx)))
        except (TypeError, ValueError):
            continue
    # small local window to allow local-peak correction around current anchor
    for delta in range(-6, 7):
        candidates.add(max(0, frame_idx + delta))
    return {x for x in candidates if x >= 0}


def _topk_rows(item: Dict[str, object]) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for c in list(item.get("top_k_candidates") or []):
        try:
            out.append((int(c.get("frame_index", -1)), float(c.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
    return [(fi, conf) for fi, conf in out if fi >= 0]


def _dense_focus_window(
    item: Dict[str, object],
    available_frames: set[int],
    *,
    width: int = 48,
    candidate_width: int = 64,
) -> set[int]:
    cur = int(item.get("frame_index", 0))
    points = {cur}
    for fi, _ in _topk_rows(item):
        points.add(fi)
    dense: set[int] = set()
    for p in points:
        for x in range(max(0, p - candidate_width), p + candidate_width + 1):
            dense.add(x)
    if available_frames:
        for af in available_frames:
            if abs(af - cur) <= width:
                dense.add(af)
            for fi, _ in _topk_rows(item):
                if abs(af - fi) <= width:
                    dense.add(af)
    return {x for x in dense if x >= 0}


def _confidence_shape_score(item: Dict[str, object], cand: int) -> float:
    base = float(item.get("confidence", 0.0))
    topk = _topk_rows(item)
    if not topk:
        return base
    best = base
    for fi, conf in topk:
        dist = abs(cand - fi)
        decay = max(0.0, 1.0 - (dist / 80.0))
        best = max(best, conf * decay)
    return best


def _pick_best_focus_frame(item: Dict[str, object], available_frames: set[int]) -> tuple[int, float]:
    current_idx = int(item.get("frame_index", 0))
    candidates = _build_focus_candidates(item, current_idx) | _dense_focus_window(item, available_frames)
    best_idx = current_idx
    best_score = float("-inf")
    best_conf = float(item.get("confidence", 0.0))

    for cand in candidates:
        cand_conf = max(_candidate_confidence(item, cand), _confidence_shape_score(item, cand))
        nearest = _nearest_frame(cand, available_frames, max_dist=18)
        align_bonus = 0.08 if cand in available_frames else 0.0
        if nearest is None:
            align_penalty = 0.03
        else:
            align_penalty = min(0.025, abs(nearest - cand) * 0.0016)
        move_penalty = min(0.02, abs(cand - current_idx) * 0.0008)
        score = cand_conf + align_bonus - align_penalty - move_penalty

        if score > best_score:
            best_score = score
            best_idx = cand
            best_conf = cand_conf

    if best_idx not in available_frames:
        snapped = _nearest_frame(best_idx, available_frames, max_dist=12)
        if snapped is not None:
            best_idx = snapped
            best_conf = max(best_conf, _candidate_confidence(item, best_idx))

    adjusted_conf = round(max(0.0, min(0.99, best_conf)), 4)
    return best_idx, adjusted_conf


def _enforce_event_spacing(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = [dict(x) for x in rows]
    by_name = {str(x.get("event_name") or ""): x for x in out}
    min_gap = {
        ("Top", "Mid-downswing"): 4,
        ("Mid-downswing", "Impact"): 4,
        ("Impact", "Mid-follow-through"): 4,
        ("Mid-follow-through", "Finish"): 5,
    }
    for i in range(1, len(EVENT_SEQUENCE)):
        left_name = EVENT_SEQUENCE[i - 1]
        right_name = EVENT_SEQUENCE[i]
        l = by_name.get(left_name)
        r = by_name.get(right_name)
        if not l or not r:
            continue
        li = int(l.get("frame_index", 0))
        ri = int(r.get("frame_index", 0))
        gap = int(min_gap.get((left_name, right_name), 1))
        if ri <= li:
            ri = li + gap
        elif ri - li < gap:
            ri = li + gap
        r["frame_index"] = ri
    return out


def _item_score(item: Dict[str, object], cand: int, available_frames: set[int]) -> float:
    conf = max(_candidate_confidence(item, cand), _confidence_shape_score(item, cand))
    align = 0.08 if cand in available_frames else 0.0
    return float(conf) + align


def _event_item(rows: List[Dict[str, object]], event_name: str) -> Dict[str, object] | None:
    for r in rows:
        if str(r.get("event_name") or "") == event_name:
            return r
    return None


def _refine_finish_after_impact(rows: List[Dict[str, object]], available_frames: set[int]) -> List[Dict[str, object]]:
    """Rerank Finish vs Impact using top-k + dense window (semantic lateness, not just snap)."""
    out = [dict(x) for x in rows]
    fin = _event_item(out, "Finish")
    impact = _event_item(out, "Impact")
    if not fin or not impact:
        return out
    imp_i = int(impact.get("frame_index", 0))
    min_fin = imp_i + 6
    fin_c = sorted(_dense_focus_window(fin, available_frames, width=72, candidate_width=88))
    fin_c = [f for f in fin_c if f >= min_fin]
    if not fin_c:
        return out

    best_f: int | None = None
    best_score = float("-inf")
    for f in fin_c:
        s = _item_score(fin, f, available_frames)
        lateness = f - imp_i
        s += min(0.07, lateness * 0.0016)
        s += max(0.0, 0.03 - abs(f - (imp_i + 72)) * 0.00045)
        if s > best_score:
            best_score = s
            best_f = f
    if best_f is None:
        return out
    fin["frame_index"] = int(best_f)
    fin["confidence"] = max(
        float(fin.get("confidence", 0.0)),
        round(_confidence_shape_score(fin, best_f), 4),
    )
    return out


def _refine_core_triplet(
    rows: List[Dict[str, object]],
    available_frames: set[int],
    *,
    wide: bool = False,
) -> List[Dict[str, object]]:
    out = [dict(x) for x in rows]
    top = _event_item(out, "Top")
    mid = _event_item(out, "Mid-downswing")
    impact = _event_item(out, "Impact")
    if not top or not mid or not impact:
        return out

    tw, tcw = (72, 88) if wide else (60, 72)
    mw, mcw = (68, 84) if wide else (56, 68)
    iw, icw = (72, 88) if wide else (60, 72)
    top_c = sorted(_dense_focus_window(top, available_frames, width=tw, candidate_width=tcw))
    mid_c = sorted(_dense_focus_window(mid, available_frames, width=mw, candidate_width=mcw))
    imp_c = sorted(_dense_focus_window(impact, available_frames, width=iw, candidate_width=icw))
    if not top_c or not mid_c or not imp_c:
        return out

    def _semantic_bonus(t: int, m: int, i: int) -> float:
        # Encourage a real core-event structure instead of uniform spreading.
        top_peak = max((fi for fi, _ in _topk_rows(top)), default=t)
        impact_peak = max((fi for fi, _ in _topk_rows(impact)), default=i)
        b = 0.0
        b += max(0.0, 0.06 - abs(t - top_peak) * 0.0012)
        b += max(0.0, 0.06 - abs(i - impact_peak) * 0.0012)
        # Mid-downswing must sit between Top and Impact, biased toward the first half of downswing.
        ideal_mid = t + max(3, int(round((i - t) * 0.45)))
        b += max(0.0, 0.05 - abs(m - ideal_mid) * 0.0015)
        # Impact should be later enough than Top to avoid pseudo-impact frames.
        if i - t >= 10:
            b += 0.025
        return b

    best = None
    best_score = float("-inf")
    for t in top_c:
        s_t = _item_score(top, t, available_frames)
        for m in mid_c:
            if m - t < 4:
                continue
            s_m_base = _item_score(mid, m, available_frames)
            for i in imp_c:
                if i - m < 4:
                    continue
                if i - t < 9:
                    continue
                ideal_mid = t + max(4, int(round((i - t) * 0.42)))
                s_m = s_m_base + max(0.0, 0.035 - abs(m - ideal_mid) * 0.0018)
                s_i = _item_score(impact, i, available_frames)
                top_impact_cands = _topk_rows(impact)
                if top_impact_cands:
                    best_fi, best_cf = max(top_impact_cands, key=lambda x: x[1])
                    if abs(i - int(best_fi)) <= 10:
                        s_i += min(0.04, float(best_cf) * 0.05)
                gap_bonus = min(0.06, (i - t) * 0.0018)
                score = s_t + s_m + s_i + gap_bonus + _semantic_bonus(t, m, i)
                if score > best_score:
                    best_score = score
                    best = (t, m, i)
    if best is None:
        return out

    top["frame_index"], mid["frame_index"], impact["frame_index"] = best
    top["confidence"] = max(float(top.get("confidence", 0.0)), round(_confidence_shape_score(top, best[0]), 4))
    mid["confidence"] = max(float(mid.get("confidence", 0.0)), round(_confidence_shape_score(mid, best[1]), 4))
    impact["confidence"] = max(
        float(impact.get("confidence", 0.0)),
        round(_confidence_shape_score(impact, best[2]), 4),
    )
    return out


def refine_with_b_layer(
    keyframes: List[Dict[str, object]],
    enhanced_local_frames: List[dict],
    *,
    analysis_id: Optional[str] = None,
    analysis_video: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, object]] = None,
    analysis_frames: Optional[List[dict]] = None,
    confidence: Optional[Dict[str, float]] = None,
    fail_reasons: Optional[List[str]] = None,
    recovery_pass: bool = False,
) -> List[Dict[str, object]]:
    """B-path refinement with event-focused local re-targeting.

    Uses SwingNet cache refinement as first pass, then re-anchors high-risk events
    with top-k candidates + local windows aligned to available analysis/enhanced frames.
    """
    _ = (analysis_video, preprocess_meta)
    sw_w = 40 if recovery_pass else 12
    refined_in = (
        swingnet_b_refine(analysis_id or "", list(keyframes), window=sw_w)
        if analysis_id
        else list(keyframes)
    )

    available_frames = _frame_index_set(enhanced_local_frames)
    if analysis_frames:
        available_frames |= _frame_index_set(analysis_frames)

    focus_events = _event_focus_set(fail_reasons or [], confidence or {})
    if recovery_pass:
        focus_events |= FOCUS_EVENTS_DEFAULT | {"Mid-backswing", "Mid-follow-through"}
    out: List[Dict[str, object]] = []

    for item in refined_in:
        event_name = str(item.get("event_name") or "")
        current_idx = int(item.get("frame_index", 0))
        current_conf = float(item.get("confidence", 0.0))

        cloned = dict(item)
        if event_name in focus_events:
            new_idx, new_conf = _pick_best_focus_frame(cloned, available_frames)
            cloned["frame_index"] = new_idx
            cloned["confidence"] = max(0.0, min(0.99, new_conf))
        else:
            # For non-focus events, only very light alignment to nearest available frame.
            nearest = _nearest_frame(current_idx, available_frames, max_dist=2)
            if nearest is not None:
                cloned["frame_index"] = nearest
            cloned["confidence"] = round(max(0.0, min(0.99, current_conf)), 4)

        out.append(cloned)

    out = _refine_core_triplet(out, available_frames, wide=recovery_pass)
    out = _refine_finish_after_impact(out, available_frames)
    return _enforce_event_spacing(out)
