from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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


def _pick_best_focus_frame(item: Dict[str, object], available_frames: set[int]) -> tuple[int, float]:
    current_idx = int(item.get("frame_index", 0))
    candidates = _build_focus_candidates(item, current_idx)
    best_idx = current_idx
    best_score = float("-inf")
    best_conf = float(item.get("confidence", 0.0))

    nearest_cache: dict[int, int | None] = {}
    for cand in candidates:
        cand_conf = _candidate_confidence(item, cand)
        nearest_cache[cand] = _nearest_frame(cand, available_frames, max_dist=10)

        align_bonus = 0.06 if cand in available_frames else 0.0
        if nearest_cache[cand] is None:
            align_penalty = 0.06
        else:
            align_penalty = min(0.05, abs(nearest_cache[cand] - cand) * 0.007)

        move_penalty = min(0.03, abs(cand - current_idx) * 0.002)
        score = cand_conf + align_bonus - align_penalty - move_penalty

        if score > best_score:
            best_score = score
            best_idx = cand
            best_conf = cand_conf

    snapped = _nearest_frame(best_idx, available_frames, max_dist=6)
    if snapped is not None:
        best_idx = snapped
        best_conf = max(best_conf, _candidate_confidence(item, best_idx))

    # Confidence tracks actual relocation quality; no blanket boost.
    adjusted_conf = round(max(0.0, min(0.99, best_conf)), 4)
    return best_idx, adjusted_conf


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
) -> List[Dict[str, object]]:
    """B-path refinement with event-focused local re-targeting.

    Uses SwingNet cache refinement as first pass, then re-anchors high-risk events
    with top-k candidates + local windows aligned to available analysis/enhanced frames.
    """
    _ = (analysis_video, preprocess_meta)
    refined_in = (
        swingnet_b_refine(analysis_id or "", list(keyframes))
        if analysis_id
        else list(keyframes)
    )

    available_frames = _frame_index_set(enhanced_local_frames)
    if analysis_frames:
        available_frames |= _frame_index_set(analysis_frames)

    focus_events = _event_focus_set(fail_reasons or [], confidence or {})
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

    return out
