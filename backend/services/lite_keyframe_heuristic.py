"""Lite shared helpers: impact snap + monotonic enforcement (candidates live in ``lite_keyframe_candidate_*``)."""

from __future__ import annotations

from typing import Any

from services.lite_keyframe_constants import LITE_EVENT_SEQUENCE


def lite_refine_impact_row(rows: list[dict[str, Any]], hint_fi: int) -> list[dict[str, Any]]:
    """Snap Impact to decode index nearest hint_fi while staying after Top (gate-friendly)."""
    if len(rows) != 8:
        return rows
    top_fi = 0
    all_fi: list[int] = []
    for r in rows:
        fi = int(r.get("frame_index", 0))
        all_fi.append(fi)
        if str(r.get("event_name")) == "Top":
            top_fi = fi
    floor_i = top_fi + 4
    candidates = [x for x in all_fi if x >= floor_i] or all_fi
    best = min(candidates, key=lambda x: abs(x - hint_fi))
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        if str(d.get("event_name")) == "Impact":
            d["frame_index"] = best
            d["confidence"] = float(d.get("confidence", 0.72))
            d["top_k_candidates"] = [{"frame_index": best, "confidence": d["confidence"]}]
        out.append(d)
    order = {name: i for i, name in enumerate(LITE_EVENT_SEQUENCE)}
    out.sort(key=lambda x: order.get(str(x.get("event_name")), 99))
    return out


def lite_enforce_monotonic_frame_indices(rows: list[dict[str, Any]], max_fi: int) -> list[dict[str, Any]]:
    """Ensure Address→Finish decode indices are strictly increasing (canonical phase order)."""
    by_name = {str(r.get("event_name")): dict(r) for r in rows}
    prev = -1
    out: list[dict[str, Any]] = []
    for name in LITE_EVENT_SEQUENCE:
        r = by_name.get(name)
        if not r:
            continue
        fi = int(r.get("frame_index", 0))
        fi = min(max(max_fi, 0), max(prev + 1, fi))
        conf = float(r.get("confidence", 0.72))
        r["frame_index"] = fi
        r["top_k_candidates"] = [{"frame_index": fi, "confidence": conf}]
        out.append(r)
        prev = fi
    return out
