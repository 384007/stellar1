"""Lite-only: eight phase rows from the <=400-frame timeline + motion (no Pro A/B engines)."""

from __future__ import annotations

from typing import Any

from lib.prov3.keyframes.constants import EVENT_SEQUENCE


def lite_build_eight_keyframe_rows(
    frame_indices: list[int],
    motions: list[float],
) -> list[dict[str, Any]]:
    """
    Pick monotonic decode indices for EVENT_SEQUENCE using motion peak as impact anchor.
    All indices refer to the same lite timeline (decode indices in cleaned video).
    """
    n = len(frame_indices)
    if n < 8:
        raise RuntimeError("lite_timeline_too_short")
    m = motions[1:] if len(motions) > 1 else [0.0]
    if m:
        impact_offset = 1 + int(max(range(len(m)), key=lambda i: m[i]))
    else:
        impact_offset = n // 2
    impact_k = max(3, min(n - 3, impact_offset))
    impact_k = max(n // 4, min(3 * n // 4, impact_k))

    p = [
        0,
        max(1, n // 12),
        max(2, n // 6),
        max(3, min(int(impact_k * 0.52), impact_k - 8)),
        max(4, impact_k - max(2, n // 24)),
        impact_k,
        min(n - 2, impact_k + max(1, n // 11)),
        n - 1,
    ]
    for j in range(1, 8):
        if p[j] <= p[j - 1]:
            p[j] = min(n - 1, p[j - 1] + 1)
    p[0] = 0
    p[7] = n - 1

    rows: list[dict[str, Any]] = []
    for ev, pos in zip(EVENT_SEQUENCE, p):
        fi = int(frame_indices[pos])
        rows.append(
            {
                "event_name": ev,
                "frame_index": fi,
                "confidence": 0.72,
                "top_k_candidates": [{"frame_index": fi, "confidence": 0.72}],
            }
        )
    return rows


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
    order = {name: i for i, name in enumerate(EVENT_SEQUENCE)}
    out.sort(key=lambda x: order.get(str(x.get("event_name")), 99))
    return out


def lite_enforce_monotonic_frame_indices(rows: list[dict[str, Any]], max_fi: int) -> list[dict[str, Any]]:
    """Ensure Address→Finish decode indices are strictly increasing (EVENT_SEQUENCE order)."""
    by_name = {str(r.get("event_name")): dict(r) for r in rows}
    prev = -1
    out: list[dict[str, Any]] = []
    for name in EVENT_SEQUENCE:
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
