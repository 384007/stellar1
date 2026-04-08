"""Candidate B: recovery path for noisy mobile video (jitter, uneven spacing, weak club cues)."""

from __future__ import annotations

from typing import Any

from services.lite_keyframe_constants import LITE_EVENT_SEQUENCE


def _smooth_motion_1d(values: list[float]) -> list[float]:
    if len(values) <= 2:
        return list(values)
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - 1)
        hi = min(len(values), i + 2)
        out.append(sum(values[lo:hi]) / float(hi - lo))
    return out


def lite_build_candidate_b_rows(
    frame_indices: list[int],
    motions: list[float],
) -> list[dict[str, Any]]:
    """
    Uses temporally smoothed motion to stabilize impact anchor; wider tolerances for
    top–impact spacing and follow-through spread so uneven mobile timelines still yield
    a coherent phase order (not a copy of candidate A indices).
    """
    n = len(frame_indices)
    if n < 8:
        raise RuntimeError("lite_timeline_too_short")
    raw = motions[1:] if len(motions) > 1 else [0.0]
    m = _smooth_motion_1d(raw) if raw else [0.0]
    if m:
        impact_offset = 1 + int(max(range(len(m)), key=lambda i: m[i]))
    else:
        impact_offset = n // 2
    impact_k = max(4, min(n - 4, impact_offset))
    min_top_imp_sep = max(3, n // 7)
    top_slot = max(3, impact_k - min_top_imp_sep)
    # Pull mid-backswing / top earlier when timeline is short (mobile clips).
    early_cap = max(2, min(top_slot - 1, int(0.42 * n)))
    p = [
        0,
        max(1, int(0.10 * n)),
        max(2, int(0.18 * n)),
        max(3, min(early_cap, top_slot - 1)),
        max(4, top_slot),
        impact_k,
        min(n - 2, impact_k + max(2, n // 10)),
        n - 1,
    ]
    for j in range(1, 8):
        if p[j] <= p[j - 1]:
            p[j] = min(n - 1, p[j - 1] + 1)
    p[0] = 0
    p[7] = n - 1
    p[5] = max(p[4] + 1, min(p[5], n - 3))
    p[6] = max(p[5] + 1, min(p[6], n - 2))

    base_conf = 0.60
    rows: list[dict[str, Any]] = []
    for ev, pos in zip(LITE_EVENT_SEQUENCE, p):
        fi = int(frame_indices[pos])
        rows.append(
            {
                "event_name": ev,
                "frame_index": fi,
                "confidence": base_conf,
                "top_k_candidates": [{"frame_index": fi, "confidence": base_conf}],
            }
        )
    return rows
