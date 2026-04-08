"""Candidate A: standard stable path for cleaner / steadier mobile swings."""

from __future__ import annotations

from typing import Any

from services.lite_keyframe_constants import LITE_EVENT_SEQUENCE


def lite_build_candidate_a_rows(
    frame_indices: list[int],
    motions: list[float],
) -> list[dict[str, Any]]:
    """
    Eight phase rows from the uniform timeline + motion peak as impact anchor.
    Stricter spacing assumptions than candidate B (main path, not recovery).
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
    base_conf = 0.72
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
