"""Choose final Lite keyframe rows from independent A/B gates (deterministic, no randomness)."""

from __future__ import annotations

import statistics
from typing import Any

from services.lite_keyframe_constants import LITE_EVENT_SEQUENCE

_LOG_PREFIX = "[lite_ab]"


def _avg_confidence(rows: list[dict[str, Any]]) -> float:
    vals = [float(r.get("confidence", 0.0)) for r in rows]
    return float(statistics.mean(vals)) if vals else 0.0


def _impact_fi(rows: list[dict[str, Any]]) -> int:
    for r in rows:
        if str(r.get("event_name")) == "Impact":
            return int(r.get("frame_index", 0))
    return 0


def _min_adjacent_gap(rows: list[dict[str, Any]]) -> int:
    by_ev = {str(r.get("event_name")): int(r.get("frame_index", 0)) for r in rows}
    ordered = [by_ev.get(n, -1) for n in LITE_EVENT_SEQUENCE]
    if any(x < 0 for x in ordered):
        return 0
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    return min(gaps) if gaps else 0


def lite_ab_quality_score(
    rows: list[dict[str, Any]],
    *,
    impact_hint_frame_index: int,
    fail_reasons: list[str],
) -> float:
    """
    Higher is better. Simple weighted sum (maintainable, not a black-box ML score).
    """
    hint_pen = abs(_impact_fi(rows) - int(impact_hint_frame_index))
    fr = float(len(fail_reasons))
    gap = float(_min_adjacent_gap(rows))
    ac = _avg_confidence(rows)
    return ac * 55.0 + gap * 2.4 - hint_pen * 0.22 - fr * 9.0


def select_lite_ab_final_rows(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    status_a: str,
    reasons_a: list[str],
    status_b: str,
    reasons_b: list[str],
    impact_hint_frame_index: int,
    logger: Any | None = None,
) -> tuple[list[dict[str, Any]], bool, str]:
    """
    Returns (final_rows, phase_validation_passed, internal_path_label_for_logs_only).

    Rules:
    - Only A passes -> A
    - Only B passes -> B
    - Both pass -> higher quality score (tie -> A)
    - Neither passes -> higher score, phase_validation_passed False
    """
    a_pass = status_a == "pass"
    b_pass = status_b == "pass"
    qa = lite_ab_quality_score(rows_a, impact_hint_frame_index=impact_hint_frame_index, fail_reasons=reasons_a)
    qb = lite_ab_quality_score(rows_b, impact_hint_frame_index=impact_hint_frame_index, fail_reasons=reasons_b)

    path = "unknown"
    final: list[dict[str, Any]]
    phase_ok: bool

    if a_pass and not b_pass:
        final, phase_ok, path = rows_a, True, "a"
    elif b_pass and not a_pass:
        final, phase_ok, path = rows_b, True, "b"
    elif a_pass and b_pass:
        if qa >= qb:
            final, phase_ok, path = rows_a, True, "a"
        else:
            final, phase_ok, path = rows_b, True, "b"
    else:
        phase_ok = False
        if qa >= qb:
            final, path = rows_a, "a_degraded"
        else:
            final, path = rows_b, "b_degraded"

    if logger is not None:
        logger.info(
            "%s final selected path=%s phase_validation_passed=%s qa=%.3f qb=%.3f a_pass=%s b_pass=%s",
            _LOG_PREFIX,
            path,
            phase_ok,
            qa,
            qb,
            a_pass,
            b_pass,
        )

    return final, phase_ok, path
