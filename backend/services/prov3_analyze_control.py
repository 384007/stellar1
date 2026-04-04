"""Cooperative cancel + in-flight counter for ``POST /pro-v3/analyze`` (per worker process)."""

from __future__ import annotations

import threading

PROV3_ANALYZE_CANCELLED = "prov3_analyze_cancelled"

_lock = threading.Lock()
_running = 0
_cancel = threading.Event()


def prov3_begin_analyze() -> None:
    """Clear cancel flag and increment running count (call when starting product analyze)."""
    global _running
    _cancel.clear()
    with _lock:
        _running += 1


def prov3_finish_analyze() -> None:
    """Decrement running count (always call in ``finally``)."""
    global _running
    with _lock:
        _running = max(0, _running - 1)


def prov3_analyze_in_flight_count() -> int:
    """How many ``/pro-v3/analyze`` runs are active on this process (includes Gemini enrich phase)."""
    with _lock:
        return _running


def prov3_request_cancel() -> None:
    """Signal the current run to stop at the next cancel check (thread + async enrich)."""
    _cancel.set()


def prov3_cancel_requested() -> bool:
    return _cancel.is_set()


def prov3_check_cancelled() -> None:
    if _cancel.is_set():
        raise RuntimeError(PROV3_ANALYZE_CANCELLED)
