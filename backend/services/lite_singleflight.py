"""
Process-local single-flight + idempotency for POST /analyze/lite only.

One concurrent lite analyze per worker; duplicate keys while running -> rejected;
completed responses cached in-memory with TTL (failed runs are not cached, retry allowed).
"""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# ~12 minutes (spec: 10–15)
_CACHE_TTL_S = 12 * 60

_lock = asyncio.Lock()
_global_busy: bool = False
_active_request_id: Optional[str] = None
_completed: dict[str, tuple[float, dict[str, Any]]] = {}


def _purge_expired_unlocked() -> None:
    now = time.time()
    expired = [k for k, (ts, _) in _completed.items() if now - ts > _CACHE_TTL_S]
    for k in expired:
        del _completed[k]


async def begin_lite_analyze(request_id: str) -> tuple[Literal["cached", "busy", "run"], Optional[dict[str, Any]]]:
    async with _lock:
        _purge_expired_unlocked()
        if request_id in _completed:
            ts, body = _completed[request_id]
            if time.time() - ts <= _CACHE_TTL_S:
                logger.info("[lite_singleflight] cached_hit request_id=%s", request_id)
                return ("cached", copy.deepcopy(body))
            del _completed[request_id]

        global _global_busy, _active_request_id
        if _global_busy:
            logger.warning(
                "[lite_singleflight] conflict_409 busy=1 incoming_request_id=%s active_request_id=%s",
                request_id,
                _active_request_id,
            )
            return ("busy", None)

        _global_busy = True
        _active_request_id = request_id
        logger.info("[lite_singleflight] acquired request_id=%s", request_id)
        return ("run", None)


async def complete_lite_analyze_success(request_id: str, public_result: dict[str, Any]) -> None:
    async with _lock:
        global _global_busy, _active_request_id
        _global_busy = False
        _active_request_id = None
        _completed[request_id] = (time.time(), copy.deepcopy(public_result))
        logger.info("[lite_singleflight] completed request_id=%s", request_id)


async def complete_lite_analyze_failure(request_id: str) -> None:
    async with _lock:
        global _global_busy, _active_request_id
        _global_busy = False
        _active_request_id = None
        logger.warning("[lite_singleflight] failed request_id=%s (not cached; retry allowed)", request_id)
