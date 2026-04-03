from __future__ import annotations

from typing import Any


def provider_result(
    *,
    role: str,
    provider_name: str,
    provider_version: str = "unknown",
    backend_profile: str = "default",
    status: str = "ok",
    frame_count: int = 0,
    timestamps: list[float] | None = None,
    frame_indices: list[int] | None = None,
    confidence_summary: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    error_reason: str | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "role": role,
        "provider_name": provider_name,
        "provider_version": provider_version,
        "backend_profile": backend_profile,
        "status": status,
        "frame_count": int(frame_count),
        "timestamps": list(timestamps or []),
        "frame_indices": list(frame_indices or []),
        "confidence_summary": dict(confidence_summary or {}),
        "payload": dict(payload or {}),
        "error_reason": error_reason,
        "fallback_used": bool(fallback_used),
    }
