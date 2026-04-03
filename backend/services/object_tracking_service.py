"""Object tracking facade returning stable track schema."""

from __future__ import annotations

import os
from typing import Any

from services.provider_registry import role_log


def build_tracks_from_detections(detections_bundle: dict[str, Any] | None) -> dict[str, Any]:
    detections_bundle = detections_bundle or {}
    mode = (os.getenv("STELLAR_TRACKING_BACKEND") or "bytetrack").strip().lower()
    if mode == "disabled" or not bool(detections_bundle.get("enabled")):
        role_log("[ROLE=BYTETRACK] status=disabled")
        return {
            "person_tracks": [],
            "club_tracks": [],
            "ball_tracks": [],
            "track_confidence": 0.0,
            "occlusion_flags": {},
            "provider": "disabled",
            "status": "disabled",
        }
    dets = list(detections_bundle.get("detections") or [])
    from services.providers.tracking_bytetrack_provider import run

    r = run(dets)
    payload = dict(r.get("payload") or {})
    return {
        "person_tracks": list(payload.get("person_tracks") or []),
        "club_tracks": list(payload.get("club_tracks") or []),
        "ball_tracks": list(payload.get("ball_tracks") or []),
        "track_confidence": float(payload.get("track_confidence") or 0.0),
        "occlusion_flags": dict(payload.get("occlusion_flags") or {}),
        "provider": r.get("provider_name"),
        "status": r.get("status"),
        "error_reason": r.get("error_reason"),
        "provider_meta": r,
    }
