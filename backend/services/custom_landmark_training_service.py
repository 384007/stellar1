"""Custom landmark training data/metadata management facade."""

from __future__ import annotations

import os
from typing import Any

from services.provider_registry import role_log


def export_training_manifest(items: list[dict]) -> dict[str, Any]:
    return {
        "count": len(items or []),
        "items": list(items or []),
        "ready": bool(items),
    }


def run_research_refine(frame_count: int, video_path: str | None = None) -> dict[str, Any]:
    mode = (os.getenv("STELLAR_RESEARCH_BACKEND") or "disabled").strip().lower()
    if mode != "deeplabcut":
        role_log(f"[ROLE=DEEPLABCUT] status=disabled frames={frame_count}")
        return {"enabled": False, "provider": "disabled", "status": "disabled", "refined_keypoints": []}
    from services.providers.research_deeplabcut_provider import run

    r = run(video_path or "", frame_count)
    payload = dict(r.get("payload") or {})
    enabled = bool(r.get("status") == "ok")
    return {
        "enabled": enabled,
        "provider": r.get("provider_name"),
        "status": r.get("status"),
        "refined_keypoints": list(payload.get("refined_keypoints") or []) if enabled else [],
        "provider_meta": r,
    }
