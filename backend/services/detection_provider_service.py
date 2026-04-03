"""Detection provider facade — YOLO11 preferred; callers degrade without fabricating boxes."""

from __future__ import annotations

import logging
import os
from typing import Any

from services.provider_registry import role_log

logger = logging.getLogger(__name__)


def get_video_detections(video_path: str, poses: list[dict] | None = None) -> dict[str, Any]:
    mode = (os.getenv("STELLAR_DETECTION_BACKEND") or os.getenv("STELLAR_DETECTION_MODE") or "yolo11").strip().lower()
    if mode != "yolo11":
        role_log(f"[ROLE=YOLO11] status=unavailable requested={mode} reason=only_yolo11_supported")
        logger.warning(
            "[STELLAR_PLUS_PIPELINE] yolo11_branch=skipped detection_active=false status=unavailable reason=only_yolo11_supported "
            "continuing_degraded_pose_chain"
        )
        return {
            "detections": [],
            "enabled": False,
            "provider": "yolo11",
            "status": "unavailable",
            "error_reason": "only_yolo11_supported",
            "yolo11_degraded": True,
            "provider_meta": {"provider_name": "yolo11", "status": "unavailable", "error_reason": "only_yolo11_supported"},
        }

    from services.providers.detection_yolo11_provider import run

    r = run(video_path)
    payload = dict(r.get("payload") or {})
    detections = list(payload.get("detections") or [])
    enabled = bool(r.get("status") == "ok")
    if not enabled:
        logger.warning(
            "[STELLAR_PLUS_PIPELINE] yolo11_branch=degraded detection_active=false status=%s error_reason=%s "
            "continuing_degraded_pose_chain (no_fake_detections)",
            r.get("status"),
            r.get("error_reason"),
        )
    else:
        logger.info(
            "[STELLAR_PLUS_PIPELINE] yolo11_branch=ok detection_active=true status=ok detections=%d",
            len(detections),
        )
    return {
        "detections": detections,
        "enabled": enabled,
        "provider": r.get("provider_name"),
        "status": r.get("status"),
        "error_reason": r.get("error_reason"),
        "provider_meta": r,
        "yolo11_degraded": not enabled,
    }
