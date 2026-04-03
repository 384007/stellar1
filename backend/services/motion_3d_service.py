"""3D motion facade — MotionBERT JIT when available; else MediaPipe world / joint z (always on pose pipeline)."""

from __future__ import annotations

import logging
from typing import Any

from services.provider_registry import role_log

logger = logging.getLogger(__name__)


def _pack_motion3d_provider(r: dict) -> dict[str, Any]:
    payload = dict(r.get("payload") or {})
    enabled = bool(r.get("status") == "ok")
    return {
        "enabled": enabled,
        "motion_3d": list(payload.get("joints3d") or []) if enabled else [],
        "provider": r.get("provider_name"),
        "status": r.get("status"),
        "provider_meta": r,
        "lift_confidence": float(payload.get("lift_confidence") or 0.0) if enabled else 0.0,
    }


def lift_motion_3d(poses: list[dict]) -> dict[str, Any]:
    from services.providers.pose3d_motionbert_provider import run as motionbert_run
    from services.providers.pose3d_mediapipe_world_provider import run as mp_world_run

    mb = motionbert_run(poses)
    if mb.get("status") == "ok":
        out = _pack_motion3d_provider(mb)
        role_log(
            f"[ROLE=MOTION3D] provider={out['provider']} status={out['status']} "
            f"enabled={out['enabled']} frames={len(poses)}"
        )
        logger.info("[STELLAR_PLUS_PIPELINE] pose3d_primary=motionbert pose3d_active=motionbert lift_provider=motionbert")
        return out

    mp = mp_world_run(poses)
    out = _pack_motion3d_provider(mp)
    if out["enabled"]:
        meta = dict(out["provider_meta"]) if isinstance(out["provider_meta"], dict) else {}
        meta["motionbert_skipped_status"] = mb.get("status")
        meta["motionbert_skipped_reason"] = mb.get("error_reason")
        out["provider_meta"] = meta
        role_log(
            f"[ROLE=MOTION3D] provider={out['provider']} status={out['status']} "
            f"enabled={out['enabled']} frames={len(poses)} (MotionBERT skipped={mb.get('status')})"
        )
        logger.info(
            "[STELLAR_PLUS_PIPELINE] pose3d_primary=motionbert_skipped pose3d_active=%s lift_provider=%s motionbert_status=%s",
            out["provider"],
            out["provider"],
            mb.get("status"),
        )
        return out

    meta = dict(out["provider_meta"]) if isinstance(out["provider_meta"], dict) else {}
    meta["motionbert_attempt"] = {"status": mb.get("status"), "error_reason": mb.get("error_reason")}
    out["provider_meta"] = meta
    role_log(
        f"[ROLE=MOTION3D] provider={out['provider']} status={out['status']} "
        f"enabled={out['enabled']} frames={len(poses)}"
    )
    logger.info(
        "[STELLAR_PLUS_PIPELINE] pose3d_disabled motionbert_status=%s mediapipe_world_status=%s",
        mb.get("status"),
        mp.get("status"),
    )
    return out
