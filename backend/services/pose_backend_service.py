"""Unified pose backend service with stable schema for routers/services."""

from __future__ import annotations

import os
from typing import Any

from services.provider_registry import role_log
from services.providers import pose_mediapipe_provider

def _pose_mode() -> str:
    return (os.getenv("STELLAR_POSE_BACKEND") or os.getenv("STELLAR_POSE_MODE") or "mediapipe").strip().lower()


def _pack_pose_stream(
    base: dict[str, Any],
    *,
    backend_profile: str,
    requested: str,
    active: str,
) -> dict[str, Any]:
    poses = list((base.get("payload") or {}).get("poses") or [])
    quality = dict((base.get("payload") or {}).get("pose_quality_bundle") or {})
    return {
        "poses": poses,
        "pose_quality_bundle": quality,
        "landmarks_2d": [p.get("joints", []) for p in poses],
        "landmarks_3d": [p.get("joints3d", []) for p in poses],
        "visibility": [[j.get("visibility", 0.0) for j in p.get("joints", [])] for p in poses],
        "timestamps": [float(p.get("timestamp", 0.0)) for p in poses],
        "frame_indices": [int(p.get("frame_index", i)) for i, p in enumerate(poses)],
        "world_landmarks": [p.get("world_landmarks", []) for p in poses],
        "backend_profile": backend_profile,
        "provider_meta": {
            **base,
            "requested_backend": requested,
            "active_backend": active,
        },
    }


def extract_pose_stream(video_path: str, max_frames: int = 45) -> dict[str, Any]:
    mode = _pose_mode()
    if mode in {"high_precision", "rtmpose", "mmpose"}:
        from services.providers import pose_rtmpose_provider

        alt = pose_rtmpose_provider.run(video_path, max_frames=max_frames)
        if alt.get("status") == "ok":
            return _pack_pose_stream(alt, backend_profile="high_precision", requested=mode, active="high_precision")
        role_log(
            f"[ROLE=POSE_BACKEND] provider={mode} status={alt.get('status')} "
            f"reason={alt.get('error_reason')} using_mediapipe_fallback",
        )
    base = pose_mediapipe_provider.run(video_path, max_frames=max_frames)
    return _pack_pose_stream(base, backend_profile="mediapipe", requested=mode, active="mediapipe")
