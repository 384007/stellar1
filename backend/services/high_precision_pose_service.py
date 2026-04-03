"""High precision pose adapter (optional runtime path)."""

from __future__ import annotations

from typing import Any


def extract_high_precision_pose_stream(video_path: str, max_frames: int = 45) -> dict[str, Any]:
    """High-precision pose stream (legacy entrypoint).

    Prefer ``pose_backend_service.extract_pose_stream`` with
    ``STELLAR_POSE_BACKEND=rtmpose|mmpose`` and ``STELLAR_RTMPOSE_CONFIG`` /
    ``STELLAR_RTMPOSE_CHECKPOINT`` set; that path uses ``pose_rtmpose_provider``.
    """
    from services.pose_backend_service import extract_pose_stream

    return extract_pose_stream(video_path, max_frames=max_frames)
