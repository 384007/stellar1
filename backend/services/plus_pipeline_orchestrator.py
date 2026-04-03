"""Shared Plus pipeline stages (detection, tracking, segmentation) to keep routers thin."""

from __future__ import annotations

from typing import Any

from services.detection_provider_service import get_video_detections
from services.object_tracking_service import build_tracks_from_detections
from services.phase_segment_service import segment_swing_phases


def load_detections_and_tracks(video_path: str, poses: list[dict]) -> tuple[dict[str, Any], dict[str, Any]]:
    det_bundle = get_video_detections(video_path, poses=poses)
    tracks = build_tracks_from_detections(det_bundle)
    return det_bundle, tracks


def load_phase_segment_bundle(
    poses: list[dict],
    *,
    tracks: dict[str, Any] | None = None,
    detections: list[dict] | None = None,
    motion_3d: list | None = None,
    video_path: str | None = None,
) -> dict[str, Any]:
    return segment_swing_phases(
        poses,
        tracks=tracks,
        detections=list(detections or []),
        motion_3d=motion_3d,
        video_path=video_path,
    )
