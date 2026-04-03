from __future__ import annotations

from services.provider_registry import role_log
from services.provider_schema import provider_result


def run(video_path: str, max_frames: int = 45) -> dict:
    from services.pose_service import extract_poses_from_video

    poses, quality = extract_poses_from_video(video_path, max_frames=max_frames)
    avg_visible = 0.0
    if poses:
        vis = []
        for p in poses:
            joints = p.get("joints", [])
            vis.append(sum(1 for j in joints if float(j.get("visibility", 0.0)) >= 0.3))
        avg_visible = float(sum(vis) / max(len(vis), 1))
    role_log(f"[ROLE=POSE_BACKEND] provider=mediapipe frames={len(poses)} poses={len(poses)} avg_visible_joints={avg_visible:.2f}")
    return provider_result(
        role="pose",
        provider_name="mediapipe",
        provider_version="baseline",
        backend_profile="baseline",
        status="ok",
        frame_count=len(poses),
        timestamps=[float(p.get("timestamp", 0.0)) for p in poses],
        frame_indices=[int(p.get("frame_index", i)) for i, p in enumerate(poses)],
        confidence_summary={"avg_visible_joints": avg_visible},
        payload={"poses": poses, "pose_quality_bundle": quality},
    )
