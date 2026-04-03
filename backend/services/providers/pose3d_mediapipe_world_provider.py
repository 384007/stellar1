"""3D joint trajectories from MediaPipe world landmarks (or image-space x,y,z) — no external checkpoint."""

from __future__ import annotations

from services.provider_registry import role_log
from services.provider_schema import provider_result


def run(poses: list[dict]) -> dict:
    if not poses:
        role_log("[ROLE=POSE3D_MP_WORLD] status=no_poses")
        return provider_result(
            role="pose3d",
            provider_name="mediapipe_world",
            status="no_poses",
            frame_count=0,
            payload={},
            error_reason="no_poses",
        )

    joints3d: list[list[list[float]]] = []
    for p in poses:
        wl = p.get("world_landmarks")
        jlist = p.get("joints") or []
        frame_row: list[list[float]] = []
        for j in jlist:
            name = j.get("name")
            if isinstance(wl, dict) and name and isinstance(wl.get(name), dict):
                w = wl[name]
                frame_row.append(
                    [
                        float(w.get("x", 0.0)),
                        float(w.get("y", 0.0)),
                        float(w.get("z", 0.0)),
                    ]
                )
            else:
                frame_row.append(
                    [
                        float(j.get("x", 0.0)),
                        float(j.get("y", 0.0)),
                        float(j.get("z", 0.0)),
                    ]
                )
        joints3d.append(frame_row)

    if not any(frame_row for frame_row in joints3d):
        role_log("[ROLE=POSE3D_MP_WORLD] status=insufficient_joints")
        return provider_result(
            role="pose3d",
            provider_name="mediapipe_world",
            status="insufficient_joints",
            frame_count=len(poses),
            payload={},
            error_reason="insufficient_joints",
        )

    j0 = next((len(fr) for fr in joints3d if fr), 0)
    role_log(
        f"[ROLE=POSE3D_MP_WORLD] status=ok frames={len(poses)} joints_per_frame={j0} "
        f"world_lm={sum(1 for p in poses if isinstance(p.get('world_landmarks'), dict) and p.get('world_landmarks'))}"
    )
    return provider_result(
        role="pose3d",
        provider_name="mediapipe_world",
        status="ok",
        frame_count=len(poses),
        payload={"joints3d": joints3d, "lift_confidence": 0.55},
    )
