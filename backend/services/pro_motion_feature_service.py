"""Pro Stage 2: minimal motion features from 240fps-aligned pose sequence (motion-first, not AI)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _joint_xy(pose: dict, name: str) -> tuple[float, float]:
    jt = pose.get("joints") or []
    for j in jt:
        if j.get("name") == name:
            n = j.get("normalized") or {}
            return float(n.get("x", 0.5)), float(n.get("y", 0.5))
    return 0.5, 0.5


def extract_motion_features(poses: list[dict]) -> dict[str, Any]:
    """Compute per-index scalar series used only for phase window cutting."""
    n = len(poses)
    if n < 4:
        return {
            "n": n,
            "wrist_mid_y": [],
            "wrist_mid_x": [],
            "wrist_vy": [],
            "hand_speed": [],
            "hand_path_progression": [],
            "shoulder_rotation": [],
            "hip_rotation": [],
            "x_factor_delta": [],
            "spine_tilt_y": [],
        }

    ly = np.zeros(n)
    ry = np.zeros(n)
    for i, p in enumerate(poses):
        lx, ly[i] = _joint_xy(p, "left_wrist")
        _, ry[i] = _joint_xy(p, "right_wrist")
    wrist_mid_y = (ly + ry) / 2.0
    wrist_vy = np.zeros(n)
    wrist_vy[1:] = np.diff(wrist_mid_y)

    hand_speed = np.zeros(n)
    for i in range(1, n):
        dlx = _joint_xy(poses[i], "left_wrist")[0] - _joint_xy(poses[i - 1], "left_wrist")[0]
        dly = ly[i] - ly[i - 1]
        drx = _joint_xy(poses[i], "right_wrist")[0] - _joint_xy(poses[i - 1], "right_wrist")[0]
        dry = ry[i] - ry[i - 1]
        hand_speed[i] = float(np.hypot(dlx + drx, dly + dry) * 0.5 + 1e-9)

    shoulder_rotation = np.array(
        [float((p.get("angles") or {}).get("shoulder_rotation", 0.0)) for p in poses],
        dtype=np.float64,
    )
    hip_rotation = np.array(
        [float((p.get("angles") or {}).get("hip_rotation", 0.0)) for p in poses],
        dtype=np.float64,
    )
    x_factor_delta = shoulder_rotation - hip_rotation
    spine_tilt_y = np.array(
        [float((p.get("angles") or {}).get("spine_tilt", 0.0)) for p in poses],
        dtype=np.float64,
    )

    lxs = np.zeros(n)
    rxs = np.zeros(n)
    for i, p in enumerate(poses):
        lxs[i], _ = _joint_xy(p, "left_wrist")
        rxs[i], _ = _joint_xy(p, "right_wrist")
    mid_x = (lxs + rxs) / 2.0
    hand_path_prog = np.zeros(n)
    for i in range(1, n):
        step = float(
            np.hypot(mid_x[i] - mid_x[i - 1], wrist_mid_y[i] - wrist_mid_y[i - 1]),
        )
        hand_path_prog[i] = hand_path_prog[i - 1] + step

    feats = {
        "n": n,
        "wrist_mid_y": wrist_mid_y.tolist(),
        "wrist_mid_x": mid_x.tolist(),
        "wrist_vy": wrist_vy.tolist(),
        "hand_speed": hand_speed.tolist(),
        "hand_path_progression": hand_path_prog.tolist(),
        "shoulder_rotation": shoulder_rotation.tolist(),
        "hip_rotation": hip_rotation.tolist(),
        "x_factor_delta": x_factor_delta.tolist(),
        "spine_tilt_y": spine_tilt_y.tolist(),
    }
    logger.info(
        "[STELLAR_PRO][MOTION_FEATURE] stage=done n=%s hand_speed_max=%.4f",
        n,
        float(np.max(hand_speed)),
    )
    return feats
