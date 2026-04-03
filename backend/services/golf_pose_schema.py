"""Golf skeleton layout + shared 2D angle math (no MediaPipe — safe for optional backends)."""

from __future__ import annotations

import numpy as np

# MediaPipe pose landmark indices for the subset we keep in product poses.
GOLF_KEYPOINTS = {
    "head": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

GOLF_CONNECTIONS = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"],
    ["head", "left_shoulder"],
    ["head", "right_shoulder"],
]


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return round(float(angle), 1)


def compute_golf_angles(joints: list[dict]) -> dict:
    def get_joint(name: str) -> np.ndarray:
        for j in joints:
            if j["name"] == name:
                return np.array([j["x"], j["y"]])
        return np.array([0, 0])

    angles: dict = {}

    angles["left_elbow"] = calculate_angle(
        get_joint("left_shoulder"), get_joint("left_elbow"), get_joint("left_wrist")
    )
    angles["right_elbow"] = calculate_angle(
        get_joint("right_shoulder"), get_joint("right_elbow"), get_joint("right_wrist")
    )
    angles["left_knee"] = calculate_angle(
        get_joint("left_hip"), get_joint("left_knee"), get_joint("left_ankle")
    )
    angles["right_knee"] = calculate_angle(
        get_joint("right_hip"), get_joint("right_knee"), get_joint("right_ankle")
    )
    angles["left_shoulder"] = calculate_angle(
        get_joint("left_elbow"), get_joint("left_shoulder"), get_joint("left_hip")
    )
    angles["right_shoulder"] = calculate_angle(
        get_joint("right_elbow"), get_joint("right_shoulder"), get_joint("right_hip")
    )

    ls = get_joint("left_shoulder")
    rs = get_joint("right_shoulder")
    shoulder_dx = rs[0] - ls[0]
    shoulder_dy = rs[1] - ls[1]
    angles["shoulder_rotation"] = round(
        float(np.degrees(np.arctan2(shoulder_dy, shoulder_dx))), 1
    )

    lh = get_joint("left_hip")
    rh = get_joint("right_hip")
    hip_dx = rh[0] - lh[0]
    hip_dy = rh[1] - lh[1]
    angles["hip_rotation"] = round(
        float(np.degrees(np.arctan2(hip_dy, hip_dx))), 1
    )

    angles["x_factor"] = round(
        abs(angles["shoulder_rotation"] - angles["hip_rotation"]), 1
    )

    mid_hip = (lh + rh) / 2
    mid_shoulder = (ls + rs) / 2
    spine_vec = mid_shoulder - mid_hip
    angles["spine_tilt"] = round(
        float(np.degrees(np.arctan2(spine_vec[0], -spine_vec[1]))), 1
    )

    return angles
