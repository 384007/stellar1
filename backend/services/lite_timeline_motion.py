"""Lite-only: single <=400-frame timeline + motion on that chain only (no dense 2400 scan)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LITE_TIMELINE_MAX_FRAMES = 400


def lite_build_uniform_timeline(total_frames: int, fps: float) -> list[dict[str, Any]]:
    """Uniform decode indices + time_ms; at most LITE_TIMELINE_MAX_FRAMES samples."""
    if total_frames <= 0:
        return []
    n = min(LITE_TIMELINE_MAX_FRAMES, total_frames)
    fps = max(float(fps), 1e-6)
    raw = np.linspace(0, total_frames - 1, num=n, dtype=np.int64)
    out: list[dict[str, Any]] = []
    for fi in raw.tolist():
        fi = int(fi)
        out.append({"frame_index": fi, "time_ms": int(round(fi * 1000.0 / fps))})
    return out


def lite_motion_along_timeline(video_path: str, frame_indices: list[int]) -> list[float]:
    """Pairwise motion (mean abs diff on downscaled gray) at each timeline step; len == len(indices)."""
    if not frame_indices:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [0.0] * len(frame_indices)
    motions: list[float] = []
    prev: np.ndarray | None = None
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(fi)))
        ok, fr = cap.read()
        if not ok or fr is None:
            motions.append(0.0)
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (200, 112), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            motions.append(float(np.mean(np.abs(small - prev))))
        else:
            motions.append(0.0)
        prev = small
    cap.release()
    while len(motions) < len(frame_indices):
        motions.append(0.0)
    return motions[: len(frame_indices)]


def lite_impact_hint_from_timeline(
    frame_indices: list[int],
    motions: list[float],
    fps: float,
    duration_s: float,
) -> dict[str, Any]:
    """Coarse impact time from peak motion on the lite timeline only."""
    fps = max(float(fps), 1e-6)
    if not frame_indices or not motions:
        return {"impact_hint_s": 0.0, "window_s": [0.0, max(0.0, duration_s)]}
    m = np.array(motions[1:], dtype=np.float64) if len(motions) > 1 else np.array([0.0])
    k_off = int(np.argmax(m)) + 1 if m.size else 0
    k_off = max(1, min(k_off, len(frame_indices) - 1))
    fi = int(frame_indices[k_off])
    hint_s = fi / fps
    dur = max(0.0, float(duration_s))
    return {
        "impact_hint_s": round(hint_s, 4),
        "window_s": [round(max(0.0, hint_s - 0.9), 4), round(min(dur, hint_s + 0.9), 4)],
        "impact_timeline_index": k_off,
    }
