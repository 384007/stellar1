"""
Ball trajectory tracker via frame differencing.

Uses inter-frame subtraction → Gaussian blur → contour detection to locate the
ball across consecutive frames.  Centroid displacement between frames is
converted to ball speed in mph.

Pure OpenCV — no torch / ultralytics dependencies.
"""

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Calibration: reference resolution and estimated pixels-per-meter
_REF_WIDTH = 1280
_PX_PER_METER_AT_REF = 180.0

# Contour area limits (in pixels²) to filter ball-sized blobs
_MIN_CONTOUR_AREA = 20
_MAX_CONTOUR_AREA = 3000

# Ball is roughly circular: reject contours with extreme aspect ratios
_MAX_ASPECT_RATIO = 3.0


def _find_ball_centroid(diff_gray: np.ndarray) -> tuple[float, float] | None:
    """Find the centroid of the most ball-like blob in a difference frame."""
    _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_circularity = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_CONTOUR_AREA or area > _MAX_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > _MAX_ASPECT_RATIO:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        if circularity > best_circularity:
            best_circularity = circularity
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                best = (M["m10"] / M["m00"], M["m01"] / M["m00"])

    return best


def track_trajectory(
    frames: list[np.ndarray],
    fps: float = 30.0,
) -> dict:
    """
    Track the golf ball across consecutive frames and estimate speed.

    Args:
        frames: List of BGR frames (np.ndarray) covering the post-impact window.
        fps:    Video frame rate.

    Returns:
        {
            "ball_speed":      float — mph,
            "tracked_frames":  int   — number of frames with a successful detection,
            "confidence":      str   — "high" | "medium" | "low",
        }
    """
    if len(frames) < 2:
        return {"ball_speed": 0.0, "tracked_frames": 0, "confidence": "low"}

    frame_width = frames[0].shape[1]
    scale = (frame_width / _REF_WIDTH) if frame_width > 0 else 1.0
    px_per_meter = _PX_PER_METER_AT_REF * scale
    dt = 1.0 / max(fps, 1.0)

    centroids: list[tuple[float, float]] = []

    for i in range(len(frames) - 1):
        gray_a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_a, gray_b)
        pt = _find_ball_centroid(diff)
        if pt is not None:
            centroids.append(pt)

    tracked_frames = len(centroids)

    if tracked_frames < 2:
        return {"ball_speed": 0.0, "tracked_frames": tracked_frames, "confidence": "low"}

    displacements_px: list[float] = []
    for i in range(len(centroids) - 1):
        dx = centroids[i + 1][0] - centroids[i][0]
        dy = centroids[i + 1][1] - centroids[i][1]
        displacements_px.append(math.hypot(dx, dy))

    # Use the maximum displacement (closest to the actual impact moment)
    max_disp_px = max(displacements_px) if displacements_px else 0.0
    meters = max_disp_px / px_per_meter if px_per_meter > 0 else 0.0
    speed_mps = meters / dt
    speed_mph = round(speed_mps * 2.23694, 1)

    if tracked_frames >= 4:
        confidence = "high"
    elif tracked_frames >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    logger.info(
        "Trajectory: tracked=%d frames, max_disp=%.1fpx, speed=%.1f mph, confidence=%s",
        tracked_frames, max_disp_px, speed_mph, confidence,
    )

    return {
        "ball_speed": speed_mph,
        "tracked_frames": tracked_frames,
        "confidence": confidence,
    }
