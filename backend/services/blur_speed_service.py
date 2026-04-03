"""
Motion-blur streak speed estimator.

Uses Hough line transform (cv2.HoughLinesP) to detect horizontal motion
streaks near the impact zone.  The longest streak length is converted to
an estimated ball speed in mph using a pixel-distance-time model.

Pure OpenCV — no torch / ultralytics dependencies.
"""

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Calibration: pixels-per-meter at a reference resolution (720p, ~4 m camera distance)
_REF_WIDTH = 1280
_PX_PER_METER_AT_REF = 180.0

# Average inter-frame gap for 30fps / 60fps (seconds)
_DEFAULT_FPS = 30.0

# Hough parameters
_HOUGH_RHO = 1
_HOUGH_THETA = np.pi / 180
_HOUGH_THRESHOLD = 25
_HOUGH_MIN_LINE_LENGTH = 10
_HOUGH_MAX_LINE_GAP = 8

# Only keep lines within ±20° of horizontal (ball travels roughly horizontally after impact)
_MAX_ANGLE_DEG = 20.0


def _longest_horizontal_streak(gray: np.ndarray) -> int:
    """Detect the longest near-horizontal line in a grayscale diff frame."""
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        edges,
        rho=_HOUGH_RHO,
        theta=_HOUGH_THETA,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=_HOUGH_MIN_LINE_LENGTH,
        maxLineGap=_HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        return 0

    max_len = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        if angle_deg > _MAX_ANGLE_DEG and angle_deg < (180 - _MAX_ANGLE_DEG):
            continue
        length = math.hypot(dx, dy)
        if length > max_len:
            max_len = length

    return int(round(max_len))


def _streak_to_mph(streak_px: int, frame_width: int, fps: float) -> float:
    """Convert pixel streak length to approximate ball speed in mph."""
    scale = (frame_width / _REF_WIDTH) if frame_width > 0 else 1.0
    px_per_meter = _PX_PER_METER_AT_REF * scale
    meters = streak_px / px_per_meter if px_per_meter > 0 else 0.0
    dt = 1.0 / max(fps, 1.0)
    speed_mps = meters / dt
    speed_mph = speed_mps * 2.23694
    return round(speed_mph, 1)


def detect_blur_speed(
    frames: list[np.ndarray],
    fps: float = _DEFAULT_FPS,
) -> dict:
    """
    Estimate ball speed from motion blur streaks in frames around impact.

    Scans consecutive frame pairs within the supplied window (typically
    ±3 frames around impact) and returns the speed implied by the longest
    detected streak.

    Args:
        frames: List of BGR frames (np.ndarray) around the impact moment.
        fps:    Video frame rate for time-distance conversion.

    Returns:
        {
            "ball_speed":       float  — mph,
            "streak_length_px": int    — longest streak in pixels,
            "confidence":       str    — "high" | "medium" | "low",
        }
    """
    if len(frames) < 2:
        return {"ball_speed": 0.0, "streak_length_px": 0, "confidence": "low"}

    best_streak = 0
    frame_width = frames[0].shape[1] if len(frames) > 0 else _REF_WIDTH

    for i in range(len(frames) - 1):
        gray_a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_a, gray_b)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

        streak = _longest_horizontal_streak(blurred)
        if streak > best_streak:
            best_streak = streak

    if best_streak > 30:
        confidence = "high"
    elif best_streak >= 15:
        confidence = "medium"
    else:
        confidence = "low"

    ball_speed = _streak_to_mph(best_streak, frame_width, fps)

    logger.info(
        "Blur speed: streak=%dpx, speed=%.1f mph, confidence=%s",
        best_streak, ball_speed, confidence,
    )

    return {
        "ball_speed": ball_speed,
        "streak_length_px": best_streak,
        "confidence": confidence,
    }
