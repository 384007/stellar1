"""Pro v2 — dense motion features inside swing window (OpenCV only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DenseFrame:
    frame_index: int
    timestamp_s: float
    motion: float
    motion_x: float
    motion_y: float
    motion_energy_smooth: float
    is_local_peak: bool
    is_local_valley: bool


def _moving_average_arr(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, kernel, mode="valid")


def dense_scan_swing_region(
    analysis_video_path: str,
    *,
    fps: float,
    t_start_s: float,
    t_end_s: float,
    max_frames: int = 2400,
) -> list[DenseFrame]:
    """Decode swing window only; fill motion, directional proxies, smooth energy, local peak/valley flags."""
    path = str(Path(analysis_video_path))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps_v = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    lo = max(0, int(t_start_s * fps_v))
    hi = min(total - 1, int(np.ceil(t_end_s * fps_v)))
    span = hi - lo + 1
    if span < 3:
        cap.release()
        return []

    stride = max(1, int(np.ceil(span / max_frames)))

    raw_motion: list[float] = []
    raw_mx: list[float] = []
    raw_my: list[float] = []
    indices: list[int] = []
    prev_small: np.ndarray | None = None

    idx = lo
    while idx <= hi:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (200, 112), interpolation=cv2.INTER_AREA).astype(np.float64)
        if prev_small is not None:
            d = np.abs(small - prev_small)
            motion = float(np.mean(d))
            gx = cv2.Sobel(d, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(d, cv2.CV_64F, 0, 1, ksize=3)
            mx = float(np.mean(np.abs(gx)))
            my = float(np.mean(np.abs(gy)))
            raw_motion.append(motion)
            raw_mx.append(mx)
            raw_my.append(my)
            indices.append(idx)
        prev_small = small
        idx += stride

    cap.release()

    n = len(indices)
    if n == 0:
        return []

    M = np.array(raw_motion, dtype=np.float64)
    MX = np.array(raw_mx, dtype=np.float64)
    MY = np.array(raw_my, dtype=np.float64)
    M[0] = 0.0
    MX[0] = 0.0
    MY[0] = 0.0

    k_s = min(5, max(3, n | 1))
    if k_s % 2 == 0:
        k_s = max(1, k_s - 1)
    smooth = _moving_average_arr(M, k_s)

    is_peak = np.zeros(n, dtype=bool)
    is_valley = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            is_peak[i] = True
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]:
            is_valley[i] = True

    out: list[DenseFrame] = []
    for j in range(n):
        ts = indices[j] / fps_v if fps_v > 0 else 0.0
        out.append(
            DenseFrame(
                frame_index=int(indices[j]),
                timestamp_s=round(float(ts), 5),
                motion=float(M[j]),
                motion_x=float(MX[j]),
                motion_y=float(MY[j]),
                motion_energy_smooth=float(smooth[j]),
                is_local_peak=bool(is_peak[j]),
                is_local_valley=bool(is_valley[j]),
            )
        )

    peak_count = int(np.sum(is_peak))
    valley_count = int(np.sum(is_valley))
    logger.info(
        "[PRO_V2][DENSE] dense_count=%s motion_peak_count=%s motion_valley_count=%s stride=%s span=%s",
        len(out),
        peak_count,
        valley_count,
        stride,
        span,
    )
    return out
