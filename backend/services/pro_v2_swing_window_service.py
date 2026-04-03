"""Pro v2 — swing window from combined motion score (sustained run, not single spike)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, kernel, mode="valid")


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo + 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def find_swing_window_seconds(
    analysis_video_path: str,
    *,
    fps: float,
    duration_s: float,
    sample_stride: int | None = None,
    pad_s: float = 0.18,
    min_swing_s: float = 0.55,
    screen_mode: bool = False,
) -> tuple[float, float]:
    """Return (t_start, t_end) using combined score: energy + motion_x/y proxies + setup→swing drop."""
    path = str(Path(analysis_video_path))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps_native = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = sample_stride or max(1, int(round(fps_native / 24.0)))

    energies: list[float] = []
    mx_list: list[float] = []
    my_list: list[float] = []
    frame_indices: list[int] = []
    prev_gray: np.ndarray | None = None
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float64)
        if prev_gray is not None:
            d = np.abs(small - prev_gray)
            motion_e = float(np.mean(d))
            gx = cv2.Sobel(d, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(d, cv2.CV_64F, 0, 1, ksize=3)
            mx = float(np.mean(np.abs(gx)))
            my = float(np.mean(np.abs(gy)))
            energies.append(motion_e)
            mx_list.append(mx)
            my_list.append(my)
            frame_indices.append(idx)
        prev_gray = small
        idx += stride

    cap.release()

    if len(energies) < 6:
        t1 = max(0.01, duration_s * 0.98)
        t0 = max(0.0, duration_s * 0.05)
        logger.info(
            "[PRO_V2][SWING_WINDOW] fallback_short_samples swing_t0=%.4f swing_t1=%.4f score_peak=n/a chosen_run=n/a",
            t0,
            t1,
        )
        return t0, t1

    E = np.array(energies, dtype=np.float64)
    MX = np.array(mx_list, dtype=np.float64)
    MY = np.array(my_list, dtype=np.float64)

    k_ma = min(7, len(E))
    if k_ma % 2 == 0:
        k_ma = max(1, k_ma - 1)
    E_sm = _moving_average(E, k_ma)

    q = max(3, len(E) // 4)
    early_mean = float(np.mean(E_sm[:q]))
    early_std = float(np.std(E_sm[:q]) + 1e-6)
    drop = np.maximum(0.0, E_sm - (early_mean + 0.9 * early_std))

    E_n = _norm01(E_sm)
    MX_n = _norm01(MX)
    MY_n = _norm01(MY)
    DR_n = _norm01(drop)

    combined = 0.38 * E_n + 0.22 * MX_n + 0.22 * MY_n + 0.18 * DR_n
    if screen_mode:
        # Suppress single-frame refresh spikes from screen recordings.
        med = float(np.median(combined))
        mad = float(np.median(np.abs(combined - med)) + 1e-6)
        z = (combined - med) / (1.4826 * mad)
        z = np.clip(z, -2.5, 3.0)
        sustain = _moving_average(np.maximum(z, 0.0), 5)
        combined = 0.56 * _norm01(combined) + 0.44 * _norm01(sustain)
    k_c = min(5, len(combined) | 1)
    if k_c % 2 == 0:
        k_c = max(1, k_c - 1)
    comb_sm = _moving_average(combined, k_c)

    p50 = float(np.percentile(comb_sm, 50))
    p78 = float(np.percentile(comb_sm, 78))
    thresh_alpha = 0.48 if screen_mode else 0.42
    thresh = p50 + thresh_alpha * max(p78 - p50, 1e-6)
    active = comb_sm >= thresh

    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and active[j]:
            j += 1
        min_run = 3 if screen_mode else 2
        if j - i >= min_run:
            runs.append((i, j - 1))
        i = j

    if not runs:
        t1 = max(0.01, duration_s * 0.98)
        t0 = max(0.0, duration_s * 0.05)
        logger.info(
            "[PRO_V2][SWING_WINDOW] fallback_no_run swing_t0=%.4f swing_t1=%.4f score_peak=n/a chosen_run=n/a",
            t0,
            t1,
        )
        return t0, t1

    def run_sustain_score(lo: int, hi: int) -> float:
        seg = comb_sm[lo : hi + 1]
        s = float(np.sum(seg))
        length = hi - lo + 1
        peak = float(np.max(seg))
        mean_s = float(np.mean(seg) + 1e-9)
        peakiness = peak / mean_s
        return s * np.sqrt(float(length)) / (peakiness**0.65 + 0.15)

    best_run = max(runs, key=lambda ab: run_sustain_score(ab[0], ab[1]))
    i0, i1 = best_run
    score_peak = float(np.max(comb_sm[i0 : i1 + 1]))

    fi0 = frame_indices[i0]
    fi1 = frame_indices[min(i1, len(frame_indices) - 1)]

    t0 = fi0 / fps_native
    t1 = (fi1 + stride) / fps_native
    t0 = max(0.0, t0 - pad_s)
    t1 = min(duration_s, t1 + pad_s)
    if t1 - t0 < min_swing_s:
        mid = (t0 + t1) * 0.5
        t0 = max(0.0, mid - min_swing_s * 0.5)
        t1 = min(duration_s, t0 + min_swing_s)

    logger.info(
        "[PRO_V2][SWING_WINDOW] score_peak=%.5f chosen_run=(%s,%s) swing_t0=%.4f swing_t1=%.4f dur=%.4f",
        score_peak,
        fi0,
        fi1,
        t0,
        t1,
        t1 - t0,
    )
    return t0, t1
