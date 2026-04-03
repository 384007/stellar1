from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImpactCandidate:
    frame_index: int
    timestamp: float
    score: float
    sharpness: float
    motion_energy: float
    image_base64: str


def _jpeg_b64(frame: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ''
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _center_roi(frame: np.ndarray, ratio: float = 0.45) -> np.ndarray:
    h, w = frame.shape[:2]
    rw = max(8, int(w * ratio))
    rh = max(8, int(h * ratio))
    x0 = max(0, (w - rw) // 2)
    y0 = max(0, (h - rh) // 2)
    return frame[y0:y0 + rh, x0:x0 + rw]


def _sharpness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _motion_energy(prev_frame: np.ndarray, frame: np.ndarray, next_frame: np.ndarray) -> float:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    d0 = cv2.absdiff(prev_gray, gray)
    d1 = cv2.absdiff(gray, next_gray)
    return float(np.mean(d0) + np.mean(d1))


def _impact_score(prev_frame: np.ndarray, frame: np.ndarray, next_frame: np.ndarray) -> tuple[float, float, float]:
    roi_prev = _center_roi(prev_frame)
    roi = _center_roi(frame)
    roi_next = _center_roi(next_frame)
    sharpness = _sharpness_score(roi)
    motion = _motion_energy(roi_prev, roi, roi_next)
    score = (motion * 0.70) + (sharpness * 0.30)
    return score, sharpness, motion


def select_best_impact_candidate(
    video_path: str | Path,
    *,
    around_time_s: float,
    window_s: float = 0.12,
    max_candidates: int = 15,
) -> dict[str, Any]:
    """Best-effort OpenCV sweet-spot refinement around a rough impact time.

    This does not decide the full phase chain. It only helps refine the impact frame by
    scoring local candidate frames around a rough impact timestamp.
    """
    path = str(Path(video_path))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    center_idx = max(0, min(total_frames - 1, int(round(float(around_time_s) * fps))))
    radius = max(2, int(round(window_s * fps)))
    lo = max(0, center_idx - radius)
    hi = min(max(total_frames - 1, 0), center_idx + radius)

    indices = np.linspace(lo, hi, num=min(max_candidates, max(hi - lo + 1, 1)), dtype=int)
    indices = np.unique(indices)

    frames: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((int(idx), frame))
    cap.release()

    if len(frames) < 3:
        return {
            'status': 'insufficient_frames',
            'center_frame_index': center_idx,
            'candidates': [],
            'best': None,
        }

    candidates: list[ImpactCandidate] = []
    for i in range(1, len(frames) - 1):
        prev_idx, prev_frame = frames[i - 1]
        idx, frame = frames[i]
        next_idx, next_frame = frames[i + 1]
        score, sharpness, motion = _impact_score(prev_frame, frame, next_frame)
        ts = float(idx) / fps if fps > 0 else 0.0
        candidates.append(
            ImpactCandidate(
                frame_index=idx,
                timestamp=round(ts, 4),
                score=round(score, 4),
                sharpness=round(sharpness, 4),
                motion_energy=round(motion, 4),
                image_base64=_jpeg_b64(frame),
            )
        )

    candidates.sort(key=lambda x: x.score, reverse=True)
    best = candidates[0] if candidates else None
    logger.info(
        '[ROLE=OPENCV_IMPACT] around=%.4f center_idx=%s best_idx=%s best_ts=%s best_score=%s',
        around_time_s,
        center_idx,
        None if best is None else best.frame_index,
        None if best is None else best.timestamp,
        None if best is None else best.score,
    )
    return {
        'status': 'ok',
        'center_frame_index': center_idx,
        'window_frame_range': [lo, hi],
        'candidates': [c.__dict__ for c in candidates],
        'best': None if best is None else best.__dict__,
    }
