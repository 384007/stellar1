"""Lite-only: sample a few decoded BGR frames for backend club vision."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def lite_sample_preview_bgr(video_path: str, fractions: tuple[float, ...] = (0.25, 0.4, 0.6)) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out: list[np.ndarray] = []
    try:
        if total <= 0:
            return []
        for frac in fractions:
            fi = int(np.clip(round((total - 1) * frac), 0, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, fr = cap.read()
            if ok and fr is not None:
                out.append(fr)
    finally:
        cap.release()
    return out
