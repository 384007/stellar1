"""Pro v2 screen-mode preprocess: detect inner video region and crop before main v2 chain."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _even(v: int) -> int:
    return max(2, int(v) // 2 * 2)


def _score_candidate(x: int, y: int, w: int, h: int, fw: int, fh: int, rectangularity: float) -> float:
    area_ratio = (w * h) / float(max(1, fw * fh))
    if area_ratio < 0.18 or area_ratio > 0.99:
        return -1.0
    ar = w / float(max(1, h))
    if ar < 0.9 or ar > 2.6:
        return -1.0
    cx = x + w / 2.0
    cy = y + h / 2.0
    center_penalty = (abs(cx - fw / 2.0) / fw) + (abs(cy - fh / 2.0) / fh)
    return area_ratio * 0.55 + rectangularity * 0.35 - center_penalty * 0.20


def _detect_screen_box(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, tuple[int, int, int, int]] | None = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < (fw * fh * 0.04):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        rect_area = float(max(1, w * h))
        rectangularity = float(area / rect_area)
        score = _score_candidate(x, y, w, h, fw, fh, rectangularity)
        if score < 0:
            continue
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))

    if best:
        return best[1]

    # Fallback: remove pure black borders only.
    mask = gray > 12
    ys, xs = np.where(mask)
    if len(xs) < 100 or len(ys) < 100:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    if _score_candidate(x0, y0, w, h, fw, fh, rectangularity=0.9) < 0:
        return None
    return (x0, y0, w, h)


def run_pro_v2_screen_preprocess(input_video_path: str, work_dir: str) -> dict[str, Any]:
    """Detect/crop embedded screen video region. Raises RuntimeError when detection/crop fails."""
    logger.info("[PRO_V2][SCREEN] entered")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("screen preprocess failed: cannot open input video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_count = 20
    idxs = np.linspace(0, max(0, total - 1), num=sample_count, dtype=int) if total > 0 else np.array([], dtype=int)

    boxes: list[tuple[int, int, int, int]] = []
    for idx in idxs.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        box = _detect_screen_box(frame)
        if box:
            boxes.append(box)
    cap.release()

    if len(boxes) < 3:
        raise RuntimeError("screen preprocess failed: insufficient screen box candidates")

    arr = np.array(boxes, dtype=np.float32)
    x, y, w, h = np.median(arr, axis=0).astype(int).tolist()
    x, y = max(0, x), max(0, y)
    w, h = _even(w), _even(h)
    confidence = float(min(1.0, max(0.0, len(boxes) / sample_count)))

    out_path = str(work / "pro_v2_screen_cropped.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-vf",
        f"crop={w}:{h}:{x}:{y}",
        "-an",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"screen preprocess failed: ffmpeg crop failed ({proc.stderr[-300:]})")

    logger.info("[PRO_V2][SCREEN] crop_box=%s", {"x": x, "y": y, "w": w, "h": h})
    logger.info("[PRO_V2][SCREEN] cropped_video_path=%s", out_path)
    return {
        "screen_mode_detected": True,
        "cropped_video_path": out_path,
        "crop_box": {"x": x, "y": y, "w": w, "h": h},
        "confidence": round(confidence, 3),
    }

