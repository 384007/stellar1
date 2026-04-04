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


def _clamp_box(x: int, y: int, w: int, h: int, fw: int, fh: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, fw - 2))
    y = max(0, min(y, fh - 2))
    w = max(2, min(w, fw - x))
    h = max(2, min(h, fh - y))
    return x, y, _even(w), _even(h)


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


def _trim_inner_noise(gray: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return box
    row_var = np.var(roi.astype(np.float32), axis=1)
    col_var = np.var(roi.astype(np.float32), axis=0)
    r_thr = float(np.percentile(row_var, 25))
    c_thr = float(np.percentile(col_var, 25))
    keep_rows = np.where(row_var >= r_thr)[0]
    keep_cols = np.where(col_var >= c_thr)[0]
    if len(keep_rows) > 20 and len(keep_cols) > 20:
        y0, y1 = int(keep_rows[0]), int(keep_rows[-1])
        x0, x1 = int(keep_cols[0]), int(keep_cols[-1])
        x = x + x0
        y = y + y0
        w = x1 - x0 + 1
        h = y1 - y0 + 1
    return x, y, w, h


def _detect_screen_box(frame: np.ndarray) -> tuple[int, int, int, int, float] | None:
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
        bx = _trim_inner_noise(gray, best[1])
        x, y, w, h = _clamp_box(*bx, fw, fh)
        conf = float(max(0.0, min(1.0, best[0])))
        return x, y, w, h, conf

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
    x0, y0, w, h = _clamp_box(x0, y0, w, h, fw, fh)
    return (x0, y0, w, h, 0.42)


def run_pro_v2_screen_preprocess(
    input_video_path: str,
    work_dir: str,
    *,
    relaxed_margin: float = 0.0,
    apply_unsharp: bool = False,
    unsharp_profile: str = "normal",
) -> dict[str, Any]:
    """Detect/crop embedded screen video region. Raises RuntimeError when detection/crop fails.

    relaxed_margin: 0..0.12 expands crop outward (retry round 2) to include more context.
    apply_unsharp: when True (routing use_deblur or retry policy), unsharp after crop.
    unsharp_profile: ``normal`` | ``strong`` — stronger pass for retry / missing-frame recovery.
    """
    prof = str(unsharp_profile or "normal").strip().lower()
    if prof not in ("normal", "strong"):
        prof = "normal"
    logger.info(
        "[PRO_V2][SCREEN] entered relaxed_margin=%.4f apply_unsharp=%s unsharp_profile=%s",
        float(relaxed_margin or 0.0),
        "true" if apply_unsharp else "false",
        prof,
    )
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("screen preprocess failed: cannot open input video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 1080
    sample_count = 20
    idxs = np.linspace(0, max(0, total - 1), num=sample_count, dtype=int) if total > 0 else np.array([], dtype=int)

    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    for idx in idxs.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        box = _detect_screen_box(frame)
        if box:
            x, y, w, h, conf = box
            boxes.append((x, y, w, h))
            scores.append(conf)
    cap.release()

    if len(boxes) < 3:
        raise RuntimeError("screen preprocess failed: insufficient screen box candidates")

    arr = np.array(boxes, dtype=np.float32)
    x, y, w, h = np.median(arr, axis=0).astype(int).tolist()
    x, y, w, h = _clamp_box(x, y, w, h, fw=frame_w, fh=frame_h)
    rm = max(0.0, min(0.12, float(relaxed_margin or 0.0)))
    if rm > 0:
        pad_x = int(round(w * rm))
        pad_y = int(round(h * rm))
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(frame_w - x, w + 2 * pad_x)
        h = min(frame_h - y, h + 2 * pad_y)
        x, y, w, h = _clamp_box(x, y, w, h, fw=frame_w, fh=frame_h)
        logger.info("[PRO_V2][RETRY] screen_preprocess relaxed expand pad=(%s,%s)", pad_x, pad_y)
    confidence = float(min(1.0, max(0.0, (len(boxes) / sample_count) * 0.6 + (float(np.median(scores)) if scores else 0.0) * 0.4)))

    out_path = str(work / "pro_v2_screen_cropped.mp4")
    crop_vf = f"crop={w}:{h}:{x}:{y}"
    if apply_unsharp:
        amt = 1.05 if prof == "strong" else 0.78
        crop_vf += f",unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={amt}"
        logger.info("[PRO_V2][SCREEN] ffmpeg_unsharp luma_amount=%s profile=%s", amt, prof)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-vf",
        crop_vf,
        "-an",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"screen preprocess failed: ffmpeg crop failed ({proc.stderr[-300:]})")

    logger.info("[PRO_V2][SCREEN] screen_mode_detected=true")
    logger.info("[PRO_V2][SCREEN] crop_box=%s", {"x": x, "y": y, "w": w, "h": h})
    logger.info("[PRO_V2][SCREEN] confidence=%.3f", confidence)
    logger.info("[PRO_V2][SCREEN] cropped_video_path=%s", out_path)
    return {
        "screen_mode_detected": True,
        "cropped_video_path": out_path,
        "crop_box": {"x": x, "y": y, "w": w, "h": h},
        "confidence": round(confidence, 3),
        "source_frame_size": {"w": frame_w, "h": frame_h},
    }
