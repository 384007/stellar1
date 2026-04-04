"""Render display keyframes from source videos (never from analysis_240)."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)



def _jpeg_b64(frame_bgr: Any, quality: int = 90) -> str:
    ok, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ''
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _source_meta(video_path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        return 0.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def _read_frame_by_timestamp(video_path: str, ts_s: float, *, fallback_idx: int = 0) -> tuple[Any | None, int]:
    fps, n = _source_meta(video_path)
    if fps <= 0.0 or n <= 0:
        return None, -1
    idx = int(round(max(0.0, float(ts_s)) * fps))
    if idx < 0:
        idx = 0
    if n > 0:
        idx = min(idx, n - 1)
    if idx < 0:
        idx = max(0, int(fallback_idx))
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        return None, -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, -1
    return frame, idx


def render_display_keyframes_from_sources(
    keyframes: list[dict[str, Any]],
    *,
    screen_clean_video_path: str | None,
    screen_cropped_video_path: str | None,
    raw_video_path: str,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Back-source keyframes by timestamp. Preferred source: clean -> cropped -> raw."""
    sources: list[tuple[str, str]] = []
    if screen_clean_video_path:
        sources.append(("screen_clean", screen_clean_video_path))
    if screen_cropped_video_path:
        sources.append(("screen_cropped", screen_cropped_video_path))
    sources.append(("raw", raw_video_path))

    out: list[dict[str, Any]] = []
    reasons: list[str] = []
    source_used = "raw"

    for row in keyframes:
        phase = str(row.get("phase") or "")
        ts = float(row.get("timestamp") or 0.0)
        rendered = False
        chosen = dict(row)
        for sk, sp in sources:
            frame, src_idx = _read_frame_by_timestamp(sp, ts, fallback_idx=int(row.get("frame_index") or 0))
            if frame is None:
                continue
            b64 = _jpeg_b64(frame)
            if len(b64) < 48:
                continue
            chosen["image_base64"] = b64
            chosen["display_source_kind"] = sk
            chosen["display_source_frame_index"] = int(src_idx)
            source_used = sk
            rendered = True
            logger.info(
                "[PRO_V2][KEYFRAME_RENDER] phase=%s ts=%.4f source_kind=%s source_frame_index=%s",
                phase,
                ts,
                sk,
                src_idx,
            )
            break
        if not rendered:
            reasons.append(f"{phase.upper()}_IMAGE_MISSING")
            chosen.setdefault("image_base64", "")
            chosen["display_source_kind"] = "missing"
            chosen["display_source_frame_index"] = -1
        out.append(chosen)

    logger.info("[PRO_V2][KEYFRAME_SOURCE] source_used=%s missing=%s", source_used, reasons[:8])
    return out, source_used, list(dict.fromkeys(reasons))
