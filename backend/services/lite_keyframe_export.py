"""
Lite-only keyframe JPEG export: OpenCV seek/read/write only.
No Pro ffmpeg, ffprobe, frame_enhance_service, or R2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from lib.prov3.keyframes.constants import EVENT_SEQUENCE

_EVENT_FILE_NAMES: dict[str, str] = {
    "Address": "address.jpg",
    "Toe-up": "toe_up.jpg",
    "Mid-backswing": "mid_backswing.jpg",
    "Top": "top.jpg",
    "Mid-downswing": "mid_downswing.jpg",
    "Impact": "impact.jpg",
    "Mid-follow-through": "mid_follow_through.jpg",
    "Finish": "finish.jpg",
}


def lite_persist_keyframe_images(
    video_path: str,
    keyframes: list[dict[str, Any]],
    output_dir: str,
) -> list[dict[str, Any]]:
    """
    Write exactly eight phase JPEGs using decode indices on ``video_path`` (cleaned lite video).
    ``frame_index`` in each row is treated as OpenCV decode index (same as unified lite timeline).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    by_event: dict[str, dict[str, Any]] = {}
    for row in keyframes:
        ev = str(row.get("event_name") or "")
        if ev in _EVENT_FILE_NAMES:
            by_event[ev] = row

    if len(by_event) != 8:
        missing = [e for e in EVENT_SEQUENCE if e not in by_event]
        raise RuntimeError(
            f"lite_keyframe_export: expected 8 events, got {len(by_event)}, missing={missing}"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"lite_keyframe_export: cannot open video {video_path!r}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"lite_keyframe_export: zero frame count for {video_path!r}")

    max_fi = max(0, total - 1)
    saved: list[dict[str, Any]] = []

    try:
        for event_name in EVENT_SEQUENCE:
            row = by_event[event_name]
            fi_raw = int(row.get("frame_index") or 0)
            fi = max(0, min(fi_raw, max_fi))
            fname = _EVENT_FILE_NAMES[event_name]
            out_path = str(Path(output_dir) / fname)

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(
                    f"lite_keyframe_export: decode_failed event={event_name} "
                    f"frame_index={fi} (raw={fi_raw}) video={video_path!r}"
                )
            if not cv2.imwrite(out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"lite_keyframe_export: imwrite_failed path={out_path}")
            p = Path(out_path)
            if not p.is_file() or p.stat().st_size < 256:
                raise RuntimeError(f"lite_keyframe_export: empty_or_tiny_jpeg path={out_path}")

            saved.append(
                {
                    "event_name": event_name,
                    "frame_index": fi_raw,
                    "file_name": fname,
                    "file_path": out_path,
                    "image_path": out_path,
                    "keyframe_image_source": "lite_opencv",
                }
            )
    finally:
        cap.release()

    if len(saved) != 8:
        raise RuntimeError(f"lite_keyframe_export: expected 8 saved files, got {len(saved)}")
    return saved
