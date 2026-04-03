"""Frame reads for Pro v2 analysis video (H.264 / MP4).

OpenCV's CAP_PROP_POS_FRAMES + read() can snap to keyframes, so multiple seeks to
different indices may return the same decoded image. Use sequential decode for the
eight keyframe JPEG batch (`read_frames_bgr_at_indices`). For scattered single reads
(gate repick, impact window samples, final re-encode) seek is fast enough and errors
stay local; decoding from frame 0 to a late index on every call is prohibitively slow.

Sequential decode from frame 0 through the largest requested index can still be huge
on long files; cap via STELLAR_PRO_V2_MAX_SEQUENTIAL_FRAMES (default 65536) and fall
back to per-index seek so workers stay within time limits (possible H.264 snap dupes).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)

# ~273s at 240fps; override if your pipeline only uses short clips.
_MAX_SEQ_FRAMES = int(os.getenv("STELLAR_PRO_V2_MAX_SEQUENTIAL_FRAMES", "65536"))


def read_frame_bgr_seek(video_path: str, frame_index: int) -> Any | None:
    """One frame via CAP_PROP_POS_FRAMES + read(). Fast; may snap near keyframes."""
    idx = int(frame_index)
    if idx < 0:
        return None
    path = str(Path(video_path))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def read_frames_bgr_at_indices(video_path: str, indices: list[int]) -> dict[int, Any]:
    """Decode from frame 0; return {frame_index: bgr ndarray} for requested indices.

    If the last index is too large, use per-index seek instead of scanning from 0
    (avoids timeouts that broke Modal/Render after naive sequential-only reads).
    """
    wanted = sorted({int(i) for i in indices if int(i) >= 0})
    if not wanted:
        return {}
    path = str(Path(video_path))
    end = wanted[-1]
    if end > _MAX_SEQ_FRAMES:
        logger.warning(
            "[PRO_V2][FRAME_READ] max_index=%s > cap=%s: using seek per index (H.264 snaps possible)",
            end,
            _MAX_SEQ_FRAMES,
        )
        out: dict[int, Any] = {}
        for i in wanted:
            f = read_frame_bgr_seek(path, i)
            if f is not None:
                out[i] = f
        return out

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    want_set = set(wanted)
    out = {}
    pos = 0
    while pos <= end:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if pos in want_set:
            out[pos] = frame.copy()
        pos += 1
    cap.release()
    return out


def read_frame_bgr_at_index(video_path: str, frame_index: int) -> Any | None:
    """Alias for seek-based read; prefer `read_frames_bgr_at_indices` for batched exact indices."""
    return read_frame_bgr_seek(video_path, frame_index)
