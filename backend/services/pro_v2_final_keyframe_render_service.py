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


def _decode_frames_batch_sequential(video_path: str, indices: list[int]) -> tuple[dict[int, Any], str | None]:
    """Decode requested indices in one sequential pass (no per-index seek)."""
    wanted = sorted({int(i) for i in indices if int(i) >= 0})
    if not wanted:
        return {}, None
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        return {}, 'VIDEO_OPEN_FAILED'

    out: dict[int, Any] = {}
    want_set = set(wanted)
    end = wanted[-1]
    pos = 0
    while pos <= end:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return out, f'DECODE_STOP_AT_{pos}'
        if pos in want_set:
            out[pos] = frame.copy()
            if len(out) == len(wanted):
                break
        pos += 1
    cap.release()

    if len(out) != len(wanted):
        missing = [i for i in wanted if i not in out][:4]
        return out, f'PARTIAL_BATCH_MISSING_{missing}'
    return out, None


def _target_frame_index(ts_s: float, fps: float, frame_count: int, fallback_idx: int) -> int:
    if fps <= 0.0 or frame_count <= 0:
        return max(0, int(fallback_idx))
    idx = int(round(max(0.0, float(ts_s)) * fps))
    idx = max(0, min(idx, frame_count - 1))
    return idx


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

    source_meta: dict[str, dict[str, Any]] = {}
    for sk, sp in sources:
        fps, n = _source_meta(sp)
        source_meta[sk] = {"path": sp, "fps": fps, "frame_count": n}

    # Build per-source frame index plans first, then decode each source once.
    planned_indices_by_source: dict[str, list[int]] = {sk: [] for sk, _ in sources}
    frame_plan_by_phase: dict[str, dict[str, dict[str, Any]]] = {}
    for row in keyframes:
        phase = str(row.get("phase") or "")
        ts = float(row.get("timestamp") or 0.0)
        fallback_idx = int(row.get("frame_index") or 0)
        frame_plan_by_phase[phase] = {}
        for sk, _ in sources:
            fps = float(source_meta[sk]["fps"])
            n = int(source_meta[sk]["frame_count"])
            idx = _target_frame_index(ts, fps, n, fallback_idx)
            frame_plan_by_phase[phase][sk] = {
                "analysis_timestamp": ts,
                "source_timestamp": (float(idx) / fps) if fps > 0.0 else 0.0,
                "source_frame_index": idx,
            }
            planned_indices_by_source[sk].append(idx)

    decoded_frames_by_source: dict[str, dict[int, Any]] = {}
    decode_err_by_source: dict[str, str] = {}
    for sk, _ in sources:
        uniq_indices = sorted({int(i) for i in planned_indices_by_source.get(sk, []) if int(i) >= 0})
        frames, err = _decode_frames_batch_sequential(str(source_meta[sk]["path"]), uniq_indices)
        decoded_frames_by_source[sk] = frames
        if err:
            decode_err_by_source[sk] = err
        logger.info(
            "[PRO_V2][KEYFRAME_RENDER_BATCH] source_kind=%s sequential_decode=true ffmpeg_extract=false requested=%s decoded=%s error=%s",
            sk,
            len(uniq_indices),
            len(frames),
            err or "",
        )

    out: list[dict[str, Any]] = []
    reasons: list[str] = []
    source_used = "raw"

    for row in keyframes:
        phase = str(row.get("phase") or "")
        ts = float(row.get("timestamp") or 0.0)
        chosen = dict(row)
        rendered = False

        for sk, _ in sources:
            plan = ((frame_plan_by_phase.get(phase) or {}).get(sk) or {})
            src_idx = int(plan.get("source_frame_index") or 0)
            src_ts = float(plan.get("source_timestamp") or 0.0)
            frame = (decoded_frames_by_source.get(sk) or {}).get(src_idx)
            if frame is None:
                continue
            b64 = _jpeg_b64(frame)
            if len(b64) < 48:
                continue
            chosen["image_base64"] = b64
            chosen["analysis_timestamp"] = ts
            chosen["display_source_kind"] = sk
            chosen["display_source_timestamp"] = src_ts
            chosen["display_source_frame_index"] = src_idx
            chosen["display_render_ok"] = True
            chosen["display_render_error"] = ""
            chosen["display_debug"] = {
                "phase": phase,
                "analysis_timestamp": ts,
                "source_kind": sk,
                "source_timestamp": src_ts,
                "source_frame_index": src_idx,
                "read_success": True,
                "failure_reason": "",
                "sequential_decode": True,
                "ffmpeg_extract": False,
            }
            source_used = sk
            rendered = True
            logger.info(
                "[PRO_V2][KEYFRAME_RENDER] phase=%s analysis_ts=%.4f source_kind=%s source_ts=%.4f source_frame_index=%s sequential_decode=true ffmpeg_extract=false",
                phase,
                ts,
                sk,
                src_ts,
                src_idx,
            )
            break

        if not rendered:
            fail_reason = "FRAME_NOT_DECODED"
            for sk, _ in sources:
                if sk in decode_err_by_source:
                    fail_reason = f"{sk}:{decode_err_by_source[sk]}"
                    break
            reasons.append(f"{phase.upper()}_IMAGE_MISSING")
            chosen.setdefault("image_base64", "")
            chosen["analysis_timestamp"] = ts
            chosen["display_source_kind"] = "missing"
            chosen["display_source_timestamp"] = 0.0
            chosen["display_source_frame_index"] = -1
            chosen["display_render_ok"] = False
            chosen["display_render_error"] = fail_reason
            chosen["display_debug"] = {
                "phase": phase,
                "analysis_timestamp": ts,
                "source_kind": "missing",
                "source_timestamp": 0.0,
                "source_frame_index": -1,
                "read_success": False,
                "failure_reason": fail_reason,
                "sequential_decode": True,
                "ffmpeg_extract": False,
            }
            logger.warning(
                "[PRO_V2][KEYFRAME_RENDER_FAIL] phase=%s analysis_ts=%.4f source_kind=missing source_ts=0 source_frame_index=-1 sequential_decode=true ffmpeg_extract=false reason=%s",
                phase,
                ts,
                fail_reason,
            )
        out.append(chosen)

    logger.info("[PRO_V2][KEYFRAME_SOURCE] source_used=%s missing=%s", source_used, reasons[:8])
    return out, source_used, list(dict.fromkeys(reasons))
