"""Pro v2 — single-pass validation + at most one local re-pick from dense frames."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2

from services.pro_v2_dense_scan_service import DenseFrame
from services.pro_v2_frame_read import read_frame_bgr_seek

logger = logging.getLogger(__name__)

MIN_GAP_FRAMES = 5
LATE_MIN_GAP_S = 0.08
IMPACT_FOLLOW_MIN_GAP_S = 0.05


def _jpeg_b64(frame: Any, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _violations(
    keyframes: list[dict[str, Any]],
    *,
    min_gap_s: float,
) -> list[str]:
    issues: list[str] = []
    for a, b in zip(keyframes, keyframes[1:], strict=False):
        if float(b["timestamp"]) <= float(a["timestamp"]):
            issues.append("non_monotonic_time")
        if int(b["frame_index"]) == int(a["frame_index"]):
            issues.append("adjacent_duplicate_frame")
        if float(b["timestamp"]) - float(a["timestamp"]) < min_gap_s - 1e-9:
            issues.append("frame_gap")

    ft = next((k for k in keyframes if k.get("phase") == "follow_through"), None)
    fin = next((k for k in keyframes if k.get("phase") == "finish"), None)
    imp = next((k for k in keyframes if k.get("phase") == "impact"), None)
    if imp and ft and float(ft["timestamp"]) - float(imp["timestamp"]) < IMPACT_FOLLOW_MIN_GAP_S:
        issues.append("impact_follow_crowded")
    if ft and fin and float(fin["timestamp"]) - float(ft["timestamp"]) < LATE_MIN_GAP_S:
        issues.append("follow_finish_crowded")
    return issues


def _assign_from_dense(
    video_path: str,
    kf: dict[str, Any],
    d: DenseFrame,
    *,
    fps: float,
) -> None:
    frame = read_frame_bgr_seek(video_path, d.frame_index)
    if frame is None:
        return
    fi = int(d.frame_index)
    kf["frame_index"] = fi
    kf["source_frame_index"] = fi
    kf["source_pose_idx"] = fi
    kf["timestamp"] = float(d.timestamp_s) if d.timestamp_s > 0 else round(d.frame_index / fps, 5)
    kf["image_base64"] = _jpeg_b64(frame)


def _first_dense_after(
    dense: list[DenseFrame],
    min_frame: int,
    *,
    min_step: int,
) -> DenseFrame | None:
    need = int(min_frame) + int(min_step)
    for d in dense:
        if d.frame_index >= need:
            return d
    return None


def _first_dense_low_motion_after(
    dense: list[DenseFrame],
    min_frame: int,
    *,
    min_step: int,
) -> DenseFrame | None:
    need = int(min_frame) + int(min_step)
    tail = [d for d in dense if d.frame_index >= need]
    if not tail:
        return None
    return min(tail, key=lambda d: float(d.motion_energy_smooth))


def _one_local_repick(
    video_path: str,
    keyframes: list[dict[str, Any]],
    dense: list[DenseFrame],
    *,
    fps: float,
    min_gap_frames: int,
) -> bool:
    """Apply a single local fix using the dense ladder. Returns True if a change was made."""
    min_step = max(MIN_GAP_FRAMES, min_gap_frames)
    kfs = keyframes
    for i in range(len(kfs) - 1):
        a, b = kfs[i], kfs[i + 1]
        fi_a = int(a["frame_index"])
        fi_b = int(b["frame_index"])
        if fi_b == fi_a:
            cand = _first_dense_after(dense, fi_a, min_step=1)
            if cand:
                _assign_from_dense(video_path, b, cand, fps=fps)
                return True
        if float(b["timestamp"]) <= float(a["timestamp"]) or fi_b <= fi_a:
            cand = _first_dense_after(dense, fi_a, min_step=1)
            if cand:
                _assign_from_dense(video_path, b, cand, fps=fps)
                return True
        min_gap_s = min_step / fps
        if float(b["timestamp"]) - float(a["timestamp"]) < min_gap_s:
            cand = _first_dense_after(dense, fi_a, min_step=min_step)
            if cand:
                _assign_from_dense(video_path, b, cand, fps=fps)
                return True

    imp = next((k for k in kfs if k.get("phase") == "impact"), None)
    ft = next((k for k in kfs if k.get("phase") == "follow_through"), None)
    fin = next((k for k in kfs if k.get("phase") == "finish"), None)
    if imp and ft:
        gap_if = float(ft["timestamp"]) - float(imp["timestamp"])
        logger.info("[PRO_V2][GATE] impact_follow_gap=%.5f", gap_if)
        if gap_if < IMPACT_FOLLOW_MIN_GAP_S:
            cand = _first_dense_after(
                dense,
                int(imp["frame_index"]),
                min_step=max(MIN_GAP_FRAMES, int(round(IMPACT_FOLLOW_MIN_GAP_S * fps))),
            )
            if cand:
                logger.info("[PRO_V2][GATE] local_repick_target=follow_through")
                _assign_from_dense(video_path, ft, cand, fps=fps)
                return True

    if ft and fin:
        gap_ff = float(fin["timestamp"]) - float(ft["timestamp"])
        logger.info("[PRO_V2][GATE] follow_finish_gap=%.5f", gap_ff)
        need = int(ft["frame_index"]) + max(MIN_GAP_FRAMES, 3)
        if int(fin["frame_index"]) < need or gap_ff < LATE_MIN_GAP_S:
            cand = _first_dense_low_motion_after(dense, int(ft["frame_index"]), min_step=max(MIN_GAP_FRAMES, 4))
            if cand:
                logger.info("[PRO_V2][GATE] local_repick_target=finish")
                _assign_from_dense(video_path, fin, cand, fps=fps)
                return True
    return False


def run_simple_gate(
    keyframes: list[dict[str, Any]],
    *,
    fps: float,
    analysis_video_path: str | None = None,
    dense: list[DenseFrame] | None = None,
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Validate ordering/gaps; if needed, at most one local re-pick from dense (refreshes JPEG)."""
    kfs = [dict(k) for k in keyframes]
    min_gap_s = max(MIN_GAP_FRAMES / max(fps, 1e-3), 5.0 / 240.0)

    bad = _violations(kfs, min_gap_s=min_gap_s)
    if bad and analysis_video_path and dense:
        changed = _one_local_repick(
            analysis_video_path,
            kfs,
            dense,
            fps=fps,
            min_gap_frames=MIN_GAP_FRAMES,
        )
        if changed:
            logger.info("[PRO_V2][GATE] applied_one_local_repick first_fail_was=%s", bad[:3])

    bad2 = _violations(kfs, min_gap_s=min_gap_s)
    ok = len(bad2) == 0
    if not ok:
        logger.warning("[PRO_V2][GATE] fail=%s", bad2)
    return kfs, ok, bad2
