"""Pro Stage 5: OpenCV-first impact refinement on high-fps impact clip; AI optional confirm stub."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from services.video_utils import get_video_rotation, read_frame_pose_pipeline

logger = logging.getLogger(__name__)


def _laplacian_focus_score(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def refine_impact_pose_index(
    impact_window_video_path: str | None,
    analysis_video_path: str,
    poses: list[dict],
    rough_pose_idx: int,
    *,
    impact_window_start_s: float | None = None,
    analysis_fps: float = 240.0,
    search_radius_poses: int = 8,
) -> tuple[int, dict[str, Any]]:
    """
    Prefer sharpness peak on impact_window clip; map winning clip frame → analysis timeline
    via impact_window_start_s + clip_fps, then nearest pose by frame_index.
    """
    n = len(poses)
    rough_pose_idx = max(0, min(n - 1, int(rough_pose_idx)))
    meta: dict[str, Any] = {"method": "opencv_laplacian", "clip_used": False}

    if impact_window_video_path and impact_window_start_s is not None:
        cap = cv2.VideoCapture(impact_window_video_path)
        if cap.isOpened():
            clip_fps = float(cap.get(cv2.CAP_PROP_FPS) or analysis_fps)
            scored: list[tuple[float, int]] = []
            fi = 0
            while True:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                sc = _laplacian_focus_score(fr)
                scored.append((sc, fi))
                fi += 1
            cap.release()
            best_fi = -1
            best_sc = -1.0
            if scored:
                best_sc = float(max(s for s, _ in scored))
                # Global max is often post-impact (ball in flight, club still sharp).
                # Prefer the earliest frame within ~top 8% sharpness (contact-side).
                thresh = best_sc * 0.92
                early = [i for s, i in scored if s >= thresh]
                best_fi = int(min(early)) if early else int(max(scored, key=lambda x: x[0])[1])
            if best_fi >= 0 and clip_fps > 1e-3:
                meta["clip_used"] = True
                meta["best_frame_in_clip"] = int(best_fi)
                meta["laplacian_score"] = round(best_sc, 3)
                t_hit = float(impact_window_start_s) + float(best_fi) / clip_fps
                target_fi = int(round(t_hit * float(analysis_fps)))
                best_pi = rough_pose_idx
                best_dist = 1e18
                for pi in range(n):
                    d = abs(int(poses[pi].get("frame_index", pi)) - target_fi)
                    if d < best_dist:
                        best_dist = d
                        best_pi = pi
                logger.info(
                    "[STELLAR_PRO][IMPACT_REFINE] stage=done clip=yes rough=%s refined=%s target_fi=%s",
                    rough_pose_idx,
                    best_pi,
                    target_fi,
                )
                return int(best_pi), meta

    # Fallback: scan analysis video around rough frame indices
    cap = cv2.VideoCapture(analysis_video_path)
    if not cap.isOpened():
        return rough_pose_idx, meta
    rotation = get_video_rotation(analysis_video_path)
    lo = max(0, rough_pose_idx - search_radius_poses)
    hi = min(n - 1, rough_pose_idx + search_radius_poses)
    scored_pi: list[tuple[float, int]] = []
    for pi in range(lo, hi + 1):
        fi = int(poses[pi].get("frame_index", pi))
        fr = read_frame_pose_pipeline(cap, fi, rotation)
        if fr is None:
            continue
        sc = _laplacian_focus_score(fr)
        scored_pi.append((sc, pi))
    best_pi = rough_pose_idx
    best_sc = -1.0
    if scored_pi:
        best_sc = float(max(s for s, _ in scored_pi))
        thresh = best_sc * 0.92
        early_pi = [pi for s, pi in scored_pi if s >= thresh]
        best_pi = int(min(early_pi)) if early_pi else int(max(scored_pi, key=lambda x: x[0])[1])
    cap.release()
    meta["clip_used"] = False
    meta["laplacian_score"] = round(best_sc, 3) if best_sc >= 0 else None
    logger.info(
        "[STELLAR_PRO][IMPACT_REFINE] stage=done clip=no rough=%s refined=%s",
        rough_pose_idx,
        best_pi,
    )
    return int(best_pi), meta
