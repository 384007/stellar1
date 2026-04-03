"""Pro Stage 6: lightweight quality gate — ordering, spacing, duplicates. Not a keyframe picker."""

from __future__ import annotations

import logging
from typing import Any

import cv2

from services.keyframe_service import (
    PHASE_ORDER,
    recompute_keyframe_details_from_final_strip,
    validate_final_keyframes_for_ai,
)
from services.pro_motion_keyframe_service import (
    build_keyframes_from_motion_picks,
    enforce_monotonic_phase_picks,
)
from services.pro_phase_motion_pick_service import pick_phase_pose_index_from_window
from services.video_utils import get_video_rotation

logger = logging.getLogger(__name__)


def _bad_phases_from_details(details: list[dict]) -> set[str]:
    out: set[str] = set()
    for d in details:
        if not isinstance(d, dict):
            continue
        ph = str(d.get("phase") or "")
        if not ph:
            continue
        if (
            not d.get("validation_passed", True)
            or d.get("is_near_duplicate")
            or d.get("time_too_close")
        ):
            out.add(ph)
    return out


def run_pro_final_gate(
    analysis_video_path: str,
    poses: list[dict],
    phase_keyframes: dict[str, int],
    windows: list[dict[str, Any]],
    *,
    features: dict[str, Any],
    events: dict[str, Any],
    analysis_fps: float,
    keyframe_width: int,
    min_time_gap: float,
    impact_refine_fn: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """
    Validate strip; on failure snap only failing phases (window center + motion re-pick once),
    re-enforce monotonic order, optionally re-run impact refine if impact was adjusted.
    """
    t0 = __import__("time").perf_counter()
    pk = dict(phase_keyframes)
    wby = {str(w["phase"]): w for w in windows}

    def _one_pass(picks: dict[str, int]) -> tuple[list[dict], list[dict], dict]:
        kfs = build_keyframes_from_motion_picks(
            analysis_video_path,
            poses,
            picks,
            analysis_fps=analysis_fps,
            keyframe_width=keyframe_width,
        )
        cap = cv2.VideoCapture(analysis_video_path)
        if not cap.isOpened():
            raise RuntimeError("analysis_video_missing")
        rotation = get_video_rotation(analysis_video_path)
        try:
            det = recompute_keyframe_details_from_final_strip(
                cap, rotation, float(analysis_fps), poses, kfs, min_time_gap,
            )
            gate = validate_final_keyframes_for_ai(
                kfs, picks, det, poses=poses, fps=float(analysis_fps),
            )
        finally:
            cap.release()
        return kfs, det, gate

    kfs, det, gate = _one_pass(pk)
    passed = bool(gate.get("pass"))
    retry_used = False

    if not passed:
        bad = _bad_phases_from_details(det)
        if not bad:
            bad = set(PHASE_ORDER[-3:])

        for ph in bad:
            if ph in wby:
                w = wby[ph]
                pk[ph] = pick_phase_pose_index_from_window(
                    ph,
                    int(w["start_pose_idx"]),
                    int(w["end_pose_idx"]),
                    poses=poses,
                    features=features,
                    events=events,
                )
        pk = enforce_monotonic_phase_picks(pk, len(poses))

        if "impact" in bad and callable(impact_refine_fn):
            pk["impact"] = int(impact_refine_fn(int(pk["impact"])))
            pk = enforce_monotonic_phase_picks(pk, len(poses))

        kfs, det, gate = _one_pass(pk)
        retry_used = True

    wall = round(__import__("time").perf_counter() - t0, 3)
    summary = {
        "pass": bool(gate.get("pass")),
        "retry_used": retry_used,
        "wall_s": wall,
        "near_duplicates": int(gate.get("near_duplicates", 0)),
        "time_too_close_count": int(gate.get("time_too_close_count", 0)),
    }
    logger.info("[STELLAR_PRO][FINAL_GATE] stage=done %s", summary)
    return kfs, pk, summary
