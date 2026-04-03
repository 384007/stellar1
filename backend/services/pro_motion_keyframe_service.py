"""Pro Stage 4: motion engine — sole authority for which pose index represents each phase.

OpenCV refines impact only (see pro_impact_refine_service). AI does not select frames.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2

from services.keyframe_service import (
    PHASE_ORDER,
    SWING_PHASE_META,
    _pose_snapshot_for_keyframe,
    _read_frame_pose_matched,
    _read_frame_with_decode_fallback,
    _resize_frame,
    frame_to_base64,
)
from services.pro_phase_motion_pick_service import (
    log_motion_pick_summary,
    pick_phase_pose_index_from_window,
)
from services.video_utils import collect_frames_at_indices_sequential, get_video_rotation

logger = logging.getLogger(__name__)


def enforce_monotonic_phase_picks(picks: dict[str, int], n_poses: int) -> dict[str, int]:
    """Force strictly increasing pose indices in canonical phase order."""
    out: dict[str, int] = {}
    prev = -1
    for ph in PHASE_ORDER:
        v = int(picks.get(ph, prev + 1))
        v = max(prev + 1, min(n_poses - 1, v))
        out[ph] = v
        prev = v
    return out


def _pose_fi(poses: list[dict], pi: int) -> int:
    pi = max(0, min(len(poses) - 1, int(pi)))
    return int(poses[pi].get("frame_index", pi))


def enforce_late_strip_spacing(
    picks: dict[str, int],
    poses: list[dict],
    windows: list[dict[str, Any]],
    *,
    n_poses: int,
) -> dict[str, int]:
    """Spread downswing → finish so late phases do not collapse near impact."""
    wby = {str(w["phase"]): w for w in windows}
    out = dict(picks)
    min_pose = max(2, n_poses // 48)
    fi0 = _pose_fi(poses, 0)
    fi1 = _pose_fi(poses, n_poses - 1)
    span_fi = max(1, fi1 - fi0)
    min_fi = max(2, int(span_fi * 0.032))

    late = ["downswing", "impact", "follow_through", "finish"]

    def nudge_next(ph_prev: str, ph_next: str) -> None:
        p0 = out[ph_prev]
        p1 = out[ph_next]
        if p1 <= p0:
            w = wby.get(ph_next, {})
            lo, hi = int(w["start_pose_idx"]), int(w["end_pose_idx"])
            out[ph_next] = max(lo, min(hi, p0 + 1))
            p1 = out[ph_next]
        if p1 < p0 + min_pose:
            need = p0 + min_pose
            w = wby.get(ph_next, {})
            lo, hi = int(w["start_pose_idx"]), int(w["end_pose_idx"])
            out[ph_next] = int(max(lo, min(hi, need)))
            p1 = out[ph_next]
        f_prev = _pose_fi(poses, out[ph_prev])
        f_next = _pose_fi(poses, out[ph_next])
        if f_next < f_prev + min_fi:
            target_fi = f_prev + min_fi
            w = wby[ph_next]
            lo, hi = int(w["start_pose_idx"]), int(w["end_pose_idx"])
            best_pi = out[ph_next]
            best_cost = 1e18
            for pi in range(lo, hi + 1):
                if pi <= out[ph_prev]:
                    continue
                cost = abs(_pose_fi(poses, pi) - target_fi) + 0.3 * abs(pi - (lo + hi) // 2)
                if cost < best_cost:
                    best_cost = cost
                    best_pi = pi
            out[ph_next] = int(best_pi)

    for i in range(len(late) - 1):
        nudge_next(late[i], late[i + 1])

    imp = out["impact"]
    fin = out["finish"]
    min_finish_pose = imp + max(min_pose * 2, n_poses // 14)
    if fin < min_finish_pose:
        w = wby["finish"]
        lo, hi = int(w["start_pose_idx"]), int(w["end_pose_idx"])
        out["finish"] = int(max(lo, min(hi, max(fin, min_finish_pose))))

    return out


def _late_strip_gap_summary(picks: dict[str, int], poses: list[dict]) -> dict[str, Any]:
    late = ["downswing", "impact", "follow_through", "finish"]
    pose_gaps = []
    fi_gaps = []
    for i in range(len(late) - 1):
        a, b = late[i], late[i + 1]
        pose_gaps.append(picks[b] - picks[a])
        fi_gaps.append(_pose_fi(poses, picks[b]) - _pose_fi(poses, picks[a]))
    return {
        "pose_gaps_ds_imp_ft_fn": pose_gaps,
        "frame_gaps_ds_imp_ft_fn": fi_gaps,
        "picks": {k: picks[k] for k in late},
    }


def select_motion_keyframe_picks(
    windows: list[dict[str, Any]],
    poses: list[dict],
    features: dict[str, Any],
    events: dict[str, Any],
) -> dict[str, int]:
    """Pick one pose per phase; refine late-strip spacing; then monotonic clamp."""
    n = len(poses)
    wby = {str(w["phase"]): w for w in windows}
    picks: dict[str, int] = {}
    pick_meta: list[dict[str, Any]] = []

    for ph in PHASE_ORDER:
        if ph not in wby:
            raise RuntimeError(f"STELLAR_PRO_MOTION_KF: missing window for {ph}")
        w = wby[ph]
        picks[ph] = pick_phase_pose_index_from_window(
            ph,
            int(w["start_pose_idx"]),
            int(w["end_pose_idx"]),
            poses=poses,
            features=features,
            events=events,
            pick_meta_out=pick_meta,
        )

    log_motion_pick_summary(pick_meta)

    before = {k: picks[k] for k in PHASE_ORDER}
    picks = enforce_late_strip_spacing(picks, poses, windows, n_poses=n)
    strip_sum = _late_strip_gap_summary(picks, poses)
    logger.info(
        "[STELLAR_PRO][LATE_STRIP] before=%s after=%s gaps_pose=%s gaps_fi=%s",
        before,
        {k: picks[k] for k in PHASE_ORDER},
        strip_sum["pose_gaps_ds_imp_ft_fn"],
        strip_sum["frame_gaps_ds_imp_ft_fn"],
    )

    picks = enforce_monotonic_phase_picks(picks, n)
    logger.info(
        "[STELLAR_PRO][MOTION_KEYFRAME] final_picks=%s",
        {k: picks[k] for k in PHASE_ORDER},
    )
    return picks


def build_keyframes_from_motion_picks(
    analysis_video_path: str,
    poses: list[dict],
    phase_keyframes: dict[str, int],
    *,
    analysis_fps: float,
    keyframe_width: int,
) -> list[dict[str, Any]]:
    """Decode one frame per phase at motion-picked pose indices; encode JPEG base64."""
    rotation = get_video_rotation(analysis_video_path)
    planned: list[tuple[str, int, int]] = []
    for phase in PHASE_ORDER:
        pi = int(phase_keyframes[phase])
        pi = max(0, min(len(poses) - 1, pi))
        pose = poses[pi]
        fi = int(pose.get("frame_index", pi))
        planned.append((phase, pi, fi))

    fis = [p[2] for p in planned]
    frame_by_fi = collect_frames_at_indices_sequential(
        analysis_video_path, fis, rotation=rotation,
    )

    cap_fb: cv2.VideoCapture | None = None
    rows: list[dict[str, Any]] = []
    try:
        for phase, pi, fi in planned:
            ts = float(
                poses[pi].get("timestamp", round(fi / max(analysis_fps, 1e-6), 4)),
            )
            frame = frame_by_fi.get(fi)
            if frame is None:
                if cap_fb is None:
                    cap_fb = cv2.VideoCapture(analysis_video_path)
                    if not cap_fb.isOpened():
                        raise RuntimeError("cannot_open_analysis_video")
                frame = _read_frame_pose_matched(cap_fb, fi, rotation)
                if frame is None:
                    frame = _read_frame_with_decode_fallback(cap_fb, fi, rotation)
            if frame is None:
                raise RuntimeError(f"frame_decode_fail phase={phase} fi={fi}")
            resized = _resize_frame(frame, keyframe_width)
            meta = SWING_PHASE_META.get(phase, {})
            pose = poses[pi]
            rows.append({
                "phase": phase,
                "label_en": meta.get("label_en", phase),
                "label_zh": meta.get("label_zh", phase),
                "source_pose_idx": pi,
                "source_frame_index": fi,
                "frame_index": fi,
                "timestamp": round(ts, 4),
                "confidence": 0.88,
                "pose_snapshot": _pose_snapshot_for_keyframe(pose),
                "image_base64": frame_to_base64(resized),
                "width": resized.shape[1],
                "height": resized.shape[0],
            })
    finally:
        if cap_fb is not None:
            cap_fb.release()
    return rows
