"""
Swing Flow phase detection and stabilization.
Uses multi-signal joint kinematics (wrist, shoulder, hip) to identify
8 golf swing phases with monotonic progression.

Includes pose quality filtering to handle camera cuts and close-ups
in broadcast videos.
"""

import logging
import math
import numpy as np
from typing import Any, Optional

from services.json_sanitize import safe_float

logger = logging.getLogger(__name__)

SWING_PHASES = [
    {"id": "address",        "en": "Address",        "zh": "准备", "pct": (0,  8)},
    {"id": "takeaway",       "en": "Takeaway",       "zh": "起杆", "pct": (8,  22)},
    {"id": "backswing",      "en": "Backswing",      "zh": "上杆", "pct": (22, 42)},
    {"id": "top",            "en": "Top",            "zh": "顶点", "pct": (42, 52)},
    {"id": "downswing",      "en": "Downswing",      "zh": "下杆", "pct": (52, 68)},
    {"id": "impact",         "en": "Impact",         "zh": "击球", "pct": (68, 76)},
    {"id": "follow_through", "en": "Follow-Through", "zh": "送杆", "pct": (76, 90)},
    {"id": "finish",         "en": "Finish",         "zh": "收杆", "pct": (90, 100)},
]

PHASE_ORDER = {p["id"]: i for i, p in enumerate(SWING_PHASES)}

_PHASE_IDS = [
    "address", "takeaway", "backswing", "top",
    "downswing", "impact", "follow_through", "finish",
]

# Kinematic phase detector revision (HAR / debugging).
PHASE_DETECTOR_VERSION = "3.0-agnostic"


def _finite_round_metric(x, nd: int = 4, default: float = 0.0) -> float:
    v = safe_float(x, default)
    if not math.isfinite(v):
        v = default
    return round(v, nd)


def _safe_kinematic_scalar(x, default: float = 0.0) -> float:
    """Finite scalar for debug / JSON — no NaN/inf propagation."""
    v = safe_float(x, default)
    return float(v) if math.isfinite(v) else float(default)


def _sanitize_phase_event_debug_dict(d: dict) -> None:
    """Ensure candidate debug payloads never carry NaN/inf (JSON + gate safe)."""
    if not isinstance(d, dict):
        return
    if d.get("reason") == "kinematics_unavailable":
        return
    for k, v in list(d.items()):
        if k == "signals" and isinstance(v, str):
            continue
        if isinstance(v, list) and k == "window" and len(v) == 2:
            d[k] = [int(max(0, safe_float(v[0], 0))), int(max(0, safe_float(v[1], 0)))]
            continue
        if k == "excursion_apex_idx":
            d[k] = int(max(0, safe_float(v, 0)))
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            d[k] = _finite_round_metric(v, nd=4, default=0.0)


def _sanitize_phase_detection_payload(ev: dict) -> None:
    _sanitize_phase_event_debug_dict(ev.get("top_candidate_debug") or {})
    _sanitize_phase_event_debug_dict(ev.get("impact_candidate_debug") or {})


def _composite_top_candidate_score(i: int, kin: dict, w0: int, w1: int) -> float:
    """Same weighting as detect_phase_events_agnostic top scan (for local reselection)."""
    valid = kin["valid"]
    # Match detect_phase_events_agnostic top scan: i in (w0, w1-1).
    if i <= w0 or i >= w1 - 1 or not valid[i]:
        return -1.0
    excursion = kin["excursion"]
    speed = kin["speed_s"]
    xf = kin["xf"]
    sr = kin["sr"]
    dot_rev = kin["dot_rev"]
    q = kin["q"]
    exc_win = excursion.copy()
    exc_win[~valid] = -1.0
    exc_max = float(np.max(exc_win[w0:w1])) if w1 > w0 else 1e-6
    xf_max_w = float(np.max(xf[w0:w1])) if w1 > w0 else 1e-6
    sr_max_w = float(np.max(sr[w0:w1])) if w1 > w0 else 1e-6
    spd_win = speed[w0:w1]
    spd_win = spd_win[valid[w0:w1]]
    spd_p85 = float(np.percentile(speed[valid], 85)) if np.any(valid) else 1e-6
    exc_n = float(excursion[i]) / max(exc_max, 1e-6)
    slow_n = 1.0 - min(float(speed[i]) / max(spd_p85, 1e-6), 1.0)
    xf_n = float(xf[i]) / max(xf_max_w, 1e-6)
    sr_n = float(sr[i]) / max(sr_max_w, 1e-6)
    rev_n = min(float(dot_rev[i]) * 4.0, 1.0)
    return (
        0.24 * exc_n + 0.22 * slow_n + 0.18 * xf_n + 0.18 * sr_n
        + 0.12 * rev_n + 0.06 * q[i]
    )


def _composite_impact_candidate_score(
    i: int,
    top_i: int,
    exc_apex: int,
    kin: dict,
    lo: int,
    hi: int,
    n: int,
    best_top: int,
) -> float:
    valid = kin["valid"]
    if i < lo or i >= hi or not valid[i] or i <= top_i + 1:
        return -1.0
    speed = kin["speed_s"]
    xf_d = kin["xf_d"]
    hand_hip = kin["hand_hip"]
    q = kin["q"]
    spd_m = speed[lo:hi][valid[lo:hi]]
    sp95 = max(float(np.percentile(spd_m, 95)), 1e-6) if spd_m.size > 0 else 1e-6
    neg_clip = np.clip(-xf_d[lo:hi], 0, None)
    drop_ref = max(float(np.percentile(neg_clip, 90)), 1e-6) if neg_clip.size > 0 else 1e-6
    mask_slice = valid[lo:hi]
    hh_m = hand_hip[lo:hi][mask_slice]
    hip_ref = max(float(np.percentile(hh_m, 90)), 1e-6) if hh_m.size > 0 else 1e-6
    sp_n = min(float(speed[i]) / sp95, 1.0)
    unwind = float(np.clip((-xf_d[i]) / drop_ref, 0.0, 1.0))
    zone = 1.0 - min(float(hand_hip[i]) / hip_ref, 1.0)
    t_rel = (i - exc_apex) / max(hi - exc_apex, 1)
    timing = 1.0 - min(abs(t_rel - 0.38) / 0.42, 1.0)
    robust_top = max(best_top, exc_apex)
    after_top = min(1.0, (i - robust_top) / max(n - robust_top, 1) * 2.5)
    return 0.34 * sp_n + 0.22 * unwind + 0.18 * zone + 0.14 * timing + 0.08 * after_top + 0.04 * q[i]


def _track_frame_set_for_chain(tracks: dict[str, Any] | None) -> set[int]:
    rows: list[dict] = []
    if isinstance(tracks, dict):
        for k in ("person_tracks", "club_tracks", "ball_tracks"):
            rows.extend(tracks.get(k) or [])
    return {int(r.get("frame_index", -1)) for r in rows if int(r.get("frame_index", -1)) >= 0}


def validate_follow_through_semantic_at_index(
    idx: int,
    impact_idx: int,
    kin: dict,
) -> tuple[bool, dict]:
    """Post-impact release semantics: clearly after impact, not an impact-speed clone."""
    n = len(kin.get("speed_s", []) or [])
    if n < 1 or idx < 0 or idx >= n or impact_idx < 0 or impact_idx >= n:
        return False, {"reason": "bad_index"}
    speed = np.asarray(kin["speed_s"], dtype=np.float64)
    if idx <= impact_idx:
        return False, {"reason": "not_after_impact"}
    if float(speed[idx]) > float(speed[impact_idx]) * 0.97:
        return False, {"reason": "speed_still_impact_cluster"}
    return True, {"ok": True}


def validate_finish_semantic_at_index(
    idx: int,
    impact_idx: int,
    follow_idx: int,
    kin: dict,
) -> tuple[bool, dict]:
    """Finish: speed well below impact cluster and below follow-through peak."""
    speed = np.asarray(kin.get("speed_s", []), dtype=np.float64)
    valid = np.asarray(kin.get("valid", np.ones(len(speed), dtype=bool)), dtype=bool)
    n = len(speed)
    if n < 1 or idx < 0 or idx >= n:
        return False, {"reason": "bad_index"}
    sp_valid = speed[valid]
    if sp_valid.size == 0:
        return False, {"reason": "no_valid_speed"}
    sp_p75 = float(np.percentile(sp_valid, 75))
    if impact_idx >= 0 and impact_idx < n:
        if float(speed[idx]) >= min(float(speed[impact_idx]) * 0.82, sp_p75):
            return False, {"reason": "finish_speed_not_decayed"}
    if 0 <= follow_idx < n and float(speed[idx]) > float(speed[follow_idx]) * 0.9:
        return False, {"reason": "finish_not_below_follow"}
    return True, {"ok": True}


def propose_post_impact_chain_indices(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    *,
    tracks: dict[str, Any] | None = None,
    anchor_penalty_scale: float = 1.0,
) -> dict[str, int]:
    """Rebuild downswing/impact/follow_through/finish locally from kinematics.

    Used by keyframe strip repairs when impact semantic or post-impact speed checks fail.
    Returns only indices that are valid and strictly increasing.
    """
    kin = _build_view_agnostic_kinematics(poses)
    n = len(poses)
    if kin is None or n < 8:
        return {}
    speed = np.asarray(kin.get("speed_s", np.zeros(n)), dtype=np.float64)
    q = np.asarray(kin.get("q", np.zeros(n)), dtype=np.float64)
    valid = np.asarray(kin.get("valid", np.ones(n, dtype=bool)), dtype=bool)
    if speed.shape[0] != n or q.shape[0] != n or valid.shape[0] != n:
        return {}

    top_i = max(1, min(int(phase_keyframes.get("top", max(1, n // 3))), n - 5))
    min_gap = max(2, n // 30)
    lo = max(top_i + 1, int(n * 0.44))
    hi = min(n - 2, int(n * 0.98))
    if lo >= hi:
        return {}

    sp_valid = speed[valid]
    if sp_valid.size == 0:
        return {}
    sp_med = float(np.percentile(sp_valid, 50))
    sp_p75 = float(np.percentile(sp_valid, 75))
    sp_p90 = float(np.percentile(sp_valid, 90))
    sp_peak_ref = max(sp_p90, 1e-6)

    exc_apex = int(max(0, min(n - 1, phase_keyframes.get("top", top_i))))
    best: tuple[float, dict[str, int]] | None = None
    track_frames = _track_frame_set_for_chain(tracks)

    def _pose_frame_idx(i: int) -> int:
        return int(poses[int(i)].get("frame_index", int(i)))

    # Minimum separation on the **video frame index** axis (poses can be sparse).
    min_f_gap = max(3, n // 22)

    ds_hi = min(hi - 3 * min_gap, max(lo + min_gap, top_i + max(4, n // 10)))
    for ds in range(lo, max(lo, ds_hi) + 1):
        if not valid[ds]:
            continue
        imp_lo = max(ds + min_gap, top_i + 2)
        imp_hi = min(hi - 2 * min_gap, imp_lo + max(6, n // 6))
        if imp_lo > imp_hi:
            continue
        for impact in range(imp_lo, imp_hi + 1):
            if not valid[impact]:
                continue
            imp_ok, _imp_checks = validate_impact_semantic_at_index(impact, top_i, exc_apex, kin)
            if not imp_ok:
                continue
            ft_lo = impact + min_gap
            ft_hi = min(hi - min_gap, ft_lo + max(8, n // 5))
            if ft_lo > ft_hi:
                continue
            for follow in range(ft_lo, ft_hi + 1):
                if not valid[follow]:
                    continue
                ft_ok, _ = validate_follow_through_semantic_at_index(follow, impact, kin)
                if not ft_ok:
                    continue
                fin_lo = follow + min_gap
                fin_hi = hi
                if fin_lo > fin_hi:
                    continue
                for finish in range(fin_lo, fin_hi + 1):
                    if not valid[finish]:
                        continue
                    fin_ok, _ = validate_finish_semantic_at_index(finish, impact, follow, kin)
                    if not fin_ok:
                        continue
                    fi_ds = _pose_frame_idx(ds)
                    fi_imp = _pose_frame_idx(impact)
                    fi_ft = _pose_frame_idx(follow)
                    fi_fin = _pose_frame_idx(finish)
                    if not (
                        fi_ds + min_f_gap <= fi_imp
                        and fi_imp + min_f_gap <= fi_ft
                        and fi_ft + min_f_gap <= fi_fin
                    ):
                        continue
                    # Multi-objective score.
                    distinct = (
                        (speed[impact] - speed[follow]) / max(sp_peak_ref, 1e-6)
                        + (speed[follow] - speed[finish]) / max(sp_peak_ref, 1e-6)
                        + (speed[impact] - speed[finish]) / max(sp_peak_ref, 1e-6)
                    )
                    spacing = float(impact - ds + follow - impact + finish - follow) / max(3 * min_gap, 1)
                    semantic = (
                        min(speed[impact] / max(sp_p75, 1e-6), 1.5)
                        + (1.0 - min(speed[finish] / max(sp_med, 1e-6), 1.8))
                        + float(q[ds] + q[impact] + q[follow] + q[finish]) / 4.0
                    )
                    anchor_pen = (
                        abs(ds - int(phase_keyframes.get("downswing", ds)))
                        + abs(impact - int(phase_keyframes.get("impact", impact)))
                        + abs(follow - int(phase_keyframes.get("follow_through", follow)))
                        + abs(finish - int(phase_keyframes.get("finish", finish)))
                    ) / max(n, 1)
                    fi_imp_b = _pose_frame_idx(impact)
                    track_bonus = 0.0
                    if fi_imp_b in track_frames:
                        track_bonus += 0.1
                    if _pose_frame_idx(follow) in track_frames:
                        track_bonus += 0.06
                    score = (
                        1.2 * distinct
                        + 0.7 * semantic
                        + 0.45 * spacing
                        - 0.35 * anchor_pen * float(anchor_penalty_scale)
                        + track_bonus
                    )
                    cand = {"downswing": ds, "impact": impact, "follow_through": follow, "finish": finish}
                    if best is None or score > best[0]:
                        best = (float(score), cand)

    if best is None:
        return {}
    return {k: int(v) for k, v in best[1].items()}


def propose_quality_spacing_post_top_chain(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    *,
    fps: float = 30.0,
    spacing_boost: float = 1.0,
) -> dict[str, int]:
    """Jointly widen post-top pose indices when the strip is ordered/semantic-clean but too tight visually.

    Escapes NEAR_DUPLICATE / TIME_TOO_CLOSE strict failures by searching a chain with **larger**
    pose and raw-frame separation, while keeping top fixed and impact in a modest window around
    the current impact so strike semantics stay stable.

    Unlike ``propose_post_impact_chain_indices``, scoring does **not** heavily penalize deviation
    from the previous (clustered) indices, which is why that helper often returns ``material_change=False``
    on production strips that are already kinematically plausible but visually colliding.
    """
    kin = _build_view_agnostic_kinematics(poses)
    n = len(poses)
    if kin is None or n < 16:
        return {}

    speed = np.asarray(kin.get("speed_s", np.zeros(n)), dtype=np.float64)
    q = np.asarray(kin.get("q", np.zeros(n)), dtype=np.float64)
    valid = np.asarray(kin.get("valid", np.ones(n, dtype=bool)), dtype=bool)
    if speed.shape[0] != n or valid.shape[0] != n:
        return {}

    top_i = max(1, min(int(phase_keyframes.get("top", n // 3)), n - 8))
    imp0 = int(phase_keyframes.get("impact", min(top_i + 5, n - 4)))
    imp0 = max(top_i + 3, min(imp0, n - 5))

    ev = detect_phase_events_agnostic(poses)
    exc_apex = int(ev.get("excursion_apex_idx", top_i)) if ev else top_i

    boost = max(1.0, float(spacing_boost))
    min_pose = max(3, int(round(float(fps) * 0.055 * boost)), n // 22)
    min_fi = max(4, int(round(float(fps) * 0.11 * boost)), n // 16)

    def _pose_frame_idx(i: int) -> int:
        return int(poses[int(i)].get("frame_index", int(i)))

    # Wider than classic chain so impact can slide forward and make room for top→downswing separation.
    imp_win = max(6, min(16, n // 7))
    lo_scan = max(top_i + min_pose + 1, imp0 - imp_win)
    # Room for follow + finish after impact (each ≥ min_pose apart).
    impact_hi_cap = n - 2 * min_pose - 2
    hi_scan = min(impact_hi_cap, imp0 + imp_win, n - 2)
    if lo_scan > hi_scan:
        return {}

    sp_valid = speed[valid]
    sp_med = float(np.percentile(sp_valid, 50)) if sp_valid.size else 1e-6
    sp_p75 = float(np.percentile(sp_valid, 75)) if sp_valid.size else 1e-6

    best: tuple[float, dict[str, int]] | None = None

    for impact in range(lo_scan, hi_scan + 1):
        if not valid[impact]:
            continue
        imp_ok, _ = validate_impact_semantic_at_index(impact, top_i, exc_apex, kin)
        if not imp_ok:
            continue
        # Prefer downswing not glued to top when impact leaves room; else keep legal window.
        min_after_top = max(2, min(5, n // 18))
        ds_hi = impact - min_pose
        ds_lo = top_i + min_after_top if (top_i + min_after_top) <= ds_hi else top_i + 1
        if ds_lo > ds_hi:
            continue
        for ds in range(ds_lo, ds_hi + 1):
            if not valid[ds]:
                continue
            if float(speed[ds]) >= float(speed[impact]) * 0.995:
                continue
            fi_ds = _pose_frame_idx(ds)
            fi_imp = _pose_frame_idx(impact)
            if fi_ds + min_fi > fi_imp:
                continue
            ft_lo = impact + min_pose
            ft_hi = min(n - min_pose - 2, ft_lo + max(10, n // 4))
            for follow in range(ft_lo, ft_hi + 1):
                if not valid[follow]:
                    continue
                ft_ok, _ = validate_follow_through_semantic_at_index(follow, impact, kin)
                if not ft_ok:
                    continue
                fi_ft = _pose_frame_idx(follow)
                if fi_imp + min_fi > fi_ft:
                    continue
                fin_lo = follow + min_pose
                fin_hi = n - 1
                for finish in range(fin_lo, fin_hi + 1):
                    if not valid[finish]:
                        continue
                    fin_ok, _ = validate_finish_semantic_at_index(finish, follow, impact, kin)
                    if not fin_ok:
                        continue
                    fi_fin = _pose_frame_idx(finish)
                    if fi_ft + min_fi > fi_fin:
                        continue
                    pg = float((impact - ds) + (follow - impact) + (finish - follow))
                    fg = float((fi_imp - fi_ds) + (fi_ft - fi_imp) + (fi_fin - fi_ft))
                    sem = (
                        min(float(speed[impact]) / max(sp_p75, 1e-6), 1.4)
                        + (1.0 - min(float(speed[finish]) / max(sp_med, 1e-6), 1.6))
                        + float(q[ds] + q[impact] + q[follow] + q[finish]) * 0.22
                    )
                    anchor = (
                        abs(impact - imp0)
                        + abs(ds - int(phase_keyframes.get("downswing", ds))) * 0.45
                    ) / max(n, 1)
                    top_pull = float(ds - top_i) + 0.42 * float(impact - ds)
                    score = 1.55 * pg + 0.018 * fg + 0.65 * sem + 0.88 * top_pull - 0.12 * anchor
                    cand = {"downswing": ds, "impact": impact, "follow_through": follow, "finish": finish}
                    if best is None or score > best[0]:
                        best = (float(score), cand)

    if best is None:
        return {}
    return {k: int(v) for k, v in best[1].items()}


# Minimum required visible joints for a pose to be considered a full-body golf pose.
# Camera cuts (face close-ups, crowd shots) will fail this check.
_GOLF_BODY_JOINTS = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_wrist", "right_wrist",
]
_MIN_BODY_VISIBILITY = 0.3


def _is_valid_golf_pose(pose: dict) -> bool:
    """Check if a pose has enough visible body joints for swing analysis.

    Returns False for close-up face shots, crowd shots, or other frames
    where MediaPipe hallucinates body positions on partial views.
    """
    joints = pose.get("joints", [])
    if not joints:
        return False

    visible_count = 0
    for j in joints:
        if j["name"] in _GOLF_BODY_JOINTS:
            if j.get("visibility", 0) >= _MIN_BODY_VISIBILITY:
                visible_count += 1

    return visible_count >= 4


def _filter_golf_poses(poses: list[dict]) -> tuple[list[dict], list[int]]:
    """Filter poses to only include valid full-body golf frames.

    Returns (filtered_poses, original_indices) so we can map back.
    """
    filtered = []
    indices = []
    for i, pose in enumerate(poses):
        if _is_valid_golf_pose(pose):
            filtered.append(pose)
            indices.append(i)
    return filtered, indices


def _extract_trajectory(poses, joint_name, axis="y"):
    """Extract normalized trajectory for a given joint name."""
    vals = []
    for pose in poses:
        joints = pose.get("joints", [])
        jt = next((j for j in joints if j["name"] == joint_name), None)
        if jt:
            ny = jt.get("normalized", {}).get(axis, 0.5)
            vals.append(ny)
        else:
            vals.append(vals[-1] if vals else 0.5)
    return np.array(vals)


def _smooth_signal(arr, window=3):
    """Simple moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


_T_AXIS_EPS = 1e-4


def _build_time_axis_for_gradient(
    poses: list[dict],
    default_fps: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Strictly increasing time coordinates for ``np.gradient(f, t)``.

    On missing / non-finite / duplicate / non-monotonic timestamps, rebuild from
    ``frame_index`` and ``default_fps``. Any epsilon bumps to break ties are
    counted in ``dt_fixed_count``.
    """
    n = len(poses)
    inv_fps = 1.0 / max(float(default_fps), 1e-6)
    debug: dict = {
        "dt_min": 0.0,
        "dt_zero_count": 0,
        "dt_fixed_count": 0,
        "dt_axis_invalid": False,
    }
    if n == 0:
        return np.array([], dtype=np.float64), debug
    if n == 1:
        t0 = 0.0
        if poses[0].get("timestamp") is not None:
            try:
                t0 = float(poses[0]["timestamp"])
            except (TypeError, ValueError):
                t0 = 0.0
        if not math.isfinite(t0):
            t0 = 0.0
        return np.array([t0], dtype=np.float64), debug

    raw: list[float] = []
    for p in poses:
        ts = p.get("timestamp")
        if ts is None:
            raw.append(float("nan"))
        else:
            try:
                raw.append(float(ts))
            except (TypeError, ValueError):
                raw.append(float("nan"))

    bad_ts = False
    prev_v: float | None = None
    for v in raw:
        if not math.isfinite(v):
            bad_ts = True
            break
        if prev_v is not None and v <= prev_v:
            bad_ts = True
            break
        prev_v = v

    if bad_ts:
        t = np.empty(n, dtype=np.float64)
        for i, p in enumerate(poses):
            fi = p.get("frame_index")
            try:
                j = int(fi) if fi is not None else i
            except (TypeError, ValueError):
                j = i
            t[i] = j * inv_fps
    else:
        t = np.asarray(raw, dtype=np.float64)

    dt_zero_count = 0
    dt_fixed_count = 0
    for i in range(1, n):
        if t[i] <= t[i - 1]:
            dt_zero_count += 1
            t[i] = t[i - 1] + _T_AXIS_EPS
            dt_fixed_count += 1

    ddiff = np.diff(t)
    dt_min = float(np.min(ddiff)) if ddiff.size else 0.0
    debug["dt_min"] = round(dt_min, 6)
    debug["dt_zero_count"] = int(dt_zero_count)
    debug["dt_fixed_count"] = int(dt_fixed_count)
    debug["dt_axis_invalid"] = bool(bad_ts or dt_zero_count > 0)
    return t, debug


def safe_gradient(signal: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, bool]:
    """``np.gradient`` with finite output. ``had_non_finite`` is True if any raw derivative was non-finite."""
    had_non_finite = False
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    tt = np.asarray(t, dtype=np.float64)
    n = int(sig.size)
    if n != tt.size:
        had_non_finite = True
        return np.zeros(n, dtype=np.float64), had_non_finite
    if n < 2:
        return np.zeros(n, dtype=np.float64), had_non_finite
    if n < 3:
        dt = float(tt[1] - tt[0])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if not math.isfinite(dt) or abs(dt) < 1e-15:
                had_non_finite = True
                g = 0.0
            else:
                g = float((sig[1] - sig[0]) / dt)
        if not math.isfinite(g):
            had_non_finite = True
            g = 0.0
        return np.array([g, g], dtype=np.float64), had_non_finite
    if np.any(np.diff(tt) <= 0):
        had_non_finite = True
        tt = tt.astype(np.float64).copy()
        for i in range(1, len(tt)):
            if tt[i] <= tt[i - 1]:
                tt[i] = tt[i - 1] + _T_AXIS_EPS
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = np.gradient(sig, tt)
    raw = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        had_non_finite = True
    out = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return out, had_non_finite


def strictly_increasing_t(
    poses: list[dict],
    fps_hint: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Public helper: monotonic sample times + compact meta (dt_min, dt_fixed_count, dt_zero_count)."""
    t, full = _build_time_axis_for_gradient(poses, float(fps_hint))
    meta = {
        "dt_min": float(full.get("dt_min", 0.0)),
        "dt_fixed_count": int(full.get("dt_fixed_count", 0)),
        "dt_zero_count": int(full.get("dt_zero_count", 0)),
    }
    return t, meta


def _build_dt_array(poses: list[dict], default_fps: float = 30.0) -> np.ndarray:
    """Backward-compatible: monotonic sample times (same length as ``poses``) for ``np.gradient``."""
    t, _ = _build_time_axis_for_gradient(poses, default_fps)
    return t


def _get_joint(pose: dict, name: str) -> Optional[dict]:
    for j in pose.get("joints", []):
        if j.get("name") == name:
            return j
    return None


def _joint_norm(pose: dict, name: str, axis: str, default: float = 0.5) -> float:
    j = _get_joint(pose, name)
    if not j:
        return default
    return float(j.get("normalized", {}).get(axis, default))


def _pose_quality_score(pose: dict) -> float:
    """Pose quality score for keyframe ranking (0~1)."""
    required = [
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_wrist", "right_wrist", "left_knee", "right_knee",
    ]
    vis_vals = []
    in_frame = 0
    for name in required:
        j = _get_joint(pose, name)
        if not j:
            vis_vals.append(0.0)
            continue
        v = float(j.get("visibility", 0.0))
        vis_vals.append(np.clip(v, 0.0, 1.0))
        nx = float(j.get("normalized", {}).get("x", 0.5))
        ny = float(j.get("normalized", {}).get("y", 0.5))
        if 0.02 <= nx <= 0.98 and 0.02 <= ny <= 0.98:
            in_frame += 1

    mean_vis = float(np.mean(vis_vals)) if vis_vals else 0.0
    completeness = in_frame / max(len(required), 1)
    key_vis = []
    for key_name in ("left_wrist", "right_wrist", "left_hip", "right_hip"):
        j = _get_joint(pose, key_name)
        key_vis.append(float(j.get("visibility", 0.0)) if j else 0.0)
    key_joint_vis = float(np.mean(key_vis)) if key_vis else 0.0

    return float(np.clip(0.45 * mean_vis + 0.35 * completeness + 0.20 * key_joint_vis, 0.0, 1.0))


def _joints_for_detection(pose: dict) -> list:
    d = pose.get("detection") or {}
    return d.get("joints") or pose.get("joints") or []


def _angles_for_detection(pose: dict) -> dict:
    d = pose.get("detection") or {}
    ang = d.get("angles")
    return ang if isinstance(ang, dict) else (pose.get("angles") or {})


def _norm_xy_from_joints(joints: list, name: str) -> tuple[float, float]:
    jt = next((j for j in joints if j.get("name") == name), None)
    if not jt:
        return 0.5, 0.5
    nx = float(jt.get("normalized", {}).get("x", 0.5))
    ny = float(jt.get("normalized", {}).get("y", 0.5))
    return nx, ny


def _build_view_agnostic_kinematics(poses: list[dict]) -> dict | None:
    """2D image-plane kinematics from detection (pre-smooth) joints — no single-axis 'height' assumption."""
    n = len(poses)
    if n < 6:
        return None
    t, time_axis_debug = _build_time_axis_for_gradient(poses)
    kinematic_fail_codes: list[str] = []
    had_nf_grad = False
    cx = np.zeros(n)
    cy = np.zeros(n)
    lwx = np.zeros(n)
    lwy = np.zeros(n)
    rwx = np.zeros(n)
    rwy = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    q = np.zeros(n)
    for i, p in enumerate(poses):
        joints = _joints_for_detection(p)
        lx, ly = _norm_xy_from_joints(joints, "left_wrist")
        rx, ry = _norm_xy_from_joints(joints, "right_wrist")
        lwx[i], lwy[i] = lx, ly
        rwx[i], rwy[i] = rx, ry
        cx[i] = (lx + rx) / 2.0
        cy[i] = (ly + ry) / 2.0
        valid[i] = _is_valid_golf_pose(p)
        q[i] = _pose_quality_score(p)

    setup = min(max(n // 6, 3), 14)
    bx = float(np.mean(cx[:setup]))
    by = float(np.mean(cy[:setup]))
    exc_c = np.hypot(cx - bx, cy - by)
    exc_l = np.hypot(lwx - bx, lwy - by)
    exc_r = np.hypot(rwx - bx, rwy - by)
    excursion = np.maximum(np.maximum(exc_c, exc_l), exc_r)

    vx, hvx = safe_gradient(cx, t)
    had_nf_grad = had_nf_grad or hvx
    vy, hvy = safe_gradient(cy, t)
    had_nf_grad = had_nf_grad or hvy
    vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
    vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.nan_to_num(np.hypot(vx, vy), nan=0.0, posinf=0.0, neginf=0.0)
    speed_s = _smooth_signal(speed, 3)

    xf = np.array([float(_angles_for_detection(p).get("x_factor", 0.0)) for p in poses])
    sr = np.array([abs(float(_angles_for_detection(p).get("shoulder_rotation", 0.0))) for p in poses])
    xf = _smooth_signal(xf, 3)
    sr = _smooth_signal(sr, 3)
    xf_d, hxfd = safe_gradient(xf, t)
    had_nf_grad = had_nf_grad or hxfd
    xf_d = np.nan_to_num(xf_d, nan=0.0, posinf=0.0, neginf=0.0)

    hy = (_extract_trajectory(poses, "left_hip", "y") + _extract_trajectory(poses, "right_hip", "y")) / 2.0
    wy = (lwy + rwy) / 2.0
    hand_hip = np.hypot(cx - (_extract_trajectory(poses, "left_hip", "x") + _extract_trajectory(poses, "right_hip", "x")) / 2.0, wy - hy)

    dot_rev = np.zeros(n)
    for i in range(1, n - 1):
        a = np.array([vx[i - 1], vy[i - 1]], dtype=np.float64)
        b = np.array([vx[i + 1], vy[i + 1]], dtype=np.float64)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        denom = max(na * nb, 1e-12)
        dot_rev[i] = max(0.0, float(-np.dot(a, b) / denom))

    if time_axis_debug.get("dt_axis_invalid") and "DT_AXIS_INVALID" not in kinematic_fail_codes:
        kinematic_fail_codes.append("DT_AXIS_INVALID")
    if had_nf_grad and "NON_FINITE_KINEMATICS" not in kinematic_fail_codes:
        kinematic_fail_codes.append("NON_FINITE_KINEMATICS")

    return {
        "n": n,
        "dt": t,
        "time_axis_debug": time_axis_debug,
        "kinematic_fail_codes": kinematic_fail_codes,
        "valid": valid,
        "excursion": excursion,
        "speed": speed,
        "speed_s": speed_s,
        "vx": vx,
        "vy": vy,
        "xf": xf,
        "sr": sr,
        "xf_d": xf_d,
        "wy": wy,
        "hy": hy,
        "hand_hip": hand_hip,
        "dot_rev": dot_rev,
        "q": q,
        "setup": setup,
    }


def _snap_full_idx_to_valid(full_i: int, valid_indices: list[int]) -> int:
    if not valid_indices:
        return 0
    best_k = 0
    best_d = abs(valid_indices[0] - full_i)
    for k, vi in enumerate(valid_indices):
        d = abs(vi - full_i)
        if d < best_d:
            best_d = d
            best_k = k
    return int(best_k)


def detect_phase_events_agnostic(poses: list[dict]) -> dict:
    """Find top / impact in full pose index space using view-agnostic 2D kinematics + rotation peaks."""
    kin = _build_view_agnostic_kinematics(poses)
    if kin is None:
        return {
            "top_pose_idx": max(0, len(poses) // 2 - 1),
            "impact_pose_idx": min(len(poses) - 1, max(1, int(len(poses) * 0.72))),
            "excursion_apex_idx": max(1, len(poses) // 3),
            "phase_detector_confidence": 0.15,
            "top_semantic_ok": False,
            "top_semantic_detail": {},
            "impact_semantic_ok": False,
            "impact_semantic_detail": {},
            "top_candidate_debug": {"reason": "kinematics_unavailable"},
            "impact_candidate_debug": {"reason": "kinematics_unavailable"},
            "kinematic_fail_codes": [],
            "time_axis_debug": {},
        }
    n = kin["n"]
    valid = kin["valid"]
    excursion = kin["excursion"]
    speed = kin["speed_s"]
    xf = kin["xf"]
    sr = kin["sr"]
    xf_d = kin["xf_d"]
    dot_rev = kin["dot_rev"]
    hand_hip = kin["hand_hip"]
    q = kin["q"]
    setup = kin["setup"]

    w0 = min(max(setup, 1), n - 4)
    w1 = max(w0 + 2, int(n * 0.78))
    w1 = min(w1, n - 2)

    exc_win = excursion.copy()
    exc_win[~valid] = -1.0
    exc_max = float(np.max(exc_win[w0:w1])) if w1 > w0 else 1e-6
    xf_max_w = float(np.max(xf[w0:w1])) if w1 > w0 else 1e-6
    sr_max_w = float(np.max(sr[w0:w1])) if w1 > w0 else 1e-6
    spd_win = speed[w0:w1]
    spd_win = spd_win[valid[w0:w1]]
    spd_p40 = float(np.percentile(spd_win, 40)) if len(spd_win) > 0 else 1e-6
    spd_p85 = float(np.percentile(speed[valid], 85)) if np.any(valid) else 1e-6

    best_i = w0
    best_sc = -1.0
    scores_by_i: dict[int, float] = {}
    for i in range(w0 + 1, w1 - 1):
        if not valid[i]:
            continue
        exc_n = float(excursion[i]) / max(exc_max, 1e-6)
        slow_n = 1.0 - min(float(speed[i]) / max(spd_p85, 1e-6), 1.0)
        xf_n = float(xf[i]) / max(xf_max_w, 1e-6)
        sr_n = float(sr[i]) / max(sr_max_w, 1e-6)
        rev_n = min(float(dot_rev[i]) * 4.0, 1.0)
        score = (
            0.24 * exc_n + 0.22 * slow_n + 0.18 * xf_n + 0.18 * sr_n
            + 0.12 * rev_n + 0.06 * q[i]
        )
        scores_by_i[i] = score
        if score > best_sc:
            best_sc = score
            best_i = i

    exc_flat = excursion.copy()
    exc_flat[~valid] = -1.0
    exc_apex = w0 + int(np.argmax(exc_flat[w0:min(int(n * 0.55), w1)]))

    top_debug = {
        "window": [w0, w1],
        "score": _finite_round_metric(best_sc, 4),
        "excursion_at_top": _finite_round_metric(excursion[best_i], 4),
        "excursion_max_window": _finite_round_metric(exc_max, 4),
        "speed_at_top": _finite_round_metric(_safe_kinematic_scalar(speed[best_i], 0.0), 4),
        "xf_at_top": _finite_round_metric(xf[best_i], 2),
        "sr_at_top": _finite_round_metric(sr[best_i], 2),
        "reversal_metric": _finite_round_metric(dot_rev[best_i], 4),
        "signals": "excursion+speed_trough+xf_peak+sr_peak+2d_reversal+quality",
    }

    top_sem_ok, top_sem_detail = validate_top_semantic_at_index(best_i, kin)

    lo = max(exc_apex + 2, int(n * 0.42), best_i + 2)
    hi = min(n - 2, int(n * 0.93))
    if lo >= hi:
        lo, hi = max(1, n // 2), min(n - 2, n - 1)

    mask_slice = valid[lo:hi]
    impact_idx = lo + max(1, (hi - lo) // 3)
    best_is = -1.0
    neg_clip = np.clip(-xf_d[lo:hi], 0, None)
    drop_ref = max(float(np.percentile(neg_clip, 90)), 1e-6) if neg_clip.size > 0 else 1e-6
    hh_m = hand_hip[lo:hi][mask_slice]
    hip_ref = max(float(np.percentile(hh_m, 90)), 1e-6) if hh_m.size > 0 else 1e-6
    spd_m = speed[lo:hi][mask_slice]
    sp95_loop = max(float(np.percentile(spd_m, 95)), 1e-6) if spd_m.size > 0 else 1e-6

    for i in range(lo, hi):
        if not valid[i]:
            continue
        sp_n = min(float(speed[i]) / sp95_loop, 1.0)
        unwind = float(np.clip((-xf_d[i]) / drop_ref, 0.0, 1.0))
        zone = 1.0 - min(float(hand_hip[i]) / hip_ref, 1.0)
        denom_te = max(float(hi - exc_apex), 1.0)
        t_rel = float(i - exc_apex) / denom_te
        timing = 1.0 - min(abs(t_rel - 0.38) / 0.42, 1.0)
        robust_top = max(best_i, exc_apex)
        denom_rt = max(float(n - robust_top), 1.0)
        after_top = min(1.0, float(i - robust_top) / denom_rt * 2.5)
        isc = 0.34 * sp_n + 0.22 * unwind + 0.18 * zone + 0.14 * timing + 0.08 * after_top + 0.04 * q[i]
        if isc > best_is:
            best_is = isc
            impact_idx = i

    impact_debug = {
        "window": [lo, hi],
        "excursion_apex_idx": int(exc_apex),
        "score": _finite_round_metric(best_is, 4),
        "speed_at_impact": _finite_round_metric(_safe_kinematic_scalar(speed[impact_idx], 0.0), 4),
        "xf_deriv_at_impact": _finite_round_metric(_safe_kinematic_scalar(xf_d[impact_idx], 0.0), 4),
        "hand_hip_dist": _finite_round_metric(hand_hip[impact_idx], 4),
        "signals": "speed_peak+unwind+hand_zone+timing_vs_excursion_apex+post_top",
    }

    imp_sem_ok, imp_sem_detail = validate_impact_semantic_at_index(impact_idx, best_i, exc_apex, kin)

    kcodes = list(kin.get("kinematic_fail_codes") or [])
    if kcodes:
        top_sem_ok = False
        imp_sem_ok = False
        fail_code = (
            "NON_FINITE_KINEMATICS"
            if "NON_FINITE_KINEMATICS" in kcodes
            else ("DT_AXIS_INVALID" if "DT_AXIS_INVALID" in kcodes else str(kcodes[0]))
        )
        kin_merge = {"kinematic_fail_codes": kcodes, "fail_code": fail_code}
        top_sem_detail = {**(top_sem_detail or {}), **kin_merge}
        imp_sem_detail = {**(imp_sem_detail or {}), **kin_merge}

    conf = 0.35 + 0.20 * (1.0 if top_sem_ok else 0.0) + 0.25 * (1.0 if imp_sem_ok else 0.0)
    conf += 0.10 * min(best_sc, 1.0) + 0.10 * min(best_is, 1.0)
    conf = float(np.clip(conf, 0.0, 0.98))

    out_ev = {
        "top_pose_idx": int(best_i),
        "impact_pose_idx": int(impact_idx),
        "excursion_apex_idx": int(exc_apex),
        "phase_detector_confidence": conf,
        "top_semantic_ok": top_sem_ok,
        "top_semantic_detail": top_sem_detail,
        "impact_semantic_ok": imp_sem_ok,
        "impact_semantic_detail": imp_sem_detail,
        "top_candidate_debug": top_debug,
        "impact_candidate_debug": impact_debug,
        "kinematic_fail_codes": list(kin.get("kinematic_fail_codes") or []),
        "time_axis_debug": dict(kin.get("time_axis_debug") or {}),
    }
    _sanitize_phase_detection_payload(out_ev)
    return out_ev


def compute_chain_kinematic_markers(poses: list[dict]) -> dict:
    """Compact kinematic markers for multi-backend phase windows / chain solving."""
    kin = _build_view_agnostic_kinematics(poses)
    if kin is None:
        return {"ok": False, "reason": "kinematics_unavailable"}
    speed = np.asarray(kin.get("speed_s", []), dtype=np.float64)
    valid = np.asarray(kin.get("valid", []), dtype=bool)
    xf_d = np.asarray(kin.get("xf_d", []), dtype=np.float64)
    if speed.size == 0:
        return {"ok": False, "reason": "empty_speed"}
    n = len(speed)
    lo = max(1, int(n * 0.45))
    hi = min(n - 2, int(n * 0.93))
    impact_seed = lo if lo < hi else max(1, n // 2)
    if lo < hi:
        cand = [i for i in range(lo, hi + 1) if i < len(valid) and bool(valid[i])]
        if cand:
            impact_seed = max(cand, key=lambda i: float(speed[i]) + max(0.0, -float(xf_d[i])))
    return {
        "ok": True,
        "impact_seed": int(impact_seed),
        "speed_p50": float(np.percentile(speed[valid], 50)) if np.any(valid) else float(np.percentile(speed, 50)),
        "speed_p80": float(np.percentile(speed[valid], 80)) if np.any(valid) else float(np.percentile(speed, 80)),
    }


def validate_top_semantic_at_index(i: int, kin: dict) -> tuple[bool, dict]:
    """Require agreement between excursion apex neighborhood, slow hands, and rotation peaks (not wrist_y alone)."""
    kfc = list(kin.get("kinematic_fail_codes") or [])
    if kfc:
        fc = (
            "NON_FINITE_KINEMATICS"
            if "NON_FINITE_KINEMATICS" in kfc
            else ("DT_AXIS_INVALID" if "DT_AXIS_INVALID" in kfc else str(kfc[0]))
        )
        return False, {"fail_code": fc, "kinematic_fail_codes": kfc}
    n = kin["n"]
    valid = kin["valid"]
    excursion = kin["excursion"]
    speed = kin["speed_s"]
    xf = kin["xf"]
    sr = kin["sr"]
    dot_rev = kin["dot_rev"]
    setup = kin["setup"]
    if i <= setup or i >= n - 2 or not valid[i]:
        return False, {"fail": "index_or_valid"}
    w0 = min(max(setup, 1), n - 4)
    w1 = max(w0 + 2, int(n * 0.78))
    w1 = min(w1, n - 2)
    sub_e = excursion[w0:w1]
    sub_v = valid[w0:w1]
    exc_max = float(np.max(sub_e[sub_v])) if np.any(sub_v) else 1e-6
    xf_max = float(np.max(xf[w0:w1]))
    sr_max = float(np.max(sr[w0:w1]))
    spd_slice = speed[w0:w1][valid[w0:w1]]
    spd_med = float(np.percentile(spd_slice, 50)) if len(spd_slice) > 0 else 1.0

    checks = {
        "excursion_strong": float(excursion[i]) >= 0.66 * max(exc_max, 1e-6),
        "hands_slowish": float(speed[i]) <= max(spd_med * 1.35, 1e-6),
        "xf_high": float(xf[i]) >= 0.62 * max(xf_max, 1e-6),
        "sr_high": float(sr[i]) >= 0.58 * max(sr_max, 1e-6),
        "has_reversal": float(dot_rev[i]) >= 0.04,
    }
    passed = sum(1 for v in checks.values() if v) >= 3
    return passed, checks


def validate_impact_semantic_at_index(i: int, top_i: int, exc_apex: int, kin: dict) -> tuple[bool, dict]:
    kfc = list(kin.get("kinematic_fail_codes") or [])
    if kfc:
        fc = (
            "NON_FINITE_KINEMATICS"
            if "NON_FINITE_KINEMATICS" in kfc
            else ("DT_AXIS_INVALID" if "DT_AXIS_INVALID" in kfc else str(kfc[0]))
        )
        return False, {"fail_code": fc, "kinematic_fail_codes": kfc}
    n = kin["n"]
    valid = kin["valid"]
    if i <= top_i + 1 or i >= n - 1 or not valid[i]:
        return False, {"fail": "order_or_valid"}
    speed = kin["speed_s"]
    xf_d = kin["xf_d"]
    hand_hip = kin["hand_hip"]
    lo = max(exc_apex + 1, int(n * 0.40))
    hi = min(n - 1, int(n * 0.94))
    post = speed[lo:hi][valid[lo:hi]]
    sp_ref = float(np.percentile(post, 75)) if len(post) > 0 else 0.0
    checks = {
        "after_top": i > top_i + 1,
        "after_excursion_apex": i > exc_apex + 1,
        "speed_high": float(speed[i]) >= max(sp_ref * 0.82, 1e-6),
        "unwinding": float(-xf_d[i]) > 1e-6,
        "downswing_late_window": i >= max(top_i + 2, int((top_i * 0.35) + (exc_apex * 0.15) + (n * 0.30))),
        "strike_zone_reasonable": float(hand_hip[i]) <= float(
            np.percentile(
                np.nan_to_num(hand_hip[lo:hi][valid[lo:hi]], nan=0.0, posinf=0.0, neginf=0.0),
                92,
            )
        ) if np.any(valid[lo:hi]) else True,
    }
    passed = (
        checks["after_top"]
        and checks["after_excursion_apex"]
        and checks["downswing_late_window"]
        and checks["speed_high"]
        and checks["unwinding"]
        and checks["strike_zone_reasonable"]
    )
    return passed, checks


def validate_follow_through_semantic_at_index(i: int, impact_i: int, kin: dict) -> tuple[bool, dict]:
    speed = np.asarray(kin.get("speed_s", []), dtype=np.float64)
    valid = np.asarray(kin.get("valid", []), dtype=bool)
    n = int(kin.get("n", len(speed)))
    if n < 4 or i <= impact_i or i >= n or i >= len(speed):
        return False, {"fail": "index_or_order"}
    if i >= len(valid) or not bool(valid[i]):
        return False, {"fail": "invalid_pose"}
    checks = {
        "after_impact": i > impact_i + 1,
        "speed_below_impact_peak": float(speed[i]) < float(speed[impact_i]) * 0.98,
        "speed_still_active": float(speed[i]) >= float(np.percentile(speed[valid], 45)) if np.any(valid) else True,
    }
    return bool(all(checks.values())), checks


def validate_finish_semantic_at_index(i: int, follow_i: int, impact_i: int, kin: dict) -> tuple[bool, dict]:
    speed = np.asarray(kin.get("speed_s", []), dtype=np.float64)
    valid = np.asarray(kin.get("valid", []), dtype=bool)
    n = int(kin.get("n", len(speed)))
    if n < 4 or i <= follow_i or i >= n or i >= len(speed):
        return False, {"fail": "index_or_order"}
    if i >= len(valid) or not bool(valid[i]):
        return False, {"fail": "invalid_pose"}
    checks = {
        "after_follow": i > follow_i + 1,
        "after_impact": i > impact_i + 2,
        "speed_decay_vs_impact": float(speed[i]) < float(speed[impact_i]) * 0.82,
        "speed_decay_vs_follow": float(speed[i]) < float(speed[follow_i]) * 0.9,
    }
    return bool(all(checks.values())), checks


def refine_phase_keyframes_top_impact(poses: list[dict], phase_keyframes: dict[str, int]) -> None:
    """Snap ``top`` / ``impact`` pose indices toward kinematic events with local semantic search.

    Fixes bucket-midpoint drift and overly aggressive monotonic spacing that pushed
    teaching key moments away from true hand-speed / excursion events (especially
    side-on and screen-capture swings).
    """
    n = len(poses)
    if n < 12:
        return
    kin = _build_view_agnostic_kinematics(poses)
    if kin is None:
        return
    ev = detect_phase_events_agnostic(poses)
    top_ev = int(ev["top_pose_idx"])
    imp_ev = int(ev["impact_pose_idx"])
    exc = int(ev["excursion_apex_idx"])
    w = max(7, min(20, n // 7))

    best_top = top_ev
    best_sc = -1.0
    for i in range(max(1, top_ev - w), min(n - 2, top_ev + w + 1)):
        if not bool(kin["valid"][i]):
            continue
        ok, _ = validate_top_semantic_at_index(i, kin)
        exq = float(kin["excursion"][i]) * float(kin["q"][i])
        sc = exq + (0.42 if ok else 0.0)
        if sc > best_sc:
            best_sc = sc
            best_top = i

    phase_keyframes["top"] = int(min(max(best_top, 0), n - 1))

    top_i = int(phase_keyframes["top"])
    best_imp = imp_ev
    best_is = -1.0
    lo = max(top_i + 2, imp_ev - w)
    hi = min(n - 2, imp_ev + w)
    for i in range(lo, hi + 1):
        if not bool(kin["valid"][i]):
            continue
        ok, _ = validate_impact_semantic_at_index(i, top_i, exc, kin)
        sp = float(kin["speed_s"][i]) * (0.45 + 0.55 * float(kin["q"][i]))
        if ok:
            sp += 0.55
        if sp > best_is:
            best_is = sp
            best_imp = i
    if best_is < 0:
        best_imp = min(max(imp_ev, top_i + 2), n - 1)
    phase_keyframes["impact"] = int(min(max(best_imp, top_i + 2), n - 1))


def build_semantic_phase_report(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    phase_validation: dict | None,
    keyframes: list[dict] | None = None,
    final_keyframe_validation: dict | None = None,
) -> dict:
    """HAR-facing bundle: detector output vs chosen keyframes + final semantic OK."""
    ev = detect_phase_events_agnostic(poses)
    _sanitize_phase_detection_payload(ev)
    n = len(poses)
    top_ev = int(ev["top_pose_idx"])
    imp_ev = int(ev["impact_pose_idx"])
    align_tol = max(6, n // 14)
    exc_apex_ev = int(ev["excursion_apex_idx"])

    pk_top0 = int(phase_keyframes.get("top", -1))
    pk_imp0 = int(phase_keyframes.get("impact", -1))
    top_abs_err_before = abs(pk_top0 - top_ev) if pk_top0 >= 0 and top_ev >= 0 else 999999
    impact_abs_err_before = abs(pk_imp0 - imp_ev) if pk_imp0 >= 0 and imp_ev >= 0 else 999999

    kin = _build_view_agnostic_kinematics(poses)
    phase_reselection_attempted = False
    top_reselected = False
    impact_reselected = False

    if kin is not None and n >= 8:
        setup = kin["setup"]
        w0 = min(max(setup, 1), n - 4)
        w1 = max(w0 + 2, int(n * 0.78))
        w1 = min(w1, n - 2)
        lo_imp = max(exc_apex_ev + 2, int(n * 0.42), top_ev + 2)
        hi_imp = min(n - 2, int(n * 0.93))
        if lo_imp >= hi_imp:
            lo_imp, hi_imp = max(1, n // 2), min(n - 2, n - 1)

        top_k_cur = int(phase_keyframes.get("top", -1))
        if top_k_cur >= 0 and top_abs_err_before > align_tol:
            phase_reselection_attempted = True
            lo_t, hi_t = max(1, top_ev - 8), min(n - 2, top_ev + 8)
            best_pick: int | None = None
            best_key: tuple | None = None
            for i in range(lo_t, hi_t + 1):
                sc = _composite_top_candidate_score(i, kin, w0, w1)
                if sc < 0:
                    continue
                ok, _ = validate_top_semantic_at_index(i, kin)
                key = (1 if ok else 0, sc, -abs(i - top_ev))
                if best_key is None or key > best_key:
                    best_key = key
                    best_pick = i
            if best_pick is not None and best_pick != top_k_cur:
                phase_keyframes["top"] = int(best_pick)
                top_reselected = True

        top_kf = int(phase_keyframes.get("top", -1))
        imp_k_cur = int(phase_keyframes.get("impact", -1))
        if top_kf >= 0 and imp_k_cur >= 0 and impact_abs_err_before > align_tol:
            phase_reselection_attempted = True
            lo_i = max(top_kf + 2, imp_ev - 8)
            hi_i = min(n - 2, imp_ev + 8)
            if lo_i < hi_i:
                best_pick_i: int | None = None
                best_key_i: tuple | None = None
                for i in range(lo_i, hi_i + 1):
                    sc = _composite_impact_candidate_score(
                        i, top_kf, exc_apex_ev, kin, lo_imp, hi_imp, n, top_kf,
                    )
                    if sc < 0:
                        continue
                    ok, _ = validate_impact_semantic_at_index(i, top_kf, exc_apex_ev, kin)
                    key = (1 if ok else 0, sc, -abs(i - imp_ev))
                    if best_key_i is None or key > best_key_i:
                        best_key_i = key
                        best_pick_i = i
                if best_pick_i is not None and best_pick_i != imp_k_cur:
                    phase_keyframes["impact"] = int(best_pick_i)
                    impact_reselected = True

    top_kf = int(phase_keyframes.get("top", -1))
    imp_kf = int(phase_keyframes.get("impact", -1))
    top_abs_err_after = abs(top_kf - top_ev) if top_kf >= 0 and top_ev >= 0 else 999999
    impact_abs_err_after = abs(imp_kf - imp_ev) if imp_kf >= 0 and imp_ev >= 0 else 999999
    top_align = top_abs_err_after <= align_tol
    imp_align = impact_abs_err_after <= align_tol

    if kin is None or top_kf < 0 or imp_kf < 0:
        t_ok, t_det = False, {}
        i_ok, i_det = False, {}
        final_sem = False
    else:
        t_ok, t_det = validate_top_semantic_at_index(top_kf, kin)
        i_ok, i_det = validate_impact_semantic_at_index(
            imp_kf, top_kf, exc_apex_ev, kin,
        )
        final_sem = bool(t_ok and i_ok)

    align_top_b = bool(top_align)
    align_imp_b = bool(imp_align)
    keyframe_semantic_ok = bool(t_ok and i_ok)

    pv_pass_original = bool(phase_validation.get("passed")) if phase_validation else False
    phase_validation_post_reselect: dict | None = None
    if top_reselected or impact_reselected:
        phase_validation_post_reselect = validate_phase_keyframes(
            dict(phase_keyframes),
            poses,
            source="post_top_impact_reselect",
        )
        pv_pass_for_strict = bool(phase_validation_post_reselect.get("passed"))
    else:
        pv_pass_for_strict = pv_pass_original

    strict_reasons: list[str] = []
    if not keyframe_semantic_ok:
        strict_reasons.append("KEYFRAME_SEMANTIC_FAIL")
    if not align_top_b:
        strict_reasons.append("ALIGN_TOP_FAIL")
    if not align_imp_b:
        strict_reasons.append("ALIGN_IMPACT_FAIL")

    # Phase monotonicity / ordering validation is a soft signal (reliability + warning), not strict 422.
    phase_validation_soft_fail = not pv_pass_for_strict
    phase_validation_warning = (
        "" if pv_pass_for_strict else "phase_validation_soft_fail"
    )

    final_phase_semantic_ok_strict = bool(
        keyframe_semantic_ok and align_top_b and align_imp_b,
    )
    # A reselection is only considered successful if the rerun validation + semantics pass.
    if top_reselected or impact_reselected:
        reselection_verified = bool(pv_pass_for_strict and keyframe_semantic_ok and align_top_b and align_imp_b)
        if not reselection_verified:
            if pk_top0 >= 0:
                phase_keyframes["top"] = pk_top0
            if pk_imp0 >= 0:
                phase_keyframes["impact"] = pk_imp0
            top_reselected = False
            impact_reselected = False

    phase_reselection_failed = bool(
        phase_reselection_attempted
        and (
            not align_top_b
            or not align_imp_b
            or not keyframe_semantic_ok
        )
    )

    from services.keyframe_service import verify_phase_strip_semantics

    strip_sem = verify_phase_strip_semantics(keyframes, poses, phase_keyframes)
    ft_idx = int(phase_keyframes.get("follow_through", -1))
    fin_idx = int(phase_keyframes.get("finish", -1))
    ft_ok, ft_det = (False, {})
    fin_ok, fin_det = (False, {})
    if kin is not None and imp_kf >= 0 and ft_idx >= 0:
        ft_ok, ft_det = validate_follow_through_semantic_at_index(ft_idx, imp_kf, kin)
    if kin is not None and imp_kf >= 0 and ft_idx >= 0 and fin_idx >= 0:
        fin_ok, fin_det = validate_finish_semantic_at_index(fin_idx, imp_kf, ft_idx, kin)
    chain_sep_ok = bool(imp_kf < ft_idx < fin_idx) if (imp_kf >= 0 and ft_idx >= 0 and fin_idx >= 0) else False

    tad = dict(ev.get("time_axis_debug") or {})
    kfc_ev = list(ev.get("kinematic_fail_codes") or [])
    dt_axis_invalid = bool(tad.get("dt_axis_invalid"))
    non_finite_kinematics = bool("NON_FINITE_KINEMATICS" in kfc_ev)
    if dt_axis_invalid or non_finite_kinematics:
        final_phase_semantic_ok_strict = False
        keyframe_semantic_ok = False
        final_sem = False
        t_ok = False
        i_ok = False
        if dt_axis_invalid:
            strict_reasons.append("DT_AXIS_INVALID")
        if non_finite_kinematics:
            strict_reasons.append("NON_FINITE_KINEMATICS")

    fv_in = final_keyframe_validation or {}
    rebuild_used = bool(fv_in.get("rebuild_used"))

    fail_code: str | None = None
    if dt_axis_invalid:
        fail_code = "DT_AXIS_INVALID"
    elif non_finite_kinematics:
        fail_code = "NON_FINITE_KINEMATICS"
    elif kin is None and len(poses) >= 8:
        fail_code = "KINEMATICS_UNAVAILABLE"
    elif not t_ok and kin is not None:
        fail_code = "TOP_SEMANTIC_AT_KEYFRAME_FAIL"
    elif not i_ok and kin is not None:
        fail_code = "IMPACT_SEMANTIC_AT_KEYFRAME_FAIL"
    elif not bool(ev.get("top_semantic_ok")) and kin is not None:
        fail_code = "TOP_EVENT_SEMANTIC_FAIL"
    elif not bool(ev.get("impact_semantic_ok")) and kin is not None:
        fail_code = "IMPACT_EVENT_SEMANTIC_FAIL"

    def _dedupe_reasons(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    final_phase_semantic_ok_strict_reasons = _dedupe_reasons(strict_reasons)

    return {
        "phase_detector_version": PHASE_DETECTOR_VERSION,
        "phase_detector_confidence": ev["phase_detector_confidence"],
        "top_candidate_debug": ev["top_candidate_debug"],
        "impact_candidate_debug": ev["impact_candidate_debug"],
        "align_tol": int(align_tol),
        "top_abs_err": int(top_abs_err_after) if top_abs_err_after < 999999 else None,
        "impact_abs_err": int(impact_abs_err_after) if impact_abs_err_after < 999999 else None,
        "top_abs_err_before": int(top_abs_err_before) if top_abs_err_before < 999999 else None,
        "impact_abs_err_before": int(impact_abs_err_before) if impact_abs_err_before < 999999 else None,
        "top_abs_err_after": int(top_abs_err_after) if top_abs_err_after < 999999 else None,
        "impact_abs_err_after": int(impact_abs_err_after) if impact_abs_err_after < 999999 else None,
        "top_reselected": bool(top_reselected),
        "impact_reselected": bool(impact_reselected),
        "phase_reselection_failed": phase_reselection_failed,
        "top_keyframe_vs_event": {"keyframe": top_kf, "event": top_ev, "aligned": top_align},
        "impact_keyframe_vs_event": {"keyframe": imp_kf, "event": imp_ev, "aligned": imp_align},
        "top_semantic_at_keyframe": t_det if kin else {},
        "impact_semantic_at_keyframe": i_det if kin else {},
        "top_semantic_ok": bool(t_ok),
        "impact_semantic_ok": bool(i_ok),
        "keyframe_semantic_ok": keyframe_semantic_ok,
        "align_top": align_top_b,
        "align_impact": align_imp_b,
        "phase_validation_passed": pv_pass_for_strict,
        "phase_validation_passed_original": pv_pass_original,
        "phase_validation_reran_after_reselect": phase_validation_post_reselect is not None,
        "phase_validation_post_reselect": phase_validation_post_reselect,
        "phase_validation_soft_fail": bool(phase_validation_soft_fail),
        "phase_validation_warning": phase_validation_warning,
        "final_phase_semantic_ok_strict_reasons": final_phase_semantic_ok_strict_reasons,
        "final_phase_semantic_ok": bool(final_sem),
        "final_phase_semantic_ok_strict": final_phase_semantic_ok_strict,
        "semantic_validation": {
            "detector_top_pass": ev.get("top_semantic_ok"),
            "detector_impact_pass": ev.get("impact_semantic_ok"),
            "keyframe_top_pass": t_ok,
            "keyframe_impact_pass": i_ok,
            "align_top": align_top_b,
            "align_impact": align_imp_b,
            "phase_validation_passed": pv_pass_for_strict,
            "phase_validation_passed_original": pv_pass_original,
            "phase_validation_reran_after_reselect": phase_validation_post_reselect is not None,
            "phase_validation_soft_fail": bool(phase_validation_soft_fail),
            "phase_validation_warning": phase_validation_warning,
            "full_pipeline_semantic_ok": bool(final_sem and pv_pass_for_strict),
        },
        "phase_strip_semantic_ok": bool(strip_sem.get("pass")),
        "phase_strip_semantic_reasons": list(strip_sem.get("reasons") or []),
        "phase_chain_rebuilt": bool(
            top_reselected or impact_reselected or fv_in.get("phase_strip_repaired"),
        ),
        "material_change": bool(
            top_reselected or impact_reselected or fv_in.get("rebuild_used") or fv_in.get("phase_strip_repaired"),
        ),
        "follow_through_semantic_ok": bool(ft_ok),
        "follow_through_semantic_detail": dict(ft_det),
        "finish_semantic_ok": bool(fin_ok),
        "finish_semantic_detail": dict(fin_det),
        "post_impact_chain_separation_ok": chain_sep_ok,
        "dt_axis_invalid": dt_axis_invalid,
        "non_finite_kinematics": non_finite_kinematics,
        "time_axis_debug": tad,
        "kinematic_fail_codes": kfc_ev,
        "rebuild_used": rebuild_used,
        "fail_code": fail_code,
    }


def _ensure_bounds(raw: list[int], n: int) -> list[int]:
    """Clamp 7 boundary values to be strictly increasing within [1, n-1].

    This guarantees every phase gets at least 1 frame.
    """
    b = list(raw)
    k = len(b)
    for i in range(k):
        lo = (b[i - 1] + 1) if i > 0 else 1
        b[i] = max(b[i], lo)
    for i in range(k - 1, -1, -1):
        hi = (b[i + 1] - 1) if i < k - 1 else (n - 1)
        b[i] = min(b[i], hi)
    for i in range(k):
        lo = (b[i - 1] + 1) if i > 0 else 1
        b[i] = max(b[i], lo)
    return b


def detect_swing_phases(poses: list[dict]) -> list[dict]:
    """Detect 8 swing phases from pose kinematics.

    Top/impact anchors use view-agnostic 2D hand excursion + speed + rotation peaks
    (detection-time joints). Remaining boundaries use light-smoothed wrist paths.
    """
    # Filter out non-golf frames (camera cuts, close-ups, crowd shots)
    valid_poses, valid_indices = _filter_golf_poses(poses)

    if len(valid_poses) < 8:
        logger.warning("Only %d valid golf poses (need 8+), using default phases on ALL poses", len(valid_poses))
        return _default_phases(poses)

    n = len(valid_poses)

    _ev = detect_phase_events_agnostic(poses)
    top_idx = _snap_full_idx_to_valid(int(_ev["top_pose_idx"]), valid_indices)
    impact_idx = _snap_full_idx_to_valid(int(_ev["impact_pose_idx"]), valid_indices)
    impact_idx = max(impact_idx, top_idx + 2)
    impact_idx = min(impact_idx, n - 3)

    rw_y = _extract_trajectory(valid_poses, "right_wrist", "y")
    lw_y = _extract_trajectory(valid_poses, "left_wrist", "y")
    wrist_y = _smooth_signal((rw_y + lw_y) / 2.0, window=3)

    _t_grad, _ = _build_time_axis_for_gradient(valid_poses)
    wrist_vy, _ = safe_gradient(wrist_y, _t_grad)

    rw_x = _extract_trajectory(valid_poses, "right_wrist", "x")
    lw_x = _extract_trajectory(valid_poses, "left_wrist", "x")
    wrist_x = _smooth_signal((rw_x + lw_x) / 2.0, window=3)
    wrist_vx, _ = safe_gradient(wrist_x, _t_grad)
    wrist_speed = np.sqrt(wrist_vx * wrist_vx + wrist_vy * wrist_vy)

    hip_y = _smooth_signal(
        (_extract_trajectory(valid_poses, "right_hip", "y") + _extract_trajectory(valid_poses, "left_hip", "y")) / 2.0,
        window=3,
    )
    x_factor = np.array([
        float(p.get("angles", {}).get("x_factor", 0.0)) for p in valid_poses
    ])
    x_factor = _smooth_signal(x_factor, window=3)

    # Address/takeaway boundary by departure from stable setup.
    setup_len = max(3, min(top_idx, int(n * 0.16)))
    base_x = float(np.mean(wrist_x[:setup_len])) if setup_len > 0 else float(wrist_x[0])
    base_y = float(np.mean(wrist_y[:setup_len])) if setup_len > 0 else float(wrist_y[0])
    move = np.sqrt((wrist_x - base_x) ** 2 + (wrist_y - base_y) ** 2)
    move_thr = max(float(np.percentile(move[:max(4, top_idx)], 75)) * 1.35, 0.005)
    takeaway_start = max(1, min(top_idx - 1, int(setup_len)))
    for i in range(max(1, setup_len - 1), max(2, top_idx - 1)):
        if move[i] > move_thr and wrist_vy[min(i + 1, n - 1)] <= 0:
            takeaway_start = i
            break

    backswing_start = takeaway_start
    if top_idx - takeaway_start >= 3:
        peak_progress = (wrist_y[takeaway_start] - wrist_y[top_idx]) * 0.55
        for i in range(takeaway_start + 1, top_idx):
            if (wrist_y[takeaway_start] - wrist_y[i]) >= peak_progress:
                backswing_start = i
                break

    # Follow-through to finish boundary by speed decay + stable body.
    finish_start = min(n - 1, impact_idx + max(3, (n - impact_idx) // 2))
    post = range(min(n - 2, impact_idx + 2), n - 1)
    post_speed_ref = max(float(np.percentile(wrist_speed[impact_idx:], 80)), 1e-6)
    for i in post:
        stable = wrist_speed[i] < post_speed_ref * 0.28
        good_q = _pose_quality_score(valid_poses[i]) > 0.55
        if stable and good_q:
            finish_start = i
            break

    # ── 7 boundary values (each = first frame of the next phase) ──
    bounds = _ensure_bounds([
        takeaway_start,           # address → takeaway
        backswing_start,          # takeaway → backswing
        top_idx,                  # backswing → top
        top_idx + 2,              # top → downswing
        impact_idx,               # downswing → impact
        impact_idx + 2,           # impact → follow_through
        finish_start,             # follow_through → finish
    ], n)

    logger.info(
        "Phase detection: n=%d (from %d total) top=%d impact=%d takeaway=%d finish=%d bounds=%s",
        n, len(poses), top_idx, impact_idx, takeaway_start, finish_start, bounds,
    )

    # Build phase assignments for VALID poses
    valid_phases = []
    for i in range(n):
        pi = 0
        for bi, b in enumerate(bounds):
            if i >= b:
                pi = bi + 1
        pid = _PHASE_IDS[pi]
        pct = (i / max(n - 1, 1)) * 100
        info = next(p for p in SWING_PHASES if p["id"] == pid)
        valid_phases.append({
            "frame_index": valid_poses[i].get("frame_index", i),
            "phase_id": pid,
            "phase_en": info["en"],
            "phase_zh": info["zh"],
            "progress_pct": round(pct, 1),
        })

    # Map back to original pose array: valid poses get proper phases,
    # invalid poses (camera cuts) get the nearest valid pose's phase
    all_phases = []
    valid_phase_idx = 0
    for i in range(len(poses)):
        if valid_phase_idx < len(valid_indices) and i == valid_indices[valid_phase_idx]:
            all_phases.append(valid_phases[valid_phase_idx])
            valid_phase_idx += 1
        else:
            # Assign nearest valid phase
            nearest = _find_nearest_valid_phase(i, valid_indices, valid_phases, poses)
            all_phases.append(nearest)

    return all_phases


def _find_nearest_valid_phase(idx: int, valid_indices: list[int],
                               valid_phases: list[dict], poses: list[dict]) -> dict:
    """For an invalid pose, assign the nearest valid pose's phase."""
    if not valid_indices:
        pct = (idx / max(len(poses) - 1, 1)) * 100
        info = SWING_PHASES[0]
        for p in SWING_PHASES:
            if pct >= p["pct"][0]:
                info = p
        return {
            "frame_index": poses[idx].get("frame_index", idx),
            "phase_id": info["id"],
            "phase_en": info["en"],
            "phase_zh": info["zh"],
            "progress_pct": round(pct, 1),
        }

    best_vi = 0
    best_dist = abs(idx - valid_indices[0])
    for vi, vi_idx in enumerate(valid_indices):
        d = abs(idx - vi_idx)
        if d < best_dist:
            best_dist = d
            best_vi = vi

    phase = dict(valid_phases[best_vi])
    phase["frame_index"] = poses[idx].get("frame_index", idx)
    return phase


def _default_phases(poses: list[dict]) -> list[dict]:
    n = max(len(poses), 1)
    result = []
    for i, pose in enumerate(poses):
        pct = (i / max(n - 1, 1)) * 100
        info = SWING_PHASES[0]
        for p in SWING_PHASES:
            if pct >= p["pct"][0]:
                info = p
        result.append({
            "frame_index": pose.get("frame_index", i),
            "phase_id": info["id"],
            "phase_en": info["en"],
            "phase_zh": info["zh"],
            "progress_pct": round(pct, 1),
        })
    return result


def _stabilize(phases: list[dict]) -> list[dict]:
    """Ensure phase progression never goes backwards."""
    max_ord = -1
    for p in phases:
        o = PHASE_ORDER.get(p["phase_id"], 0)
        if o < max_ord:
            info = SWING_PHASES[max_ord]
            p["phase_id"] = info["id"]
            p["phase_en"] = info["en"]
            p["phase_zh"] = info["zh"]
        else:
            max_ord = o
    return phases


def build_phase_keyframes_from_top_impact_anchors(
    poses: list[dict],
    phase_keyframes_old: dict[str, int],
) -> tuple[dict[str, int], bool, dict]:
    """Lock top/impact from kinematic events, then pick the other six phases in local windows.

    Returns (phase_keyframes, anchor_ok, debug). On failure, caller should fall back to bucket logic.
    """
    n = len(poses)
    dbg: dict = {"anchor_ok": False}
    if n < 12:
        dbg["fail"] = "n_lt_12"
        return dict(phase_keyframes_old), False, dbg
    kin = _build_view_agnostic_kinematics(poses)
    if kin is None:
        dbg["fail"] = "no_kinematics"
        return dict(phase_keyframes_old), False, dbg
    ev = detect_phase_events_agnostic(poses)
    pk: dict[str, int] = {
        "top": int(ev["top_pose_idx"]),
        "impact": int(ev["impact_pose_idx"]),
    }
    refine_phase_keyframes_top_impact(poses, pk)
    top_i = int(pk["top"])
    imp_i = int(pk["impact"])
    if imp_i < top_i + 3:
        dbg["fail"] = "top_impact_gap_lt_3"
        dbg["top_i"], dbg["imp_i"] = top_i, imp_i
        return dict(phase_keyframes_old), False, dbg

    min_gap = max(1, min(4, n // 22))
    q = np.array([_pose_quality_score(p) for p in poses])
    valid_m = kin["valid"]

    def _pick_window(lo: int, hi: int, floor_prev: int) -> int | None:
        lo_e = max(int(lo), int(floor_prev) + min_gap)
        hi_e = min(int(hi), n - 1)
        if lo_e > hi_e:
            return None
        idxs = [i for i in range(lo_e, hi_e + 1) if bool(valid_m[i])]
        pool = idxs if idxs else list(range(lo_e, hi_e + 1))
        return int(max(pool, key=lambda i: float(q[i])))

    hi_addr = max(0, top_i - 4 * min_gap - 2)
    addr = _pick_window(0, hi_addr, -min_gap - 1)
    if addr is None:
        dbg["fail"] = "address_window"
        return dict(phase_keyframes_old), False, dbg
    tw = _pick_window(addr + min_gap, top_i - 2 * min_gap - 1, addr)
    if tw is None:
        dbg["fail"] = "takeaway_window"
        return dict(phase_keyframes_old), False, dbg
    bs = _pick_window(tw + min_gap, top_i - min_gap - 1, tw)
    if bs is None or bs >= top_i:
        dbg["fail"] = "backswing_window"
        return dict(phase_keyframes_old), False, dbg
    down = _pick_window(top_i + min_gap, imp_i - min_gap - 1, top_i)
    if down is None:
        dbg["fail"] = "downswing_window"
        return dict(phase_keyframes_old), False, dbg
    follow = _pick_window(imp_i + min_gap, n - min_gap - 2, imp_i)
    if follow is None:
        dbg["fail"] = "follow_through_window"
        return dict(phase_keyframes_old), False, dbg
    finish = _pick_window(follow + min_gap, n - 1, follow)
    if finish is None:
        dbg["fail"] = "finish_window"
        return dict(phase_keyframes_old), False, dbg

    result = {
        "address": addr,
        "takeaway": tw,
        "backswing": bs,
        "top": top_i,
        "downswing": down,
        "impact": imp_i,
        "follow_through": follow,
        "finish": finish,
    }
    seq = [result[p] for p in _PHASE_IDS]
    if seq != sorted(seq) or len(set(seq)) != 8:
        dbg["fail"] = "monotonic_unique"
        dbg["seq"] = seq
        return dict(phase_keyframes_old), False, dbg
    dbg["anchor_ok"] = True
    return result, True, dbg


def _get_phase_keyframes_bucket_driven(phases: list[dict], poses: list[dict]) -> dict[str, int]:
    """Representative pose index for each phase.

    Uses phase-specific event rules for all 8 phases and pose quality scoring.
    Avoids generic bucket-midpoint selection for teaching-grade key moments.
    """
    buckets: dict[str, list[int]] = {pid: [] for pid in _PHASE_IDS}
    for i, p in enumerate(phases):
        pid = p.get("phase_id")
        if pid in buckets:
            buckets[pid].append(i)

    if not poses:
        return {pid: (idxs[len(idxs) // 2] if idxs else 0) for pid, idxs in buckets.items()}

    n = len(poses)
    _ev_kf = detect_phase_events_agnostic(poses)
    _ev_top_i = int(_ev_kf["top_pose_idx"])
    _ev_imp_i = int(_ev_kf["impact_pose_idx"])
    tol_snap = max(5, n // 12)

    wrist_y = _smooth_signal(
        (_extract_trajectory(poses, "right_wrist", "y") + _extract_trajectory(poses, "left_wrist", "y")) / 2.0,
        window=3,
    )
    wrist_x = _smooth_signal(
        (_extract_trajectory(poses, "right_wrist", "x") + _extract_trajectory(poses, "left_wrist", "x")) / 2.0,
        window=3,
    )
    _t_kp, _ = _build_time_axis_for_gradient(poses)
    wrist_vy, _ = safe_gradient(wrist_y, _t_kp)
    wrist_vx, _ = safe_gradient(wrist_x, _t_kp)
    wrist_speed = np.sqrt(wrist_vx * wrist_vx + wrist_vy * wrist_vy)
    q_scores = np.array([_pose_quality_score(p) for p in poses])

    result: dict[str, int] = {}
    top_idx = None
    # address: stable body, low hand speed, high quality
    addr_idxs = buckets.get("address", [])
    if addr_idxs:
        best = max(addr_idxs, key=lambda i: (q_scores[i] * 0.65 + (1.0 - min(wrist_speed[i] / (np.percentile(wrist_speed, 90) + 1e-6), 1.0)) * 0.35))
        result["address"] = int(best)

    # takeaway: first meaningful departure from address and continued backswing direction
    tw_idxs = buckets.get("takeaway", [])
    if tw_idxs:
        base_i = result.get("address", tw_idxs[0])
        base_x, base_y = wrist_x[base_i], wrist_y[base_i]
        move = np.sqrt((wrist_x - base_x) ** 2 + (wrist_y - base_y) ** 2)
        move_scale = max(float(np.percentile(move[tw_idxs], 85)), 1e-6)
        best = max(
            tw_idxs,
            key=lambda i: (
                0.50 * min(float(move[i]) / move_scale, 1.0) +
                0.30 * (1.0 if wrist_vy[min(i + 1, n - 1)] <= 0 else 0.0) +
                0.20 * q_scores[i]
            ),
        )
        result["takeaway"] = int(best)

    # backswing: representative frame by progress toward top + quality
    bs_idxs = buckets.get("backswing", [])
    if bs_idxs:
        start_i = result.get("takeaway", bs_idxs[0])
        min_y = float(np.min(wrist_y[bs_idxs]))
        total_lift = max(float(wrist_y[start_i] - min_y), 1e-6)
        best = max(
            bs_idxs,
            key=lambda i: (
                0.45 * (1.0 - min(abs((wrist_y[start_i] - wrist_y[i]) / total_lift - 0.55) / 0.55, 1.0)) +
                0.30 * (1.0 if wrist_vy[i] <= 0 else 0.0) +
                0.25 * q_scores[i]
            ),
        )
        result["backswing"] = int(best)

    # top: event detector first; snap into bucket only when close (avoid bucket error propagation).
    top_idxs = buckets.get("top", [])
    if top_idxs:
        near = min(top_idxs, key=lambda i: abs(i - _ev_top_i))
        result["top"] = int(near if abs(near - _ev_top_i) <= tol_snap else _ev_top_i)
        top_idx = result["top"]
        if top_idx < 0 or top_idx >= n:
            top_idx = _pick_top_frame(poses, top_idxs)
            result["top"] = int(top_idx if top_idx is not None else top_idxs[len(top_idxs) // 2])
    else:
        result["top"] = int(min(max(_ev_top_i, 0), n - 1))
        top_idx = result["top"]

    # downswing: clear acceleration after top
    ds_idxs = buckets.get("downswing", [])
    if ds_idxs:
        top_ref = result.get("top", ds_idxs[0])
        vy_scale = max(float(np.percentile(np.abs(wrist_vy[ds_idxs]), 90)), 1e-6)
        best = max(
            ds_idxs,
            key=lambda i: (
                0.55 * np.clip(float(wrist_vy[i]) / vy_scale, 0.0, 1.0) +
                0.25 * np.clip((i - top_ref) / max(ds_idxs[-1] - top_ref, 1), 0.0, 1.0) +
                0.20 * q_scores[i]
            ),
        )
        result["downswing"] = int(best)

    # impact: event detector first; snap into bucket when close.
    imp_idxs = buckets.get("impact", [])
    if imp_idxs:
        top_ref = result.get("top")
        near_i = min(imp_idxs, key=lambda j: abs(j - _ev_imp_i))
        if abs(near_i - _ev_imp_i) <= tol_snap and (top_ref is None or near_i > top_ref):
            result["impact"] = int(near_i)
        else:
            impact_idx = _pick_impact_frame(poses, imp_idxs, top_idx=top_ref)
            pick = impact_idx if impact_idx is not None else _ev_imp_i
            if top_ref is not None and pick <= top_ref:
                pick = min(n - 1, top_ref + max(2, n // 20))
            result["impact"] = int(min(max(pick, 0), n - 1))
    else:
        tr = result.get("top", 0)
        result["impact"] = int(min(max(_ev_imp_i, tr + 2), n - 1))

    # follow-through: post-impact high-speed release frame
    ft_idxs = buckets.get("follow_through", [])
    if ft_idxs:
        imp_ref = result.get("impact", ft_idxs[0])
        speed_scale = max(float(np.percentile(wrist_speed[ft_idxs], 90)), 1e-6)
        hips_mid_x = (_extract_trajectory(poses, "left_hip", "x") + _extract_trajectory(poses, "right_hip", "x")) / 2.0
        best = max(
            ft_idxs,
            key=lambda i: (
                0.42 * np.clip(float(wrist_speed[i]) / speed_scale, 0.0, 1.0) +
                0.30 * min(abs(float(wrist_x[i] - hips_mid_x[i])) / 0.25, 1.0) +
                0.16 * np.clip((i - imp_ref) / max(ft_idxs[-1] - imp_ref, 1), 0.0, 1.0) +
                0.12 * q_scores[i]
            ),
        )
        result["follow_through"] = int(best)

    # finish: late stable high-quality frame, not simply last frame
    fin_idxs = buckets.get("finish", [])
    if fin_idxs:
        speed_ref = max(float(np.percentile(wrist_speed[fin_idxs], 90)), 1e-6)
        best = max(
            fin_idxs,
            key=lambda i: (
                0.55 * q_scores[i] +
                0.35 * (1.0 - min(float(wrist_speed[i]) / speed_ref, 1.0)) +
                0.10 * np.clip((i - fin_idxs[0]) / max(fin_idxs[-1] - fin_idxs[0], 1), 0.0, 1.0)
            ),
        )
        result["finish"] = int(best)

    # Missing phase recovery: prefer nearest valid neighbor frame (not fixed % time).
    chosen = {pid for pid, idx in result.items() if isinstance(idx, int) and 0 <= idx < n}
    for pid in _PHASE_IDS:
        if pid in chosen:
            continue
        idxs = buckets.get(pid, [])
        if idxs:
            result[pid] = int(max(idxs, key=lambda i: q_scores[i]))
            continue
        pos = _PHASE_IDS.index(pid)
        neighbor = None
        for step in range(1, len(_PHASE_IDS)):
            l = pos - step
            r = pos + step
            if l >= 0 and _PHASE_IDS[l] in result:
                neighbor = result[_PHASE_IDS[l]]
                break
            if r < len(_PHASE_IDS) and _PHASE_IDS[r] in result:
                neighbor = result[_PHASE_IDS[r]]
                break
        result[pid] = int(neighbor if neighbor is not None else min(max(pos * n // 8, 0), max(n - 1, 0)))

    refine_phase_keyframes_top_impact(poses, result)

    # ── Enforce minimum gap between adjacent phase indices (soft — avoids shoving top/impact apart) ──
    min_gap = max(1, min(4, n // 22))
    prev_idx = -min_gap
    for pid in _PHASE_IDS:
        idx = result.get(pid, 0)
        if idx < prev_idx + min_gap:
            idx = prev_idx + min_gap
        idx = min(idx, n - 1)
        result[pid] = idx
        prev_idx = idx
    # Backward pass: pull down if we hit the ceiling too early
    for j in range(len(_PHASE_IDS) - 2, -1, -1):
        cur_pid = _PHASE_IDS[j]
        nxt_pid = _PHASE_IDS[j + 1]
        if result[cur_pid] >= result[nxt_pid]:
            result[cur_pid] = max(0, result[nxt_pid] - min_gap)
    # Final forward pass to re-enforce min_gap after backward adjustments
    prev_idx = -min_gap
    for pid in _PHASE_IDS:
        result[pid] = max(result[pid], prev_idx + min_gap)
        result[pid] = min(result[pid], n - 1)
        prev_idx = result[pid]

    refine_phase_keyframes_top_impact(poses, result)
    if result.get("impact", 0) <= result.get("top", 0):
        result["impact"] = min(n - 1, int(result["top"]) + max(2, n // 25))

    return result


_PHASE_IDS_SET = frozenset(_PHASE_IDS)


def _keyframe_map_from_per_frame_phase(poses: list[dict], per_frame: list[str]) -> dict[str, int] | None:
    """Pick highest-quality pose index inside each contiguous phase label run."""
    n = len(poses)
    if len(per_frame) != n or n < 8:
        return None
    if not all(str(p) in _PHASE_IDS_SET for p in per_frame):
        return None
    q_scores = np.array([_pose_quality_score(p) for p in poses], dtype=np.float64)
    out: dict[str, int] = {}
    for pid in _PHASE_IDS:
        idxs = [i for i in range(n) if str(per_frame[i]) == pid]
        if not idxs:
            return None
        best_i = int(max(idxs, key=lambda i: q_scores[i]))
        out[pid] = best_i
    refine_phase_keyframes_top_impact(poses, out)
    if out.get("impact", 0) <= out.get("top", 0):
        out["impact"] = min(n - 1, int(out["top"]) + max(2, n // 25))
    return out


def get_phase_keyframes(
    phases: list[dict],
    poses: list[dict] | None = None,
    *,
    segment_bundle: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Representative pose index per phase.

    When ``segment_bundle`` / per-frame labels align with poses, prefers event-driven
    segment representatives over pure bucket midpoints. Otherwise evaluates anchor vs bucket.
    """
    if not poses:
        return _get_phase_keyframes_bucket_driven(phases, poses or [])

    n = len(poses)
    pfp = list((segment_bundle or {}).get("per_frame_phase") or [])
    if not pfp and phases and len(phases) == len(poses):
        pfp = [str(p.get("phase_id") or "") for p in phases]
    seg_map: dict[str, int] | None = None
    if pfp and len(pfp) == n:
        seg_map = _keyframe_map_from_per_frame_phase(poses, pfp)

    bucket_map = _get_phase_keyframes_bucket_driven(phases, poses)
    refine_phase_keyframes_top_impact(poses, bucket_map)
    if bucket_map.get("impact", 0) <= bucket_map.get("top", 0):
        bucket_map["impact"] = min(n - 1, int(bucket_map["top"]) + max(2, n // 25))
    bucket_v = validate_phase_keyframes(bucket_map, poses, source="bucket")

    anchor_map = dict(bucket_map)
    anchored, ok, _dbg = build_phase_keyframes_from_top_impact_anchors(poses, {})
    if ok:
        anchor_map = dict(anchored)
        refine_phase_keyframes_top_impact(poses, anchor_map)
        if anchor_map.get("impact", 0) <= anchor_map.get("top", 0):
            anchor_map["impact"] = min(n - 1, int(anchor_map["top"]) + max(2, n // 25))
    anchor_v = validate_phase_keyframes(anchor_map, poses, source="anchor")

    def _score(v: dict) -> tuple[int, int, int]:
        return (
            1 if bool(v.get("passed")) else 0,
            -len(v.get("issues") or []),
            int(v.get("min_gap") or 0),
        )

    candidates: list[tuple[dict[str, int], dict, str]] = [
        (bucket_map, bucket_v, "bucket"),
        (anchor_map, anchor_v, "anchor"),
    ]
    if seg_map is not None:
        seg_v = validate_phase_keyframes(seg_map, poses, source="segment")
        tps = float((segment_bundle or {}).get("temporal_prior_strength") or 0.0)
        relaxed = (
            tps >= 0.58
            and bool(seg_v.get("order_ok"))
            and bool(seg_v.get("spacing_ok"))
        )
        seg_for_score = dict(seg_v)
        if relaxed and not bool(seg_v.get("passed")):
            seg_for_score["passed"] = True
        # Strict pass, or strong MMAction2 temporal prior with ordered spaced segment strip.
        if bool(seg_v.get("passed")) or relaxed:
            candidates.append((seg_map, seg_for_score, "segment"))

    chosen_map, chosen_v, chosen_name = max(candidates, key=lambda t: _score(t[1]))
    logger.info(
        "get_phase_keyframes pick=%s pass=%s issues=%d",
        chosen_name,
        bool(chosen_v.get("passed")),
        len(chosen_v.get("issues") or []),
    )
    return chosen_map


def _pick_by_wrist_y(poses: list[dict], idxs: list[int], minimize: bool = True) -> int | None:
    """Pick the pose index where average wrist Y is min (highest hands) or max."""
    best_idx = None
    best_val = float("inf") if minimize else float("-inf")
    for i in idxs:
        if i >= len(poses):
            continue
        if not _is_valid_golf_pose(poses[i]):
            continue
        joints = poses[i].get("joints", [])
        rw = next((j for j in joints if j["name"] == "right_wrist"), None)
        lw = next((j for j in joints if j["name"] == "left_wrist"), None)
        if rw and lw:
            avg_y = (rw["normalized"]["y"] + lw["normalized"]["y"]) / 2
            if (minimize and avg_y < best_val) or (not minimize and avg_y > best_val):
                best_val = avg_y
                best_idx = i
    return best_idx


def _pick_top_frame(poses: list[dict], idxs: list[int]) -> int | None:
    """Top-of-backswing selector: extremum + low vy + reversal + x_factor peak + shoulder turn."""
    if not idxs:
        return None
    wrist_y = _smooth_signal(
        (_extract_trajectory(poses, "right_wrist", "y") + _extract_trajectory(poses, "left_wrist", "y")) / 2.0,
        window=5,
    )
    _t_top, _ = _build_time_axis_for_gradient(poses)
    vy, _ = safe_gradient(wrist_y, _t_top)
    y_span = max(float(np.max(wrist_y[idxs]) - np.min(wrist_y[idxs])), 1e-6)
    vy_ref = max(float(np.percentile(np.abs(vy[idxs]), 90)), 1e-6)

    # Additional signals: x_factor and shoulder rotation peak at top
    x_factor = _smooth_signal(
        np.array([float(p.get("angles", {}).get("x_factor", 0.0)) for p in poses]), 3
    )
    shoulder_rot = _smooth_signal(
        np.array([abs(float(p.get("angles", {}).get("shoulder_rotation", 0.0))) for p in poses]), 3
    )
    xf_max = max(float(np.max(x_factor[idxs])), 1e-6)
    sr_max = max(float(np.max(shoulder_rot[idxs])), 1e-6)

    best_idx = idxs[len(idxs) // 2]
    best_score = -1.0
    for i in idxs:
        if i <= 0 or i >= len(poses) - 1:
            continue
        if not _is_valid_golf_pose(poses[i]):
            continue
        height = (float(np.max(wrist_y[idxs])) - float(wrist_y[i])) / y_span
        near_zero_v = 1.0 - min(abs(float(vy[i])) / vy_ref, 1.0)
        reversal = 1.0 if vy[i - 1] <= 0.0 and vy[i + 1] >= 0.0 else 0.0
        quality = _pose_quality_score(poses[i])
        xf_score = float(x_factor[i]) / xf_max
        sr_score = float(shoulder_rot[i]) / sr_max
        score = (
            0.32 * height +
            0.20 * near_zero_v +
            0.15 * reversal +
            0.13 * xf_score +
            0.10 * sr_score +
            0.10 * quality
        )
        if score > best_score:
            best_score = score
            best_idx = i
    return int(best_idx)


def _pick_impact_frame(poses: list[dict], idxs: list[int], top_idx: Optional[int] = None) -> int | None:
    """Impact selector with multi-signal scoring + hard constraints.

    Hard constraints:
      - Must be after top_idx
      - Wrist Y must be within 20% of hip Y range (hands near ball height)
      - Wrist speed must be in top 50% of candidates
    """
    if not idxs:
        return None
    wrist_y = _smooth_signal(
        (_extract_trajectory(poses, "right_wrist", "y") + _extract_trajectory(poses, "left_wrist", "y")) / 2.0,
        window=5,
    )
    wrist_x = _smooth_signal(
        (_extract_trajectory(poses, "right_wrist", "x") + _extract_trajectory(poses, "left_wrist", "x")) / 2.0,
        window=5,
    )
    _t_imp, _ = _build_time_axis_for_gradient(poses)
    wrist_vy, _ = safe_gradient(wrist_y, _t_imp)
    wrist_vx, _ = safe_gradient(wrist_x, _t_imp)
    wrist_speed = np.sqrt(wrist_vx * wrist_vx + wrist_vy * wrist_vy)

    hip_y = (
        _extract_trajectory(poses, "left_hip", "y") +
        _extract_trajectory(poses, "right_hip", "y")
    ) / 2.0
    shoulder_rot = _smooth_signal(np.array([float(p.get("angles", {}).get("shoulder_rotation", 0.0)) for p in poses]), 3)
    hip_rot = _smooth_signal(np.array([float(p.get("angles", {}).get("hip_rotation", 0.0)) for p in poses]), 3)
    x_factor = _smooth_signal(np.array([float(p.get("angles", {}).get("x_factor", 0.0)) for p in poses]), 3)
    rot_comb = shoulder_rot - hip_rot
    drot, _ = safe_gradient(rot_comb, _t_imp)
    rot_delta = np.abs(drot)
    dxf, _ = safe_gradient(x_factor, _t_imp)
    xfactor_drop = np.clip(-dxf, 0.0, None)

    speed_ref = max(float(np.percentile(wrist_speed[idxs], 95)), 1e-6)
    vy_ref = max(float(np.percentile(np.abs(wrist_vy[idxs]), 95)), 1e-6)
    hip_ref = max(float(np.percentile(np.abs(wrist_y[idxs] - hip_y[idxs]), 90)), 1e-6)
    rot_ref = max(float(np.percentile(rot_delta[idxs], 90)), 1e-6)
    drop_ref = max(float(np.percentile(xfactor_drop[idxs], 90)), 1e-6)
    quality = np.array([_pose_quality_score(p) for p in poses])

    # Hard constraint: minimum speed threshold (top 50% of candidates)
    candidate_speeds = [float(wrist_speed[i]) for i in idxs if 0 < i < len(poses) - 1]
    speed_threshold = float(np.median(candidate_speeds)) if candidate_speeds else 0.0

    best_idx = None
    best_score = -1.0
    for i in idxs:
        if i <= 0 or i >= len(poses) - 1:
            continue
        if not _is_valid_golf_pose(poses[i]):
            continue
        if top_idx is not None and i <= top_idx:
            continue
        # Hard constraint: speed must be above median
        if float(wrist_speed[i]) < speed_threshold * 0.5:
            continue

        down_speed = np.clip(float(wrist_vy[i]) / vy_ref, 0.0, 1.0)
        speed_mag = np.clip(float(wrist_speed[i]) / speed_ref, 0.0, 1.0)
        hand_zone = 1.0 - min(abs(float(wrist_y[i] - hip_y[i])) / hip_ref, 1.0)
        rot_trend = np.clip(float(rot_delta[i]) / rot_ref, 0.0, 1.0)
        unwind = np.clip(float(xfactor_drop[i]) / drop_ref, 0.0, 1.0)
        time_term = 1.0
        if top_idx is not None:
            # Prefer early-middle of post-top strike window.
            rel = (i - top_idx) / max(len(poses) - 1 - top_idx, 1)
            time_term = 1.0 - min(abs(rel - 0.28) / 0.35, 1.0)

        score = (
            0.26 * down_speed +
            0.22 * speed_mag +
            0.17 * hand_zone +
            0.14 * rot_trend +
            0.11 * unwind +
            0.06 * time_term +
            0.04 * quality[i]
        )
        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx) if best_idx is not None else None


def smooth_pose_sequence(poses: list[dict], alpha: float = 0.4) -> list[dict]:
    """Exponential moving average smoothing of joint positions across frames."""
    if len(poses) < 2:
        return poses

    smoothed = [_deep_copy_pose(poses[0])]
    for i in range(1, len(poses)):
        prev = smoothed[-1]
        curr = _deep_copy_pose(poses[i])
        for ji, joint in enumerate(curr["joints"]):
            pj = prev["joints"][ji] if ji < len(prev["joints"]) else None
            if not pj or pj["name"] != joint["name"]:
                continue
            if joint.get("visibility", 0) < 0.2 or pj.get("visibility", 0) < 0.2:
                continue
            joint["x"] = round(alpha * joint["x"] + (1 - alpha) * pj["x"], 2)
            joint["y"] = round(alpha * joint["y"] + (1 - alpha) * pj["y"], 2)
            joint["z"] = round(alpha * joint["z"] + (1 - alpha) * pj["z"], 2)
            if "normalized" in joint and "normalized" in pj:
                joint["normalized"]["x"] = round(
                    alpha * joint["normalized"]["x"] + (1 - alpha) * pj["normalized"]["x"], 4
                )
                joint["normalized"]["y"] = round(
                    alpha * joint["normalized"]["y"] + (1 - alpha) * pj["normalized"]["y"], 4
                )
        curr["angles"] = _recompute_angles(curr["joints"])
        smoothed.append(curr)
    return smoothed


def compute_wrist_trajectory(poses: list[dict]) -> list[dict]:
    """Extract wrist trajectory with velocity for each frame."""
    traj = []
    for pose in poses:
        joints = pose.get("joints", [])
        rw = next((j for j in joints if j["name"] == "right_wrist"), None)
        if not rw:
            continue
        nx = rw.get("normalized", {}).get("x", 0)
        ny = rw.get("normalized", {}).get("y", 0)
        ts = pose.get("timestamp", 0)
        speed = 0.0
        if traj:
            dx = nx - traj[-1]["x"]
            dy = ny - traj[-1]["y"]
            dt = max(ts - traj[-1].get("timestamp", 0), 0.001)
            speed = round(((dx ** 2 + dy ** 2) ** 0.5) / dt, 3)
        traj.append({
            "frame_index": pose.get("frame_index", 0),
            "timestamp": ts,
            "x": nx,
            "y": ny,
            "speed": speed,
        })
    return traj


def _deep_copy_pose(pose: dict) -> dict:
    import copy
    return copy.deepcopy(pose)


def _recompute_angles(joints: list[dict]) -> dict:
    """Recompute golf angles from smoothed joint positions."""
    def get(name: str) -> np.ndarray:
        for j in joints:
            if j["name"] == name:
                return np.array([j["x"], j["y"]])
        return np.array([0.0, 0.0])

    def angle(a, b, c):
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return round(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), 1)

    angles = {}
    angles["left_elbow"] = angle(get("left_shoulder"), get("left_elbow"), get("left_wrist"))
    angles["right_elbow"] = angle(get("right_shoulder"), get("right_elbow"), get("right_wrist"))
    angles["left_knee"] = angle(get("left_hip"), get("left_knee"), get("left_ankle"))
    angles["right_knee"] = angle(get("right_hip"), get("right_knee"), get("right_ankle"))
    angles["left_shoulder"] = angle(get("left_elbow"), get("left_shoulder"), get("left_hip"))
    angles["right_shoulder"] = angle(get("right_elbow"), get("right_shoulder"), get("right_hip"))

    ls, rs = get("left_shoulder"), get("right_shoulder")
    sd = rs - ls
    angles["shoulder_rotation"] = round(float(np.degrees(np.arctan2(sd[1], sd[0]))), 1)

    lh, rh = get("left_hip"), get("right_hip")
    hd = rh - lh
    angles["hip_rotation"] = round(float(np.degrees(np.arctan2(hd[1], hd[0]))), 1)

    angles["x_factor"] = round(abs(angles["shoulder_rotation"] - angles["hip_rotation"]), 1)

    mid_hip = (lh + rh) / 2
    mid_sh = (ls + rs) / 2
    sv = mid_sh - mid_hip
    angles["spine_tilt"] = round(float(np.degrees(np.arctan2(sv[0], -sv[1]))), 1)

    return angles


def nearest_pose_index_for_video_frame(poses: list[dict], video_frame: int) -> int:
    """Map a video frame index to the closest extracted pose (by pose['frame_index'])."""
    if not poses:
        return 0
    vf = int(video_frame)
    best_k = 0
    best_d = abs(int(poses[0].get("frame_index", 0)) - vf)
    for k in range(1, len(poses)):
        d = abs(int(poses[k].get("frame_index", 0)) - vf)
        if d < best_d:
            best_d = d
            best_k = k
    return best_k


def map_gemini_uniform_indices_to_pose_indices(
    gemini_phases: dict[str, int],
    ai_video_frames: list[int],
    poses: list[dict],
) -> dict[str, int]:
    """Gemini picks thumbnails by index; thumbnails are fixed video frames (linspace), while
    poses are sampled on a different grid (arange step). Map via real frame numbers, then
    enforce strictly increasing pose indices."""
    if not poses or not ai_video_frames:
        return {}
    n_poses = len(poses)
    n_ai = len(ai_video_frames)
    raw: dict[str, int] = {}
    for pid in _PHASE_IDS:
        if pid not in gemini_phases:
            continue
        fidx = max(0, min(int(gemini_phases[pid]), n_ai - 1))
        vf = int(ai_video_frames[fidx])
        raw[pid] = nearest_pose_index_for_video_frame(poses, vf)
    out: dict[str, int] = {}
    prev = -1
    for pid in _PHASE_IDS:
        if pid not in raw:
            continue
        idx = max(int(raw[pid]), prev + 1)
        idx = min(idx, n_poses - 1)
        if idx <= prev:
            idx = min(prev + 1, n_poses - 1)
        out[pid] = idx
        prev = idx
    return out


# ── Phase validation ──

def validate_phase_keyframes(
    phase_keyframes: dict[str, int],
    poses: list[dict],
    source: str = "unknown",
) -> dict:
    """Validate phase keyframe assignments and return a structured report.

    Checks:
      1. Order: phase indices must be strictly increasing
      2. Spacing: adjacent phases must have minimum gap (≥ max(2, n//16))
      3. Top: view-agnostic semantic check (hand excursion / speed / rotation cues)
      4. Impact: semantic check vs top + speed / unwind cues

    Returns dict with:
      - passed: bool
      - order_ok: bool
      - spacing_ok: bool
      - top_reasonable: bool
      - impact_reasonable: bool
      - min_gap: int (smallest gap between adjacent phases)
      - issues: list[str]
      - source: str
    """
    n = len(poses)
    issues: list[str] = []

    # 1. Order check
    prev_idx = -1
    order_ok = True
    for pid in _PHASE_IDS:
        idx = phase_keyframes.get(pid)
        if idx is None:
            issues.append(f"missing_phase:{pid}")
            order_ok = False
            continue
        if idx <= prev_idx:
            issues.append(f"order_violation:{pid}={idx}<=prev={prev_idx}")
            order_ok = False
        prev_idx = idx

    # 2. Spacing check
    min_required_gap = max(2, n // 12)
    indices = [phase_keyframes.get(pid, 0) for pid in _PHASE_IDS if pid in phase_keyframes]
    gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)] if len(indices) > 1 else []
    min_gap = min(gaps) if gaps else 0
    spacing_ok = all(g >= min_required_gap for g in gaps) if gaps else False
    if not spacing_ok:
        tiny_gaps = [(i, g) for i, g in enumerate(gaps) if g < min_required_gap]
        for gi, g in tiny_gaps:
            issues.append(f"tiny_gap:{_PHASE_IDS[gi]}->{_PHASE_IDS[gi+1]}={g}<{min_required_gap}")

    # 3–4. View-agnostic semantic checks (not wrist_y-min = top).
    top_reasonable = True
    impact_reasonable = True
    top_idx = phase_keyframes.get("top")
    impact_idx = phase_keyframes.get("impact")
    ev_once = detect_phase_events_agnostic(poses)
    kin = _build_view_agnostic_kinematics(poses)

    if top_idx is not None and 0 <= top_idx < n and kin is not None:
        top_ok, top_detail = validate_top_semantic_at_index(int(top_idx), kin)
        top_reasonable = top_ok
        if not top_ok:
            issues.append(f"top_semantic_failed:{top_detail}")
    else:
        top_reasonable = False
        issues.append("top_idx_invalid_or_no_kinematics")

    if impact_idx is not None and 0 <= impact_idx < n and kin is not None and top_idx is not None:
        exc_ax = int(ev_once.get("excursion_apex_idx", max(1, n // 4)))
        imp_ok, imp_detail = validate_impact_semantic_at_index(
            int(impact_idx), int(top_idx), exc_ax, kin,
        )
        impact_reasonable = imp_ok
        if not imp_ok:
            issues.append(f"impact_semantic_failed:{imp_detail}")
        if int(impact_idx) <= int(top_idx):
            impact_reasonable = False
            issues.append(f"impact_before_top:{impact_idx}<={top_idx}")
    else:
        impact_reasonable = False
        issues.append("impact_idx_invalid_or_no_kinematics")

    passed = order_ok and spacing_ok and top_reasonable and impact_reasonable

    return {
        "passed": passed,
        "order_ok": order_ok,
        "spacing_ok": spacing_ok,
        "top_reasonable": top_reasonable,
        "impact_reasonable": impact_reasonable,
        "min_gap": min_gap,
        "min_required_gap": min_required_gap,
        "issues": issues,
        "source": source,
    }


EXPECTED_PHASE_STRIP_FRAMES = 8


def assess_gemini_uniform_map_vs_final_phase_strip(
    phase_source: str,
    gemini_mapped: dict | None,
    final_phase_keyframes: dict | None,
    *,
    pose_index_align_tolerance: int = 3,
) -> dict:
    """16-thumbnail Gemini map vs final 8-strip pose indices — exposes divergence (cannot fake alignment).

    When phase_source does not come from Gemini thumbnail picking, ``applies`` is False and
    ``aligned`` is None (no constraint). When it applies, ``aligned`` must be True for premium trust.
    """
    src = str(phase_source or "")
    # gemini / gemini_respaced / any gemini-prefixed source that used 16 uniform thumbnails
    applies = src.startswith("gemini")
    out: dict = {
        "gemini_uniform_thumbnail_map_applies": applies,
        "gemini_map_aligned_with_final_strip": None,
        "per_phase_pose_index_delta": {},
        "phase_mismatches": {},
        "strip_divergence_reason": None,
    }
    if not applies:
        out["aligned"] = None
        return out
    if not gemini_mapped or not final_phase_keyframes:
        out["gemini_map_aligned_with_final_strip"] = False
        out["aligned"] = False
        out["strip_divergence_reason"] = "missing_gemini_map_or_final_phase_keyframes"
        return out
    mismatches: dict = {}
    deltas: dict = {}
    for pid in _PHASE_IDS:
        g = gemini_mapped.get(pid)
        f = final_phase_keyframes.get(pid)
        if g is None or f is None:
            out["gemini_map_aligned_with_final_strip"] = False
            out["aligned"] = False
            out["strip_divergence_reason"] = f"missing_phase:{pid}"
            return out
        gi, fi = int(g), int(f)
        d = abs(gi - fi)
        deltas[pid] = d
        if d > pose_index_align_tolerance:
            mismatches[pid] = {"gemini_pose_idx": gi, "final_pose_idx": fi, "delta": d}
    out["per_phase_pose_index_delta"] = deltas
    if mismatches:
        out["gemini_map_aligned_with_final_strip"] = False
        out["aligned"] = False
        out["strip_divergence_reason"] = "gemini_uniform_map_pose_indices_differ_from_final_strip"
        out["phase_mismatches"] = mismatches
    else:
        out["gemini_map_aligned_with_final_strip"] = True
        out["aligned"] = True
    return out


def compute_phase_evaluations_reliable(
    *,
    final_phase_semantic_ok: bool,
    phase_validation_passed: bool,
    final_keyframe_source: str,
    final_keyframe_gate_pass: bool,
    ai_vision_frame_count: int | None = None,
    keyframe_strip_frame_count: int | None = None,
    gemini_uniform_map_applies: bool = False,
    gemini_map_aligned_with_final_strip: bool | None = None,
) -> bool:
    """True only when 8-strip vision is complete, semantic+validation pass, non-fallback source, and Gemini map matches strip if used."""
    src = str(final_keyframe_source or "")
    if not final_keyframe_gate_pass:
        return False
    if src.startswith("ordered_fallback"):
        return False
    exp = EXPECTED_PHASE_STRIP_FRAMES
    if ai_vision_frame_count is not None and ai_vision_frame_count < exp:
        return False
    if keyframe_strip_frame_count is not None and keyframe_strip_frame_count < exp:
        return False
    if gemini_uniform_map_applies:
        if gemini_map_aligned_with_final_strip is not True:
            return False
    return bool(final_phase_semantic_ok and phase_validation_passed)


def build_phase_boundary_flags(
    *,
    final_keyframe_source: str,
    keyframe_strip_frame_count: int,
    ai_vision_frame_count: int,
    gemini_strip_assessment: dict,
    analysis_route: str,
    plus_grade_phase_evaluations: bool,
) -> dict:
    """Explicit boundary labels for JSON + UI (no pretending fallback is semantic success)."""
    kf_src = str(final_keyframe_source or "")
    exp = EXPECTED_PHASE_STRIP_FRAMES
    complete = keyframe_strip_frame_count >= exp and ai_vision_frame_count >= exp
    if kf_src.startswith("ordered_fallback"):
        label = "monotonic_pose_fallback"
    elif kf_src in ("smart", "smart_repaired"):
        label = "smart_phase_strip"
    else:
        label = "other"
    return {
        "expected_phase_vision_frames": exp,
        "keyframe_strip_frame_count": keyframe_strip_frame_count,
        "ai_vision_frame_count": ai_vision_frame_count,
        "phase_vision_complete_strip": complete,
        "phase_keyframe_extraction_label": label,
        "phase_strip_is_monotonic_fallback_only": kf_src.startswith("ordered_fallback"),
        "gemini_uniform_map_vs_strip": gemini_strip_assessment,
        "analysis_route_tier": analysis_route,
        "plus_grade_phase_evaluations": bool(plus_grade_phase_evaluations),
    }


def neutralize_swing_phase_evaluations_payload(result: dict) -> dict:
    """Force phase-by-phase vision claims off when frames are not semantically trustworthy."""
    if not isinstance(result, dict):
        return result
    out = dict(result)
    if "swing_phase_evaluations" not in out:
        return out
    msg_zh = "阶段图像未通过语义可信性校验；非真实逐阶段视觉结论。"
    msg_en = "unreliable — phase frames not semantically validated."
    out["swing_phase_evaluations"] = [
        {"phase": pid, "status": "unknown", "note_zh": msg_zh, "note_en": msg_en}
        for pid in _PHASE_IDS
    ]
    return out


def respace_phase_keyframes(phase_keyframes: dict[str, int], n_poses: int) -> dict[str, int]:
    """Spread phase pose indices across the timeline with minimum gaps.

    Shared by Plus / Pro / Lite when kinematic or Gemini mapping clusters early poses.

    Blend is intentionally **kinematic-heavy**: a large uniform-grid weight makes the first
    half of the strip look like linspace temporal sampling (user-visible “时间主导”) even
    when the backend already produced motion-based indices.
    """
    if n_poses <= 1:
        return {p: 0 for p in _PHASE_IDS}
    min_gap = max(3, n_poses // 10)
    even = [int(round(i * (n_poses - 1) / 7)) for i in range(8)]
    out: dict[str, int] = {}
    prev = -min_gap
    # 0.15 uniform anchor + 0.85 raw — preserve motion pipeline; grid only breaks worst clumping.
    _ANCHOR_BLEND = 0.15
    _RAW_BLEND = 0.85
    for i, pid in enumerate(_PHASE_IDS):
        raw = phase_keyframes.get(pid)
        anchor = even[i]
        if isinstance(raw, int) and 0 <= raw < n_poses:
            blended = int(round(_ANCHOR_BLEND * anchor + _RAW_BLEND * raw))
        else:
            blended = anchor
        blended = max(0, min(blended, n_poses - 1))
        idx = max(blended, prev + min_gap)
        idx = min(idx, n_poses - 1)
        out[pid] = idx
        prev = idx
    for j in range(6, -1, -1):
        cur, nxt = _PHASE_IDS[j], _PHASE_IDS[j + 1]
        out[cur] = min(out[cur], out[nxt] - min_gap)
        out[cur] = max(out[cur], 0)
    prev = -min_gap
    for pid in _PHASE_IDS:
        out[pid] = max(out[pid], prev + min_gap)
        out[pid] = min(out[pid], n_poses - 1)
        prev = out[pid]
    return out
