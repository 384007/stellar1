import math
from typing import Optional

import numpy as np

from services.json_sanitize import safe_float
from services.distance_model import (
    compute_distance_prediction,
    resolve_baseline,
    resolve_lead_trail_metrics,
)
from services.pose_strict_config import (
    SWEET_SPOT_CONFIDENCE_LOW,
    SWEET_SPOT_VARIANCE_MAX_NORM,
    SWEET_SPOT_WINDOW,
)


CLUB_CHS_SCALE: dict[str, float] = {
    "1W": 1.0,  "3W": 0.94, "5W": 0.89,
    "3I": 0.86, "4I": 0.84, "5I": 0.82, "6I": 0.79,
    "7I": 0.76, "8I": 0.73, "9I": 0.70,
    "PW": 0.66, "AW": 0.63, "SW": 0.60, "LW": 0.55,
    "PT": 0.25,
}
GROUP_CHS_SCALE: dict[str, float] = {
    "WOOD": 0.94, "IRON": 0.76, "WEDGE": 0.63, "PUTTER": 0.25,
}

# Club-dependent launch angle ranges (min, max) in degrees
_LAUNCH_ANGLE_RANGE: dict[str, tuple[float, float]] = {
    "1W": (8.0, 14.0), "3W": (9.0, 16.0), "5W": (10.0, 18.0),
    "3I": (10.0, 18.0), "4I": (11.0, 19.0), "5I": (12.0, 20.0),
    "6I": (13.0, 22.0), "7I": (14.0, 24.0), "8I": (16.0, 26.0),
    "9I": (18.0, 28.0), "PW": (20.0, 32.0), "AW": (22.0, 34.0),
    "SW": (24.0, 36.0), "LW": (26.0, 38.0), "PT": (1.0, 5.0),
}
_GROUP_LAUNCH_RANGE: dict[str, tuple[float, float]] = {
    "WOOD": (8.0, 18.0), "IRON": (12.0, 24.0), "WEDGE": (20.0, 38.0), "PUTTER": (1.0, 5.0),
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _pick_peak(values: list[dict], key: str, default: float = 0.0, abs_mode: bool = False) -> float:
    if not values:
        return default
    arr = [safe_float(v.get(key, default), default) for v in values]
    if abs_mode:
        arr = [abs(v) for v in arr]
    return max(arr) if arr else default


def _pick_min(values: list[dict], key: str, default: float = 180.0) -> float:
    if not values:
        return default
    return min(safe_float(v.get(key, default), default) for v in values)


def calibrate_prediction(
    prediction: dict,
    club_type: Optional[str] = None,
    club_group: Optional[str] = None,
    preferred_ball_speed: Optional[float] = None,
) -> dict:
    """
    Recompute predicted_distance using the same backend dynamic distance engine.
    """
    out = dict(prediction or {})
    hand = str(out.get("hand") or "UNKNOWN")
    angles = out.get("distance_debug", {}).get("inputs", {}) if isinstance(out.get("distance_debug"), dict) else {}

    peak_x_factor = safe_float(angles.get("peak_x_factor", 35.0), 35.0)
    rotation_speed = safe_float(angles.get("rotation_speed", 55.0), 55.0)
    spine_tilt = safe_float(angles.get("spine_tilt", 8.0), 8.0)
    lead_knee = safe_float(angles.get("lead_knee", 160.0), 160.0)
    trail_elbow = safe_float(angles.get("trail_elbow", 140.0), 140.0)
    ball_speed = safe_float(out.get("ball_speed", 0.0), 0.0)
    measured_speed = None
    if preferred_ball_speed is not None:
        ms = safe_float(preferred_ball_speed, 0.0)
        measured_speed = ms if ms > 0 else None

    _, baseline, _assumed = resolve_baseline(club_type, club_group)
    dist = compute_distance_prediction(
        baseline=baseline,
        hand=hand,
        peak_x_factor=peak_x_factor,
        rotation_speed=rotation_speed,
        spine_tilt=spine_tilt,
        lead_knee=lead_knee,
        trail_elbow=trail_elbow,
        ball_speed=ball_speed,
        measured_speed=measured_speed,
    )
    out.update(dist)
    return out


def _joint_norm_vis(pose: dict, name: str) -> tuple[float, float, float]:
    for j in pose.get("joints") or []:
        if j.get("name") == name:
            n = j.get("normalized") or {}
            return (
                float(n.get("x", 0.5)),
                float(n.get("y", 0.5)),
                float(j.get("visibility", 0.0)),
            )
    return 0.5, 0.5, 0.0


def _club_head_proxy_norm(pose: dict, hand: str) -> tuple[float, float, float]:
    """Wrist + forearm extension as club-head proxy (normalized image coords)."""
    if hand == "L":
        wx, wy, wv = _joint_norm_vis(pose, "right_wrist")
        ex, ey, ev = _joint_norm_vis(pose, "right_elbow")
    else:
        wx, wy, wv = _joint_norm_vis(pose, "left_wrist")
        ex, ey, ev = _joint_norm_vis(pose, "left_elbow")
    if wv < 0.18 and ev < 0.18:
        lx, ly, lv = _joint_norm_vis(pose, "left_wrist")
        rx, ry, rv = _joint_norm_vis(pose, "right_wrist")
        if lv + rv > 1e-6:
            return (lx + rx) / 2.0, (ly + ry) / 2.0, (lv + rv) / 2.0
        return wx, wy, max(wv, 0.05)
    dx, dy = wx - ex, wy - ey
    norm = math.hypot(dx, dy) + 1e-9
    ext = 0.085
    hx = wx + (dx / norm) * ext
    hy = wy + (dy / norm) * ext
    vis = min(1.0, (wv + ev) / 2.0)
    return hx, hy, vis


def estimate_sweet_spot_robust(
    poses: list[dict],
    impact_pose_idx: int,
    *,
    window: int = SWEET_SPOT_WINDOW,
    hand: str = "UNKNOWN",
) -> dict:
    """
    Median fusion of club-head proxy over ``impact_pose_idx ± window`` (default from config).
    Low-visibility frames are skipped; position is **median** of valid (x,y), never a single-frame pick
    when two or more valid samples exist.
    """
    reasons: list[str] = []
    n = len(poses)
    if n < 1:
        return {
            "sweet_spot": {"nx": 0.5, "ny": 0.5},
            "sweet_spot_confidence": 0.0,
            "sweet_spot_window_size": 0,
            "sweet_spot_valid_frames": 0,
            "sweet_spot_unstable": True,
            "sweet_spot_reasons": ["SWEET_SPOT_UNSTABLE"],
        }
    ic = int(max(0, min(n - 1, impact_pose_idx)))
    i0 = max(0, ic - window)
    i1 = min(n, ic + window + 1)
    pts: list[tuple[float, float]] = []
    for i in range(i0, i1):
        px, py, vv = _club_head_proxy_norm(poses[i], hand)
        if vv >= 0.22:
            pts.append((px, py))
    valid = len(pts)
    win_sz = i1 - i0
    if valid == 0:
        reasons.append("SWEET_SPOT_UNSTABLE")
        conf = 0.25
        mx, my = 0.5, 0.5
        unstable = True
    elif valid == 1:
        # Usable point but under-fused: low confidence, soft reason (not API hard-fail).
        mx = float(pts[0][0])
        my = float(pts[0][1])
        reasons.append("SWEET_SPOT_LOW_VALID_FRAMES")
        conf = min(0.32, SWEET_SPOT_CONFIDENCE_LOW - 0.03)
        unstable = False
    else:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx = float(np.median(xs))
        my = float(np.median(ys))
        var = float(np.var(xs) + np.var(ys))
        conf = 0.55 + 0.35 * min(1.0, valid / max(win_sz, 1))
        conf *= max(0.2, 1.0 - min(1.0, var / max(SWEET_SPOT_VARIANCE_MAX_NORM, 1e-9)))
        conf = float(np.clip(conf, 0.0, 1.0))
        if var > SWEET_SPOT_VARIANCE_MAX_NORM:
            reasons.append("SWEET_SPOT_UNSTABLE")
            conf = min(conf, SWEET_SPOT_CONFIDENCE_LOW - 0.02)
        unstable = bool("SWEET_SPOT_UNSTABLE" in reasons)
    if unstable and conf > SWEET_SPOT_CONFIDENCE_LOW:
        conf = min(conf, SWEET_SPOT_CONFIDENCE_LOW - 0.02)
    return {
        "sweet_spot": {"nx": round(mx, 4), "ny": round(my, 4)},
        "sweet_spot_confidence": round(float(conf), 3),
        "sweet_spot_window_size": int(win_sz),
        "sweet_spot_valid_frames": int(valid),
        "sweet_spot_unstable": unstable,
        "sweet_spot_reasons": reasons,
    }


def predict_shot(
    pose_data: dict,
    swing_duration: float = 1.2,
    all_frame_angles: list[dict] | None = None,
    club_type: Optional[str] = None,
    club_group: Optional[str] = None,
    hand: str = "UNKNOWN",
    hand_confidence: float = 0.0,
    preferred_ball_speed: Optional[float] = None,
    poses: list[dict] | None = None,
    impact_pose_idx: int | None = None,
) -> dict:
    """
    Unified prediction:
    - Dynamic baseline distance model (backend source of truth)
    - Hand-aware lead/trail mapping
    """
    angles = pose_data.get("angles", {}) or {}
    frame_angles = all_frame_angles or [angles]

    peak_x_factor = _pick_peak(frame_angles, "x_factor", 35.0)
    peak_shoulder = _pick_peak(frame_angles, "shoulder_rotation", 15.0, abs_mode=True)
    spine_tilt = abs(safe_float(angles.get("spine_tilt", 8.0), 8.0))
    rotation_speed = peak_shoulder / max(safe_float(swing_duration, 1.2), 0.3)
    if not math.isfinite(rotation_speed):
        rotation_speed = 60.0

    lt = resolve_lead_trail_metrics(angles, hand)
    lead_knee = lt["lead_knee"]
    trail_elbow = lt["trail_elbow"]

    # Speed model stays simple but uses lead/trail roles (no right-hand bias).
    speed_bonus = 0.0
    speed_bonus += min(peak_x_factor / 45.0, 1.0) * 22.0
    if 130.0 <= lead_knee <= 170.0:
        speed_bonus += 6.0
    if trail_elbow < 95.0:
        speed_bonus += 10.0
    elif trail_elbow < 120.0:
        speed_bonus += 6.0
    elif trail_elbow < 140.0:
        speed_bonus += 3.0
    speed_bonus += min(rotation_speed / 80.0, 1.0) * 14.0

    raw_chs = _clamp(65.0 + speed_bonus, 55.0, 125.0)
    chs_scale = 1.0
    ct = (club_type or "").upper().strip()
    cg = (club_group or "").upper().strip()
    if ct in CLUB_CHS_SCALE:
        chs_scale = CLUB_CHS_SCALE[ct]
    elif cg in GROUP_CHS_SCALE:
        chs_scale = GROUP_CHS_SCALE[cg]
    club_head_speed = round(raw_chs * chs_scale, 1)

    smash_factor = _clamp(1.45 + (min(peak_x_factor, 50.0) / 50.0) * 0.05, 1.35, 1.52)
    ball_speed = club_head_speed * smash_factor
    # Club-dependent launch angle range
    la_lo, la_hi = 8.0, 24.0
    if ct in _LAUNCH_ANGLE_RANGE:
        la_lo, la_hi = _LAUNCH_ANGLE_RANGE[ct]
    elif cg in _GROUP_LAUNCH_RANGE:
        la_lo, la_hi = _GROUP_LAUNCH_RANGE[cg]
    launch_angle_deg = _clamp(12.0 + spine_tilt * 0.3, la_lo, la_hi)
    launch_angle_rad = math.radians(launch_angle_deg)
    spin_rate = max(2000.0, 2800.0 - (club_head_speed - 60.0) * 15.0)

    baseline_key, baseline, baseline_assumed = resolve_baseline(club_type, club_group)
    dist = compute_distance_prediction(
        baseline=baseline,
        hand=hand,
        peak_x_factor=peak_x_factor,
        rotation_speed=rotation_speed,
        spine_tilt=spine_tilt,
        lead_knee=lead_knee,
        trail_elbow=trail_elbow,
        ball_speed=ball_speed,
        measured_speed=preferred_ball_speed,
        club_assumed=baseline_assumed,
    )

    lateral_offset = 0.0
    shoulder_diff = safe_float(angles.get("shoulder_rotation", 0.0), 0.0)
    if abs(shoulder_diff) > 5:
        lateral_offset = shoulder_diff * 0.3
    if abs(lateral_offset) < 3:
        lateral_offset += np.random.uniform(-2, 2)
    lateral_offset = round(lateral_offset, 1)

    if abs(lateral_offset) < 3:
        shot_shape, shot_shape_zh = "Straight", "直球"
    elif lateral_offset > 0:
        shot_shape, shot_shape_zh = ("Slice", "右曲球") if lateral_offset > 10 else ("Fade", "轻微右曲")
    else:
        shot_shape, shot_shape_zh = ("Hook", "左曲球") if lateral_offset < -10 else ("Draw", "轻微左曲")

    _spd_src = (
        preferred_ball_speed
        if preferred_ball_speed is not None and safe_float(preferred_ball_speed, 0.0) > 0
        else ball_speed
    )
    ball_speed_ms = safe_float(_spd_src, 0.0) * 0.44704
    trajectory = _compute_trajectory(
        ball_speed_ms,
        launch_angle_rad,
        safe_float(dist.get("predicted_distance"), 0.0),
        lateral_offset,
    )

    distance_debug = dict(dist.get("distance_debug", {}))
    distance_debug["inputs"] = {
        **distance_debug.get("inputs", {}),
        "peak_x_factor": round(peak_x_factor, 2),
        "rotation_speed": round(rotation_speed, 2),
        "spine_tilt": round(spine_tilt, 2),
        "lead_knee": round(lead_knee, 2),
        "trail_elbow": round(trail_elbow, 2),
        "lead_side": lt["lead_side"],
        "trail_side": lt["trail_side"],
    }
    distance_debug["baseline_key"] = baseline_key

    result = {
        "predicted_distance": dist["predicted_distance"],
        "lateral_offset": lateral_offset,
        "shot_shape": shot_shape,
        "shot_shape_zh": shot_shape_zh,
        "club_head_speed": round(club_head_speed, 1),
        "ball_speed": round(ball_speed, 1),
        "launch_angle": round(launch_angle_deg, 1),
        "spin_rate": round(spin_rate),
        "smash_factor": round(smash_factor, 2),
        "trajectory": trajectory,
        "hand": hand if hand in ("R", "L") else "UNKNOWN",
        "hand_confidence": round(safe_float(hand_confidence, 0.0), 3),
        "baseline_distance": dist["baseline_distance"],
        "technique_multiplier": dist["technique_multiplier"],
        "strike_multiplier": dist["strike_multiplier"],
        "speed_multiplier": dist["speed_multiplier"],
        "distance_confidence": dist["distance_confidence"],
        "distance_debug": distance_debug,
    }

    # Surface assumption flags so callers know when values are guessed
    if hand not in ("R", "L"):
        result["hand_assumed"] = "R"
        result["hand_warning"] = "Handedness unknown, assuming right-handed"
    if not ct and baseline_key != (club_type or "").upper().strip():
        result["club_assumed"] = baseline_key
        result["club_warning"] = f"Club unknown, using {baseline_key} baseline"

    if poses is not None and impact_pose_idx is not None:
        ssb = estimate_sweet_spot_robust(poses, int(impact_pose_idx), hand=hand)
        result["sweet_spot"] = ssb.get("sweet_spot")
        result["sweet_spot_confidence"] = ssb.get("sweet_spot_confidence")
        result["sweet_spot_window_size"] = ssb.get("sweet_spot_window_size")
        result["sweet_spot_valid_frames"] = ssb.get("sweet_spot_valid_frames")
        result["sweet_spot_unstable"] = ssb.get("sweet_spot_unstable")
        if ssb.get("sweet_spot_reasons"):
            result["sweet_spot_reasons"] = ssb["sweet_spot_reasons"]
        if ssb.get("sweet_spot_unstable") or safe_float(ssb.get("sweet_spot_confidence"), 1.0) < SWEET_SPOT_CONFIDENCE_LOW:
            sm = safe_float(result.get("strike_multiplier"), 1.0)
            result["strike_multiplier"] = round(sm * 0.92, 4)

    return result


def _compute_trajectory(
    ball_speed_ms: float,
    launch_angle_rad: float,
    carry_yards: float,
    lateral_offset: float,
    num_points: int = 50,
) -> list[dict]:
    """Compute 2D trajectory points for animation (bird's eye + side view)."""
    ball_speed_ms = safe_float(ball_speed_ms, 0.0)
    launch_angle_rad = safe_float(launch_angle_rad, 0.0)
    carry_yards = safe_float(carry_yards, 0.0)
    lateral_offset = safe_float(lateral_offset, 0.0)
    if not math.isfinite(launch_angle_rad):
        launch_angle_rad = math.radians(12.0)
    carry_meters = max(0.0, carry_yards) * 0.9144
    sin_la = math.sin(launch_angle_rad)
    if not math.isfinite(sin_la):
        sin_la = 0.0
    total_time = 2.0 * ball_speed_ms * sin_la / 9.81
    if not math.isfinite(total_time) or total_time <= 0:
        total_time = 1.0

    points = []
    for i in range(num_points + 1):
        t_frac = i / num_points
        t = t_frac * total_time

        vx = ball_speed_ms * math.cos(launch_angle_rad)
        vy = ball_speed_ms * math.sin(launch_angle_rad)

        x = vx * t * 0.75
        y = vy * t - 0.5 * 9.81 * t * t
        y = max(y, 0)

        lateral = lateral_offset * 0.9144 * t_frac

        x_yards = round(x * 1.09361, 1)
        y_yards = round(y * 1.09361, 1)
        lateral_yards = round(lateral, 1)

        points.append(
            {
                "t": round(t_frac, 3),
                "x": x_yards,
                "y": y_yards,
                "lateral": lateral_yards,
            }
        )

    return points
