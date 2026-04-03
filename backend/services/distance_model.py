from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from services.json_sanitize import safe_float


@dataclass(frozen=True)
class ClubBaseline:
    avg_carry: float
    avg_ball_speed: float
    avg_launch_angle: float
    avg_spin_rate: float
    min_carry: float
    max_carry: float


# 14 clubs baseline (+ PT for compatibility with existing flows).
CLUB_BASELINES: dict[str, ClubBaseline] = {
    "1W": ClubBaseline(230.0, 145.0, 12.0, 2600.0, 180.0, 330.0),
    "3W": ClubBaseline(215.0, 138.0, 13.0, 3200.0, 170.0, 295.0),
    "5W": ClubBaseline(200.0, 132.0, 14.0, 3600.0, 160.0, 275.0),
    "3I": ClubBaseline(190.0, 128.0, 12.0, 4100.0, 150.0, 235.0),
    "4I": ClubBaseline(180.0, 124.0, 13.0, 4500.0, 145.0, 225.0),
    "5I": ClubBaseline(170.0, 120.0, 14.0, 5000.0, 135.0, 215.0),
    "6I": ClubBaseline(160.0, 115.0, 15.0, 5600.0, 125.0, 205.0),
    "7I": ClubBaseline(150.0, 110.0, 16.0, 6200.0, 115.0, 195.0),
    "8I": ClubBaseline(140.0, 104.0, 18.0, 7000.0, 105.0, 180.0),
    "9I": ClubBaseline(130.0, 98.0, 20.0, 7800.0, 95.0, 165.0),
    "PW": ClubBaseline(115.0, 90.0, 24.0, 8600.0, 80.0, 150.0),
    "AW": ClubBaseline(105.0, 85.0, 27.0, 9200.0, 70.0, 140.0),
    "SW": ClubBaseline(90.0, 80.0, 31.0, 9800.0, 55.0, 125.0),
    "LW": ClubBaseline(70.0, 72.0, 35.0, 10500.0, 40.0, 110.0),
    "PT": ClubBaseline(15.0, 12.0, 2.0, 200.0, 3.0, 40.0),
}

GROUP_FALLBACK = {
    "WOOD": "3W",
    "IRON": "7I",
    "WEDGE": "PW",
    "PUTTER": "PT",
}


def resolve_baseline(club_type: Optional[str], club_group: Optional[str]) -> tuple[str, ClubBaseline, bool]:
    """Returns (club_key, baseline, assumed). assumed=True when falling back to default."""
    club_key = (club_type or "").upper().strip()
    if club_key in CLUB_BASELINES:
        return club_key, CLUB_BASELINES[club_key], False
    group_key = (club_group or "").upper().strip()
    fallback = GROUP_FALLBACK.get(group_key, "7I")
    return fallback, CLUB_BASELINES[fallback], True


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _lead_trail_map(hand: str) -> dict[str, str]:
    if hand == "L":
        return {"lead": "right", "trail": "left"}
    return {"lead": "left", "trail": "right"}


def compute_distance_prediction(
    baseline: ClubBaseline,
    hand: str,
    peak_x_factor: float,
    rotation_speed: float,
    spine_tilt: float,
    lead_knee: float,
    trail_elbow: float,
    ball_speed: float,
    measured_speed: Optional[float] = None,
    club_assumed: bool = False,
) -> dict:
    """
    baseline × technique_multiplier × strike_multiplier × speed_multiplier
    """
    peak_x_factor = safe_float(peak_x_factor, 35.0)
    rotation_speed = safe_float(rotation_speed, 60.0)
    spine_tilt = safe_float(spine_tilt, 10.0)
    lead_knee = safe_float(lead_knee, 160.0)
    trail_elbow = safe_float(trail_elbow, 140.0)
    ball_speed = safe_float(ball_speed, 0.0)
    ms: float | None = None
    if measured_speed is not None:
        ms = safe_float(measured_speed, 0.0)
        if ms <= 0:
            ms = None
    measured_speed = ms

    effective_speed = measured_speed if measured_speed is not None and measured_speed > 0 else ball_speed

    technique = 1.0
    technique += (_clamp(peak_x_factor, 20.0, 55.0) - 35.0) / 100.0
    technique += (_clamp(rotation_speed, 25.0, 95.0) - 60.0) / 180.0
    technique += (_clamp(spine_tilt, 4.0, 22.0) - 10.0) / 140.0
    technique = _clamp(technique, 0.72, 1.28)

    strike = 1.0
    if 130.0 <= lead_knee <= 170.0:
        strike += 0.03
    elif lead_knee < 120.0:
        strike -= 0.04
    if trail_elbow < 100.0:
        strike += 0.05
    elif trail_elbow > 145.0:
        strike -= 0.05
    strike = _clamp(strike, 0.78, 1.22)

    speed_ratio = effective_speed / max(baseline.avg_ball_speed, 1.0)
    speed_multiplier = _clamp(0.50 + speed_ratio * 0.50, 0.72, 1.30)

    raw_distance = baseline.avg_carry * technique * strike * speed_multiplier
    predicted = _clamp(raw_distance, baseline.min_carry, baseline.max_carry)

    confidence = 0.55
    confidence += 0.15 if hand in ("R", "L") else -0.05
    confidence += 0.10 if measured_speed is not None and measured_speed > 0 else 0.0
    confidence += 0.08 if 125.0 <= lead_knee <= 172.0 else -0.05
    if club_assumed:
        confidence *= 0.6
    confidence = _clamp(confidence, 0.2, 0.95)

    result = {
        "predicted_distance": round(predicted, 1),
        "baseline_distance": round(baseline.avg_carry, 1),
        "technique_multiplier": round(technique, 3),
        "strike_multiplier": round(strike, 3),
        "speed_multiplier": round(speed_multiplier, 3),
        "distance_confidence": round(confidence, 3),
        "distance_debug": {
            "formula": "baseline*technique*strike*speed",
            "effective_ball_speed": round(effective_speed, 2),
            "baseline_ball_speed": round(baseline.avg_ball_speed, 2),
            "baseline_launch_angle": round(baseline.avg_launch_angle, 2),
            "baseline_spin_rate": round(baseline.avg_spin_rate, 1),
            "baseline_min_carry": round(baseline.min_carry, 1),
            "baseline_max_carry": round(baseline.max_carry, 1),
            "club_assumed": club_assumed,
            "inputs": {
                "peak_x_factor": round(peak_x_factor, 2),
                "rotation_speed": round(rotation_speed, 2),
                "spine_tilt": round(spine_tilt, 2),
                "lead_knee": round(lead_knee, 2),
                "trail_elbow": round(trail_elbow, 2),
                "hand": hand,
            },
        },
    }
    if hand not in ("R", "L"):
        result["hand_assumed"] = True
    return result


def resolve_lead_trail_metrics(angles: dict, hand: str) -> dict:
    side = _lead_trail_map(hand)
    lead = side["lead"]
    trail = side["trail"]
    lead_knee = safe_float(angles.get(f"{lead}_knee", 160.0), 160.0)
    trail_elbow = safe_float(angles.get(f"{trail}_elbow", 140.0), 140.0)
    result = {
        "lead_knee": lead_knee,
        "trail_elbow": trail_elbow,
        "lead_side": lead,
        "trail_side": trail,
    }
    if hand not in ("R", "L"):
        result["hand_assumed"] = True
    return result
