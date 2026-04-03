"""
Dynamic weight fusion for three speed estimation signals.

Combines blur-based speed, trajectory-based speed, and formula-based speed
using club-group-dependent base weights, then applies dynamic adjustments
based on per-signal confidence and clamps to a valid range.
"""

import logging

from services.json_sanitize import safe_float

logger = logging.getLogger(__name__)

# ── Base weights per club group ──

BASE_WEIGHTS: dict[str, dict[str, float]] = {
    "WOOD":   {"blur": 0.75, "trajectory": 0.20, "formula": 0.05},
    "IRON":   {"blur": 0.65, "trajectory": 0.25, "formula": 0.10},
    "WEDGE":  {"blur": 0.40, "trajectory": 0.25, "formula": 0.35},
    "PUTTER": {"blur": 0.00, "trajectory": 0.20, "formula": 0.80},
}

# ── Valid speed ranges per club group (mph) ──

SPEED_RANGE: dict[str, tuple[float, float]] = {
    "WOOD":   (140.0, 200.0),
    "IRON":   (80.0, 150.0),
    "WEDGE":  (50.0, 110.0),
    "PUTTER": (5.0, 30.0),
}

# Maps override club_type → club_group (import from club_detector)
from services.club_detector import CLUB_GROUP_MAP  # noqa: E402


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    """Ensure weights sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        return {"blur": 0.0, "trajectory": 0.0, "formula": 1.0}
    return {k: v / total for k, v in weights.items()}


def _compute_confidence(blur_conf: str, traj_conf: str) -> tuple[str, int]:
    """
    Derive overall speed_confidence and error_estimate_pct.

    Rules:
      - blur=high AND trajectory=high → "high", 5%
      - either is medium                → "medium", 8%
      - any is low                      → "low", 15%
    """
    if blur_conf == "high" and traj_conf == "high":
        return "high", 5
    if blur_conf == "low" or traj_conf == "low":
        return "low", 15
    return "medium", 8


def fuse_speed(
    club_group: str,
    blur_result: dict,
    trajectory_result: dict,
    formula_speed: float,
    override_club_type: str | None = None,
) -> dict:
    """
    Fuse three speed signals with dynamic, club-aware weighting.

    Args:
        club_group:         One of WOOD / IRON / WEDGE / PUTTER.
        blur_result:        Output from blur_speed_service.detect_blur_speed().
        trajectory_result:  Output from trajectory_service.track_trajectory().
        formula_speed:      ball_speed from shot_predictor.predict_shot().
        override_club_type: If the user manually selects a different club,
                            this overrides the club_group for weight lookup.

    Returns:
        {
            "fused_speed":        float,
            "fusion_weights":     {"blur": float, "trajectory": float, "formula": float},
            "speed_confidence":   "high" | "medium" | "low",
            "error_estimate_pct": int,
        }
    """
    effective_group = club_group
    if override_club_type:
        mapped = CLUB_GROUP_MAP.get(override_club_type.upper().strip())
        if mapped:
            effective_group = mapped

    if effective_group not in BASE_WEIGHTS:
        effective_group = "IRON"

    formula_speed = safe_float(formula_speed, 0.0)

    w = dict(BASE_WEIGHTS[effective_group])

    # ── Dynamic adjustment 1: low blur confidence ──
    blur_conf = blur_result.get("confidence", "low")
    if blur_conf == "low" and w["blur"] > 0:
        halved = w["blur"] * 0.5
        redistributed = w["blur"] - halved
        w["blur"] = halved
        other_total = w["trajectory"] + w["formula"]
        if other_total > 0:
            w["trajectory"] += redistributed * (w["trajectory"] / other_total)
            w["formula"] += redistributed * (w["formula"] / other_total)
        else:
            w["formula"] += redistributed

    # ── Dynamic adjustment 2: too few tracked frames ──
    tracked_frames = trajectory_result.get("tracked_frames", 0)
    traj_conf = trajectory_result.get("confidence", "low")
    if tracked_frames < 2 and w["trajectory"] > 0:
        halved = w["trajectory"] * 0.5
        redistributed = w["trajectory"] - halved
        w["trajectory"] = halved
        other_total = w["blur"] + w["formula"]
        if other_total > 0:
            w["blur"] += redistributed * (w["blur"] / other_total)
            w["formula"] += redistributed * (w["formula"] / other_total)
        else:
            w["formula"] += redistributed

    # ── Normalize to 1.0 ──
    w = _normalize(w)

    # ── Weighted fusion ──
    blur_speed = safe_float(blur_result.get("ball_speed", 0.0), 0.0)
    traj_speed = safe_float(trajectory_result.get("ball_speed", 0.0), 0.0)

    fused = (
        blur_speed * w["blur"]
        + traj_speed * w["trajectory"]
        + formula_speed * w["formula"]
    )
    fused = safe_float(fused, 0.0)

    # ── Clamp to valid range ──
    lo, hi = SPEED_RANGE.get(effective_group, (50.0, 200.0))
    fused = max(lo, min(hi, fused))

    speed_confidence, error_pct = _compute_confidence(blur_conf, traj_conf)

    result = {
        "fused_speed": round(fused, 1),
        "fusion_weights": {k: round(v, 2) for k, v in w.items()},
        "speed_confidence": speed_confidence,
        "error_estimate_pct": error_pct,
    }

    logger.info(
        "Fusion [%s]: blur=%.1f(%.0f%%) traj=%.1f(%.0f%%) formula=%.1f(%.0f%%) → %.1f mph [%s ±%d%%]",
        effective_group,
        blur_speed, w["blur"] * 100,
        traj_speed, w["trajectory"] * 100,
        formula_speed, w["formula"] * 100,
        fused, speed_confidence, error_pct,
    )

    return result
