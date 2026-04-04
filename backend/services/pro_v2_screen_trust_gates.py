"""Pro v2 Screen Mode — ROI + dense motion health before trusting keyframes."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from services.pro_v2_dense_scan_service import DenseFrame

logger = logging.getLogger(__name__)


def evaluate_screen_roi_health(
    crop_box: dict[str, Any] | None,
    frame_w: int,
    frame_h: int,
    detection_confidence: float,
) -> dict[str, Any]:
    """Heuristic ROI sanity: tiny box, extreme corner, or weak detector → not trustworthy."""
    reason_codes: list[str] = []
    fw = max(1, int(frame_w or 1))
    fh = max(1, int(frame_h or 1))
    if not crop_box or not isinstance(crop_box, dict):
        return {
            "passed": False,
            "reason_codes": ["SCREEN_ROI_MISSING"],
            "area_ratio": 0.0,
            "detection_confidence": float(detection_confidence or 0.0),
        }
    try:
        x = int(crop_box.get("x", 0))
        y = int(crop_box.get("y", 0))
        w = int(crop_box.get("w", 0))
        h = int(crop_box.get("h", 0))
    except (TypeError, ValueError):
        return {"passed": False, "reason_codes": ["SCREEN_ROI_INVALID"], "area_ratio": 0.0, "detection_confidence": 0.0}

    area_ratio = (w * h) / float(fw * fh)
    cx = (x + w / 2.0) / fw
    cy = (y + h / 2.0) / fh
    conf = float(detection_confidence or 0.0)

    if area_ratio < 0.10:
        reason_codes.append("SCREEN_ROI_AREA_TOO_SMALL")
    if area_ratio > 0.97:
        reason_codes.append("SCREEN_ROI_NEAR_FULL_FRAME")
    if cx < 0.10 or cx > 0.90:
        reason_codes.append("SCREEN_ROI_OFF_CENTER_X")
    if cy < 0.10 or cy > 0.90:
        reason_codes.append("SCREEN_ROI_OFF_CENTER_Y")
    if conf < 0.32:
        reason_codes.append("SCREEN_ROI_DETECTOR_LOW_CONF")

    passed = len(reason_codes) == 0
    out = {
        "passed": passed,
        "reason_codes": reason_codes,
        "area_ratio": round(area_ratio, 4),
        "center_norm": [round(cx, 4), round(cy, 4)],
        "detection_confidence": round(conf, 4),
        "crop_box": {"x": x, "y": y, "w": w, "h": h},
        "source_size": {"w": fw, "h": fh},
    }
    logger.info(
        "[PRO_V2][SCREEN_DEBUG] roi_health passed=%s area_ratio=%.4f conf=%.3f reasons=%s box=%s",
        passed,
        area_ratio,
        conf,
        reason_codes,
        out["crop_box"],
    )
    return out


def summarize_dense_for_debug(dense: list[DenseFrame]) -> dict[str, Any]:
    peaks = sum(1 for d in dense if d.is_local_peak)
    valleys = sum(1 for d in dense if d.is_local_valley)
    energies = np.array([d.motion_energy_smooth for d in dense], dtype=np.float64)
    mean_e = float(np.mean(energies)) if len(energies) else 0.0
    std_e = float(np.std(energies)) if len(energies) else 0.0
    cv = std_e / (mean_e + 1e-9)
    return {
        "dense_count": len(dense),
        "motion_peak_count": peaks,
        "motion_valley_count": valleys,
        "motion_energy_mean": round(mean_e, 6),
        "motion_energy_std": round(std_e, 6),
        "motion_coefficient_of_variation": round(cv, 6),
        "motion_energy_max": round(float(np.max(energies)), 6) if len(energies) else 0.0,
    }


def evaluate_dense_motion_health(
    dense: list[DenseFrame],
    swing_t0: float,
    swing_t1: float,
    source_duration_s: float,
    *,
    screen_mode: bool = True,
) -> dict[str, Any]:
    """Short clips + flat motion curves → picker cannot separate real phases."""
    reason_codes: list[str] = []
    summary = summarize_dense_for_debug(dense)
    swing_len = max(0.0, float(swing_t1) - float(swing_t0))
    dur = max(0.01, float(source_duration_s or 0.0))

    peaks = summary["motion_peak_count"]
    valleys = summary["motion_valley_count"]
    cv = summary["motion_coefficient_of_variation"]

    if screen_mode:
        if dur < 5.0 and swing_len < 0.28:
            reason_codes.append("SWING_WINDOW_TOO_SHORT")
        if peaks < 4:
            reason_codes.append("DENSE_PEAKS_TOO_FEW")
        if valleys < 2:
            reason_codes.append("DENSE_VALLEYS_TOO_FEW")
        if summary["dense_count"] < 18:
            reason_codes.append("DENSE_SAMPLE_TOO_SHALLOW")
        if cv < 0.055:
            reason_codes.append("DENSE_MOTION_COLLAPSED")

    passed = len(reason_codes) == 0
    out = {
        "passed": passed,
        "reason_codes": reason_codes,
        "swing_window_s": [round(swing_t0, 4), round(swing_t1, 4)],
        "swing_window_len_s": round(swing_len, 4),
        "source_duration_s": round(dur, 4),
        **summary,
    }
    logger.info(
        "[PRO_V2][DENSE_DEBUG] passed=%s dense_count=%s peaks=%s valleys=%s cv=%.5f swing_len=%.3f dur=%.3f reasons=%s",
        passed,
        summary["dense_count"],
        peaks,
        valleys,
        cv,
        swing_len,
        dur,
        reason_codes,
    )
    return out
