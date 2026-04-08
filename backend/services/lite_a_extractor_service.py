"""Lite A extractor: primary 8 keyframe rows + formal hand/club checks (no prov3)."""

from __future__ import annotations

import logging
import statistics
from typing import Any

from services.handedness_service import detect_handedness
from services.lite_keyframe_candidate_a import lite_build_candidate_a_rows
from services.lite_keyframe_constants import LITE_EVENT_SEQUENCE
from services.lite_keyframe_heuristic import (
    lite_enforce_monotonic_frame_indices,
    lite_refine_impact_row,
)
from services.lite_timeline_motion import lite_impact_hint_from_timeline, lite_motion_along_timeline

logger = logging.getLogger(__name__)
_LOG = "[lite_a]"

# Keyframe geometry (A path — strict)
_A_PASS_MIN_AVG_CONFIDENCE = 0.62
_A_MIN_CORE_FRAME_CONFIDENCE = 0.55
_A_CORE_EVENTS = frozenset({"Top", "Impact"})
_A_TOP_IMPACT_MIN_GAP = 4
_A_MAX_IMPACT_HINT_DEV = 38

# Formal A pass: geometry + identity signals
_MIN_HAND_CONF = 0.30
_MIN_CLUB_CONF = 0.40


def _kf_validate_event_order(keyframes: list[dict[str, Any]]) -> bool:
    event_to_idx = {str(item.get("event_name")): int(item.get("frame_index", -1)) for item in keyframes}
    ordered = [event_to_idx.get(name, -1) for name in LITE_EVENT_SEQUENCE]
    if any(idx < 0 for idx in ordered):
        return False
    return all(left < right for left, right in zip(ordered, ordered[1:]))


def _kf_core_gap_ok(keyframes: list[dict[str, Any]], min_gap: int) -> bool:
    event_to_idx = {str(item.get("event_name")): int(item.get("frame_index", -1)) for item in keyframes}
    top_idx = event_to_idx.get("Top", -1)
    impact_idx = event_to_idx.get("Impact", -1)
    return top_idx >= 0 and impact_idx >= 0 and (impact_idx - top_idx) >= min_gap


def _kf_avg_confidence(keyframes: list[dict[str, Any]]) -> float:
    values = [float(item.get("confidence", 0.0)) for item in keyframes]
    return float(statistics.mean(values)) if values else 0.0


def _kf_impact_index(keyframes: list[dict[str, Any]]) -> int:
    for item in keyframes:
        if str(item.get("event_name")) == "Impact":
            return int(item.get("frame_index", -1))
    return -1


def _lite_a_keyframe_fail_reasons(
    keyframes: list[dict[str, Any]],
    *,
    impact_hint_frame_index: int,
) -> list[str]:
    reasons: list[str] = []
    if not _kf_validate_event_order(keyframes):
        reasons.append("event_order_invalid")
    if not _kf_core_gap_ok(keyframes, _A_TOP_IMPACT_MIN_GAP):
        reasons.append("top_impact_gap_invalid")
    avg_c = _kf_avg_confidence(keyframes)
    if avg_c < _A_PASS_MIN_AVG_CONFIDENCE:
        reasons.append("low_overall_confidence")
    event_conf = {str(item.get("event_name")): float(item.get("confidence", 0.0)) for item in keyframes}
    for core_event in _A_CORE_EVENTS:
        if event_conf.get(core_event, 0.0) < _A_MIN_CORE_FRAME_CONFIDENCE:
            reasons.append(f"{core_event.lower()}_confidence_low")
    if any(float(item.get("confidence", 0.0)) < 0.35 for item in keyframes):
        reasons.append("possible_club_visibility_issue")
    imp = _kf_impact_index(keyframes)
    if imp >= 0 and abs(imp - int(impact_hint_frame_index)) > _A_MAX_IMPACT_HINT_DEV:
        reasons.append("impact_hint_mismatch")
    return reasons


async def _club_from_previews(frames: list[Any], region: str) -> dict[str, Any]:
    from services.club_detector import detect_club

    results: list[dict[str, Any]] = []
    for f in frames:
        try:
            r = await detect_club(f, region)
            results.append(dict(r))
        except Exception as exc:
            logger.warning("%s club frame failed: %s", _LOG, exc)
    valid = [r for r in results if str(r.get("club_type") or "").upper() not in ("", "UNKNOWN")]
    if not valid:
        return {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0}
    votes: dict[str, dict[str, Any]] = {}
    for r in valid:
        ct = str(r.get("club_type") or "").upper()
        cg = str(r.get("club_group") or "IRON").upper()
        conf = float(r.get("confidence") or 0.0)
        if ct not in votes:
            votes[ct] = {"count": 0, "total_conf": 0.0, "group": cg}
        votes[ct]["count"] += 1
        votes[ct]["total_conf"] += conf
    winner = max(votes.items(), key=lambda x: (x[1]["count"], x[1]["total_conf"]))[0]
    agg = votes[winner]
    avg_conf = agg["total_conf"] / max(agg["count"], 1)
    return {
        "club_type": winner,
        "club_group": str(agg.get("group") or "IRON"),
        "confidence": round(min(1.0, avg_conf), 4),
    }


async def run_lite_a_extract(preprocess: dict[str, Any], *, region: str) -> dict[str, Any]:
    """
    Build A rows on preprocess timeline; run keyframe checks + handedness + club as one A pass.
    Returns internal fail_reasons for logs only.
    """
    analysis_path = str(preprocess["analysis_video_path"])
    vfps = float(preprocess["analysis_fps"])
    total_frames = int(preprocess["total_frames"])
    duration_s = float(preprocess["duration_s"])
    timeline = list(preprocess["timeline"])
    poses = list(preprocess["poses"])
    preview_bgr = list(preprocess["preview_bgr"])

    if len(timeline) < 8:
        raise RuntimeError("lite_timeline_too_short")

    indices = [int(t["frame_index"]) for t in timeline]
    motions = lite_motion_along_timeline(analysis_path, indices)
    preloc = lite_impact_hint_from_timeline(indices, motions, vfps, duration_s)
    hint_fi = int(round(float(preloc.get("impact_hint_s") or 0.0) * vfps))
    max_fi = max(0, total_frames - 1)

    rows0 = lite_build_candidate_a_rows(indices, motions)
    rows0 = lite_refine_impact_row(rows0, hint_fi)
    rows = lite_enforce_monotonic_frame_indices(rows0, max_fi)

    kf_reasons = _lite_a_keyframe_fail_reasons(rows, impact_hint_frame_index=hint_fi)
    kf_pass = len(kf_reasons) == 0

    hand_info = detect_handedness(poses, None) if poses else {"hand": "UNKNOWN", "confidence": 0.0}
    hand = str(hand_info.get("hand") or "UNKNOWN")
    hconf = float(hand_info.get("confidence") or 0.0)
    hand_ok = hand != "UNKNOWN" and hconf >= _MIN_HAND_CONF

    club_info = await _club_from_previews(preview_bgr, region)
    ct = str(club_info.get("club_type") or "UNKNOWN").upper()
    cconf = float(club_info.get("confidence") or 0.0)
    club_ok = ct != "UNKNOWN" and cconf >= _MIN_CLUB_CONF

    fail_reasons = list(kf_reasons)
    if not hand_ok:
        fail_reasons.append("hand_check_fail")
    if not club_ok:
        fail_reasons.append("club_check_fail")

    a_pass = kf_pass and hand_ok and club_ok

    logger.info(
        "%s extract kf_pass=%s hand_ok=%s club_ok=%s a_pass=%s reasons=%s",
        _LOG,
        kf_pass,
        hand_ok,
        club_ok,
        a_pass,
        fail_reasons,
    )

    return {
        "rows": rows,
        "a_pass": a_pass,
        "fail_reasons": fail_reasons,
        "kf_pass": kf_pass,
        "kf_reasons": kf_reasons,
        "hand": hand,
        "hand_info": hand_info,
        "hand_ok": hand_ok,
        "club_info": club_info,
        "indices": indices,
        "motions": motions,
        "impact_hint_frame_index": hint_fi,
        "max_fi": max_fi,
        "preloc": preloc,
    }
