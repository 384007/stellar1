from __future__ import annotations

from typing import Any

from services.json_sanitize import sanitize_json_floats

_LITE_RELIABILITY_REASON_PUBLIC: dict[str, str] = {
    "phase_validation_soft_fail": "Swing phase alignment may be less certain.",
    "hand_unknown": "Handedness was not detected clearly.",
    "club_unknown": "Club type was not detected clearly.",
    "tracking_weak": "Body tracking covered fewer frames than ideal.",
    "phase_vision_unreliable": "Some swing images had limited clarity.",
    "diagnosis_corrected": "Summary was adjusted for consistency.",
    "sweet_spot_unstable": "Strike timing signal was unstable.",
    "club_assumed_7i": "A default club was assumed for distance estimates.",
    "club_assumed_7I": "A default club was assumed for distance estimates.",
    "limited_data": "Motion data was limited for this clip.",
    "lite_trust_medium": "Evidence quality is moderate for this clip; results are still useful but less certain.",
}

_ALLOWED_TOP_LEVEL_FIELDS = {
    "analysis_id",
    "type",
    "analysis_mode",
    "keyframes",
    "summary",
    "summary_zh",
    "issues",
    "issues_zh",
    "suggestions",
    "suggestions_zh",
    "scores",
    "total_score",
    "analysis_reliability",
    "prediction",
}

_KEYFRAME_PUBLIC_FIELDS = frozenset(
    {"phase", "label_en", "label_zh", "timestamp", "image_base64", "keyframe_image_url", "frame_index"}
)

_PREDICTION_PUBLIC_FIELDS = frozenset(
    {
        "predicted_distance",
        "lateral_offset",
        "shot_shape",
        "shot_shape_zh",
        "club_head_speed",
        "ball_speed",
        "launch_angle",
        "spin_rate",
        "smash_factor",
        "trajectory",
        "hand",
        "hand_confidence",
        "club_type",
        "club_group",
        "club_detection_confidence",
        "speed_confidence",
        "distance_confidence",
        "error_estimate_pct",
        "blur_speed",
        "trajectory_speed",
        "fused_speed",
        "fusion_weights",
        "trajectory_tracked_frames",
        "trajectory_confidence",
        "blur_confidence",
    }
)


def _sanitize_keyframes(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        row = {k: item[k] for k in _KEYFRAME_PUBLIC_FIELDS if k in item}
        out.append(row)
    return out


def _sanitize_prediction(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        out: dict[str, Any] = {}
    else:
        out = {k: raw[k] for k in _PREDICTION_PUBLIC_FIELDS if k in raw}
    out.setdefault("hand", "UNKNOWN")
    out.setdefault("hand_confidence", 0.0)
    out.setdefault("club_type", "UNKNOWN")
    out.setdefault("club_group", "IRON")
    out.setdefault("club_detection_confidence", 0.0)
    return out


def _lite_public_reason_line(reason_key: str) -> str:
    if reason_key in _LITE_RELIABILITY_REASON_PUBLIC:
        return _LITE_RELIABILITY_REASON_PUBLIC[reason_key]
    if "_" in reason_key and reason_key == reason_key.lower():
        return "Some quality signals were limited for this analysis."
    return reason_key


def _sanitize_lite_analysis_reliability(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"level": "low", "capped_confidence": 35, "reasons": ["Limited data was available for this analysis."]}
    out = {k: raw[k] for k in ("level", "original_confidence", "capped_confidence", "penalty") if k in raw}
    reasons_in = raw.get("reasons")
    seen: set[str] = set()
    pub_reasons: list[str] = []
    if isinstance(reasons_in, list):
        for r in reasons_in:
            if not isinstance(r, str) or not r.strip():
                continue
            msg = _lite_public_reason_line(r.strip())
            if msg not in seen:
                seen.add(msg)
                pub_reasons.append(msg)
    if not pub_reasons:
        pub_reasons = ["Limited data was available for this analysis."]
    out["reasons"] = pub_reasons
    pv = raw.get("phase_validation")
    if isinstance(pv, dict):
        out["phase_validation"] = {"passed": bool(pv.get("passed", True))}
    elif pv is not None:
        out["phase_validation"] = {"passed": True}
    return out


def pack_lite_public_response(internal_result: dict[str, Any]) -> dict[str, Any]:
    """Whitelist-only lite response. Drops internal/debug/engine fields."""
    out: dict[str, Any] = {}
    for key in _ALLOWED_TOP_LEVEL_FIELDS:
        if key not in internal_result:
            continue
        if key == "keyframes":
            out[key] = _sanitize_keyframes(internal_result[key])
        elif key == "prediction":
            out[key] = _sanitize_prediction(internal_result[key])
        elif key == "analysis_reliability":
            out[key] = _sanitize_lite_analysis_reliability(internal_result[key])
        else:
            out[key] = internal_result[key]

    out["type"] = "lite"
    out.setdefault("analysis_mode", "standard")
    out.setdefault("keyframes", [])
    out.setdefault("summary", "")
    out.setdefault("summary_zh", "")
    out.setdefault("issues", [])
    out.setdefault("issues_zh", [])
    out.setdefault("suggestions", [])
    out.setdefault("suggestions_zh", [])
    out.setdefault("scores", {})
    out.setdefault("total_score", 0)
    out.setdefault("analysis_reliability", {"level": "low", "capped_confidence": 35, "reasons": ["limited_data"]})
    out["analysis_reliability"] = _sanitize_lite_analysis_reliability(out.get("analysis_reliability"))
    out["prediction"] = _sanitize_prediction(out.get("prediction", {}))

    return sanitize_json_floats(out)
