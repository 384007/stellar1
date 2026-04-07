from __future__ import annotations

from typing import Any

from services.json_sanitize import sanitize_json_floats

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
        return {}
    return {k: raw[k] for k in _PREDICTION_PUBLIC_FIELDS if k in raw}


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
    out["prediction"] = _sanitize_prediction(out.get("prediction", {}))

    return sanitize_json_floats(out)
