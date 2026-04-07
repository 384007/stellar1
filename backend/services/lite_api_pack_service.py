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
    "quality_warning",
    "keyframe_warning",
    "hand_warning",
    "club_warning",
}


def pack_lite_public_response(internal_result: dict[str, Any]) -> dict[str, Any]:
    """Whitelist-only lite response. Any internal/debug field is dropped."""
    out: dict[str, Any] = {}
    for key in _ALLOWED_TOP_LEVEL_FIELDS:
        if key in internal_result:
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

    return sanitize_json_floats(out)
