"""Pack internal analysis artifacts into product-safe API payloads."""

from __future__ import annotations

from typing import Any

_FORBIDDEN_TOKENS = [
    "mediapipe", "blazepose", "yolo", "ultralytics", "bytetrack", "mmaction", "mmpose", "rtmpose",
    "deeplabcut", "motionbert", "backend_model", "track_backend", "phase_backend", "provider",
]

# Never forward to the client (internal / oversized debug / provider-identifying fields).
_PACK_EXCLUDE_KEYS: frozenset[str] = frozenset(
    {
        "provider_debug",
        "optional_modules",
        "debug_keyframes",
        "debug_phase_keyframes",
        "ai_key",
        "phase_debug",
        "yolo11_degraded",
        "yolo11_status",
        "unreliable_debug_only",
    }
)



def _contains_forbidden_text(v: str) -> bool:
    lv = (v or "").lower()
    return any(tok in lv for tok in _FORBIDDEN_TOKENS)


# Base64 blobs are pseudo-random ASCII; substring checks would false-positive and wipe JPEGs.
_SCRUB_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "image_base64",
        "skeleton_overlay",
        "segmentation_mask",
    }
)


def _scrub_walk(v: Any, key: str | None) -> Any:
    if isinstance(v, str):
        if key and (key in _SCRUB_SKIP_KEYS or str(key).endswith("_base64")):
            return v
        return "" if _contains_forbidden_text(v) else v
    if isinstance(v, list):
        return [_scrub_walk(x, key) for x in v]
    if isinstance(v, dict):
        return {k: _scrub_walk(val, k) for k, val in v.items()}
    return v


def _scrub_text(v: Any) -> Any:
    return _scrub_walk(v, None)


def pack_plus_response(raw: dict[str, Any]) -> dict[str, Any]:
    # Product-only response contract.
    keyframes = list(raw.get("keyframes", []) or [])
    image_missing = any(not str(k.get("image_base64") or "").strip() for k in keyframes)
    final_gate_pass = bool(raw.get("final_keyframe_gate_pass"))
    final_source = str(raw.get("final_keyframe_source") or "")
    degraded = bool(raw.get("keyframes_degraded", not final_gate_pass))
    partial_mode = bool(raw.get("partial_mode")) or (not final_gate_pass) or degraded or final_source == "smart_gate_failed"
    pipeline_degraded = bool(raw.get("plus_pipeline_degraded"))
    degraded_flags = list(raw.get("plus_degraded_flags") or [])
    out = {
        "analysis_id": raw.get("analysis_id"),
        "type": raw.get("type", "plus"),
        "analysis_mode": raw.get("analysis_mode"),
        "plus_pipeline_degraded": pipeline_degraded,
        "plus_degraded_flags": degraded_flags,
        "biomech_hard_passed": raw.get("biomech_hard_passed"),
        "keyframes": keyframes,
        "phase_keyframes": raw.get("phase_keyframes", {}),
        "prediction": raw.get("prediction", {}),
        "trajectory": raw.get("trajectory", {}),
        "report_status": raw.get("report_status"),
        "report_error_code": raw.get("report_error_code"),
        "score_pack_blocked_reason": raw.get("score_pack_blocked_reason"),
        "report_pack_blocked_reason": raw.get("report_pack_blocked_reason"),
        "final_ui_safe_score_state": raw.get("final_ui_safe_score_state"),
        "fallback_rebuild_material_change": raw.get("fallback_rebuild_material_change"),
        "scores": raw["scores"] if "scores" in raw else {},
        "issues": raw.get("issues", []),
        "training": raw.get("training", {}),
        "analysis_reliability": raw.get("analysis_reliability", {}),
        "gemini_observation": raw.get("gemini_observation"),
        "final_keyframe_gate_pass": final_gate_pass,
        "final_keyframe_source": final_source,
        "keyframes_degraded": degraded,
        "keyframe_display_mode": raw.get("keyframe_display_mode", "degraded_failed"),
        "phase_source": raw.get("phase_source"),
        "phase_validation": raw.get("phase_validation", {}),
        "phase_repair_failed": bool(raw.get("phase_repair_failed")),
        "partial_mode": partial_mode,
        "result_partial": partial_mode,
        "image_missing": bool(image_missing),
        "result_source": "r2",
        "keyframe_warning": raw.get("keyframe_warning"),
    }
    # Merge the rest of the product-safe analysis payload. Previously only a tiny subset was
    # returned, which stripped pose_frames / skeleton_data / video_meta / diagnosis text and
    # broke video overlay (skeleton, phases, sweet spot, trajectory HUD).
    for key, val in raw.items():
        if key in _PACK_EXCLUDE_KEYS:
            continue
        if key not in out:
            out[key] = val
    return _scrub_text(out)
