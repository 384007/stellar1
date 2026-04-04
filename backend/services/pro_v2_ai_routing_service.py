"""Pro v2 — AI video routing (screen vs standard pipeline strategy) before heavy analysis."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.gemini_service import analyze_pro_v2_screen_route

logger = logging.getLogger(__name__)


def _jpeg_b64(frame_bgr: Any, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def sample_video_storyboard_b64(video_path: str, *, max_frames: int = 8) -> list[str]:
    """Evenly spaced JPEG samples for routing (low cost)."""
    path = str(Path(video_path))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            ok, fr = cap.read()
            cap.release()
            return [_jpeg_b64(fr)] if ok and fr is not None else []
        idxs = np.linspace(0, max(0, n - 1), num=min(max_frames, max(1, n)), dtype=int)
        out: list[str] = []
        for idx in idxs.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok and fr is not None:
                b = _jpeg_b64(fr)
                if b:
                    out.append(b)
        return out
    finally:
        cap.release()


def _boolish(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no", ""):
        return False
    return default


def _normalize_route(raw: dict[str, Any], *, screen_mode_requested: bool) -> dict[str, Any]:
    ql = str(raw.get("quality_level") or "medium").strip().lower()
    if ql not in ("high", "medium", "low"):
        ql = "medium"
    pipe = str(raw.get("recommended_pipeline") or "").strip()
    if screen_mode_requested and pipe != "screen_mode_pipeline":
        pipe = "screen_mode_pipeline"
    if not screen_mode_requested and pipe not in ("standard_pipeline", "screen_mode_pipeline"):
        pipe = "standard_pipeline"
    ceiling = raw.get("expected_confidence_ceiling")
    try:
        ce = float(ceiling)
    except (TypeError, ValueError):
        ce = 0.78 if screen_mode_requested else 0.92
    ce = max(0.05, min(0.99, ce))
    return {
        "screen_mode_confirmed": _boolish(raw.get("screen_mode_confirmed"), screen_mode_requested),
        "recommended_pipeline": pipe,
        "quality_level": ql,
        "use_deblur": _boolish(raw.get("use_deblur"), screen_mode_requested),
        "use_heavy_club_tracking": _boolish(raw.get("use_heavy_club_tracking"), screen_mode_requested),
        "pose_priority": _boolish(raw.get("pose_priority"), False),
        "expected_confidence_ceiling": round(ce, 4),
    }


async def run_pro_v2_ai_routing(
    input_video_path: str,
    *,
    screen_mode_requested: bool,
) -> dict[str, Any]:
    """AI pass 1: structured backend strategy (no report, no keyframe picks)."""
    logger.info(
        "[PRO_V2][ROUTE] screen_mode_requested=%s analysis_input=raw",
        "true" if screen_mode_requested else "false",
    )
    shots = sample_video_storyboard_b64(input_video_path, max_frames=8)
    if not shots:
        logger.warning("[PRO_V2][ROUTE] no_storyboard_frames — using defaults")
        return _normalize_route({}, screen_mode_requested=screen_mode_requested)

    raw = await analyze_pro_v2_screen_route(
        shots,
        screen_mode_requested=screen_mode_requested,
        call_label="pro_v2_route",
    )
    norm = _normalize_route(raw, screen_mode_requested=screen_mode_requested)
    logger.info(
        "[PRO_V2][ROUTE] screen_mode_confirmed=%s pipeline=%s quality=%s deblur=%s heavy_club=%s pose_pri=%s ceiling=%s",
        norm["screen_mode_confirmed"],
        norm["recommended_pipeline"],
        norm["quality_level"],
        norm["use_deblur"],
        norm["use_heavy_club_tracking"],
        norm["pose_priority"],
        norm["expected_confidence_ceiling"],
    )
    return norm
