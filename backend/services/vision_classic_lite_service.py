"""
Classic Lite vision analysis (image/video) — product JSON only, no provider fields.
Mirrors legacy Edge ``/api/analyze`` prompt shape for Modal/FastAPI.

Default AI path: NVIDIA/video-capable OpenAI-compatible providers from Modal secrets.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Optional

from services.video_upload_suffix import is_likely_video_filename, looks_like_video_mime

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are an expert PGA-level golf coach and biomechanics analyst.

STEP 1 — DETECTION (CRITICAL):
Describe what you actually see. Determine whether the image/video shows a REAL golf swing.
Set "is_golf_swing" to true ONLY if you can clearly see a person swinging a golf club.
If the content shows anything else — a person standing, a landscape, a non-golf activity, a random object, or an unclear image — you MUST set "is_golf_swing" to false.

STEP 2 — ANALYSIS (only if is_golf_swing is true):
Evaluate these 5 dimensions (0-100):
1. Grip 2. Stance 3. Backswing 4. Downswing 5. Follow-through
Be brutally honest. Amateur golfers typically score 40-75. Do NOT inflate scores.

If is_golf_swing is false: set all scores to 0, total_score to 0, issues/suggestions to empty arrays, and describe what you see in summary.

Respond with ONLY this JSON (no markdown, no backticks):
{
  "what_i_see": "<describe what is visible>",
  "what_i_see_zh": "<中文描述>",
  "is_golf_swing": true or false,
  "scores": {"grip": <0-100>, "stance": <0-100>, "backswing": <0-100>, "downswing": <0-100>, "follow_through": <0-100>},
  "total_score": <weighted average>,
  "issues": ["<issue 1>","<issue 2>","<issue 3>"],
  "issues_zh": ["<问题1>","<问题2>","<问题3>"],
  "suggestions": ["<fix 1>","<fix 2>","<fix 3>"],
  "suggestions_zh": ["<建议1>","<建议2>","<建议3>"],
  "summary": "<English analysis 150-200 words>",
  "summary_zh": "<中文分析150-200字>",
  "prediction": {"predicted_distance":<yards>,"lateral_offset":<yards>,"shot_shape":"<shape>","shot_shape_zh":"<中文>","club_head_speed":<mph>,"ball_speed":<mph>,"launch_angle":<deg>,"spin_rate":<rpm>,"smash_factor":<ratio>}
}"""

class VisionClassicLiteError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _build_product(parsed: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise VisionClassicLiteError("Invalid AI parse result", 502)
    pred_raw = parsed.get("prediction")
    pred = pred_raw if isinstance(pred_raw, dict) else {}
    pid = int(time.time() * 1000)
    scores = parsed.get("scores") or {
        "grip": 0,
        "stance": 0,
        "backswing": 0,
        "downswing": 0,
        "follow_through": 0,
    }
    return {
        "analysis_id": f"stellar-{pid}",
        "type": "lite",
        "what_i_see": str(parsed.get("what_i_see") or ""),
        "what_i_see_zh": str(parsed.get("what_i_see_zh") or ""),
        "is_golf_swing": parsed.get("is_golf_swing") is True,
        "scores": scores,
        "total_score": float(parsed.get("total_score") or 0),
        "issues": parsed.get("issues") or [],
        "issues_zh": parsed.get("issues_zh") or [],
        "suggestions": parsed.get("suggestions") or [],
        "suggestions_zh": parsed.get("suggestions_zh") or [],
        "summary": str(parsed.get("summary") or ""),
        "summary_zh": str(parsed.get("summary_zh") or ""),
        "keyframes": parsed.get("keyframes") if isinstance(parsed.get("keyframes"), list) else [],
        "skeleton_data": {"frames": [], "total_frames": 0},
        "prediction": {
            "predicted_distance": float(pred.get("predicted_distance") or 0),
            "lateral_offset": float(pred.get("lateral_offset") or 0),
            "shot_shape": str(pred.get("shot_shape") or "N/A"),
            "shot_shape_zh": str(pred.get("shot_shape_zh") or "未知"),
            "club_head_speed": float(pred.get("club_head_speed") or 0),
            "ball_speed": float(pred.get("ball_speed") or 0),
            "launch_angle": float(pred.get("launch_angle") or 0),
            "spin_rate": float(pred.get("spin_rate") or 0),
            "smash_factor": float(pred.get("smash_factor") or 0),
            "trajectory": [],
        },
    }


def _video_ai_classic_sync(tmp_path: str, mime_type: str, filename: str) -> dict[str, Any]:
    from services.gemini_service import extract_json_from_response, run_video_ai_sync

    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    raw_type = mime_type or ""
    is_video = looks_like_video_mime(raw_type) or is_likely_video_filename(filename)
    mt = "video/mp4" if raw_type == "video/quicktime" else (raw_type or ("video/mp4" if is_video else "image/jpeg"))
    images = [] if is_video else [b64]
    videos = [(b64, mt)] if is_video else []
    text, provider, key_label = run_video_ai_sync(
        ANALYSIS_PROMPT,
        images_b64=images,
        videos_b64=videos,
        max_tokens=4096,
        temperature=0.3,
        label="vision_classic",
    )
    logger.info("[vision-classic] video_ai provider=%s ai_key=%s", provider, key_label or "-")
    return extract_json_from_response(text)


def run_vision_classic_lite_sync(
    tmp_path: Optional[str],
    file_uri: Optional[str],
    mime_type: str,
    filename: str,
    region_cn: bool,
    key_hint: Optional[int],
) -> dict[str, Any]:
    _ = file_uri, region_cn, key_hint

    if tmp_path:
        try:
            parsed = _video_ai_classic_sync(tmp_path, mime_type, filename)
            return _build_product(parsed)
        except Exception as e:
            logger.warning("[vision-classic] NVIDIA/video AI path fail: %s", e)
            raise VisionClassicLiteError(str(e), 503) from e

    raise VisionClassicLiteError("NVIDIA video AI unavailable; raw file required", 503)
