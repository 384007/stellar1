from __future__ import annotations

import base64
import logging
from typing import Any

from services.gemini_service import extract_json_from_response, run_video_ai_sync
from services.video_upload_suffix import is_likely_video_filename, looks_like_video_mime

logger = logging.getLogger(__name__)

LAB_ANALYSIS_PROMPT = """You are an expert PGA-level golf biomechanics analyst performing phone-camera-only video analysis for a product called "Shot Lab".
IMPORTANT: All values are ESTIMATES from video analysis - NOT radar/launch-monitor measurements. Be honest about what you can and cannot determine from video.

Analyze this golf swing and return ONLY the following JSON (no markdown, no backticks):
{
  "is_golf_swing": true or false,
  "what_i_see": "<describe what is visible in the video>",
  "what_i_see_zh": "<中文描述>",
  "metrics": {
    "ball_speed_mph": <number or null if not estimable>,
    "ball_speed_confidence": <0.0-1.0>,
    "launch_angle_deg": <number or null>,
    "launch_angle_confidence": <0.0-1.0>,
    "launch_direction_deg": <number or null, positive=right of target>,
    "launch_direction_confidence": <0.0-1.0>,
    "backswing_time_sec": <number or null>,
    "downswing_time_sec": <number or null>,
    "tempo_ratio": <backswing/downswing ratio or null>,
    "tempo_confidence": <0.0-1.0>,
    "carry_distance_yards": <number or null>,
    "carry_distance_confidence": <0.0-1.0>,
    "contact_quality_score": <0-100 or null>,
    "contact_quality_confidence": <0.0-1.0>
  },
  "issues": [
    {
      "id": "<snake_case_id>",
      "title": "<English title>",
      "title_zh": "<中文标题>",
      "description": "<English description 1-2 sentences>",
      "description_zh": "<中文描述>",
      "severity": "high" or "medium" or "low",
      "drill": "<English drill recommendation>",
      "drill_zh": "<中文训练建议>"
    }
  ],
  "summary": "<English analysis 100-200 words>",
  "summary_zh": "<中文分析总结100-200字>",
  "full_report": "<English detailed structured report 300-500 words covering setup, backswing, transition, downswing, impact, follow-through with specific observations>",
  "full_report_zh": "<中文详细报告300-500字>",
  "drills": [
    {
      "title": "<English drill title>",
      "title_zh": "<中文训练名称>",
      "description": "<English description 2-3 sentences>",
      "description_zh": "<中文描述>"
    }
  ]
}

Rules:
- Identify at least 3 issues if it IS a golf swing, up to 10
- Provide at least 3 drills
- For metrics you cannot estimate from the video, use null (do NOT fabricate numbers)
- Ball speed/carry distance are rough estimates based on visual club speed and contact quality
- Tempo is measurable from frame timing if the video has sufficient frame rate
- All metric sources must be video-based estimation, label them honestly"""


def run_lab_video_ai_sync(tmp_path: str, mime_type: str, filename: str) -> dict[str, Any]:
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    raw_type = (mime_type or "").strip()
    is_video = looks_like_video_mime(raw_type) or is_likely_video_filename(filename)
    mt = "video/mp4" if raw_type == "video/quicktime" else (raw_type or ("video/mp4" if is_video else "image/jpeg"))
    images = [] if is_video else [b64]
    videos = [(b64, mt)] if is_video else []
    text, provider, key_label = run_video_ai_sync(
        LAB_ANALYSIS_PROMPT,
        images_b64=images,
        videos_b64=videos,
        max_tokens=8192,
        temperature=0.3,
        label="shot_lab",
    )
    logger.info("[shot-lab] video_ai provider=%s ai_key=%s", provider, key_label or "-")
    return extract_json_from_response(text)
