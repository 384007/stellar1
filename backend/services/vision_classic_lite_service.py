"""
Classic Lite vision analysis (image/video) — product JSON only, no provider fields.
Mirrors legacy Edge ``/api/analyze`` prompt shape for Modal/FastAPI.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from typing import Any, Optional

import httpx

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

QWEN_MODEL = "qwen-vl-max-latest"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class VisionClassicLiteError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _files_name_from_uri(file_uri: str) -> Optional[str]:
    m = re.search(r"/files/([^/?#]+)", file_uri)
    if not m:
        return None
    return f"files/{m.group(1)}"


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


def _qwen_classic_sync(tmp_path: str, mime_type: str, filename: str) -> dict[str, Any]:
    from services.gemini_service import extract_json_from_response

    key = (os.getenv("QWEN_API_KEY") or "").strip()
    if not key:
        raise VisionClassicLiteError("通义千问 API 密钥未配置 (QWEN_API_KEY)", 503)
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = (filename or "").lower()
    is_video = (mime_type or "").startswith("video/") or ext.endswith((".mp4", ".mov", ".webm", ".avi"))
    mt = "video/mp4" if mime_type == "video/quicktime" else (mime_type or ("video/mp4" if is_video else "image/jpeg"))
    media: dict[str, Any] = (
        {"type": "video_url", "video_url": {"url": f"data:{mt};base64,{b64}"}}
        if is_video
        else {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
    )
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": ANALYSIS_PROMPT}, media]}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            QWEN_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
    if r.status_code != 200:
        raise VisionClassicLiteError(f"Qwen 分析错误 [{r.status_code}]: {r.text[:200]}", 502)
    data = r.json()
    raw = str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")
    return extract_json_from_response(raw)


def _wait_genai_file_active(genai: Any, fref: Any) -> None:
    while fref.state.name == "PROCESSING":
        time.sleep(2)
        fref = genai.get_file(fref.name)
    if fref.state.name != "ACTIVE":
        raise RuntimeError(f"File not ACTIVE: {fref.state.name}")


def run_vision_classic_lite_sync(
    tmp_path: Optional[str],
    file_uri: Optional[str],
    mime_type: str,
    filename: str,
    region_cn: bool,
    key_hint: Optional[int],
) -> dict[str, Any]:
    import google.generativeai as genai

    from services.gemini_service import (
        _collect_developer_api_keys,
        _gemini_dev_lock,
        _gemini_modal_cn_proxy_first,
        _genai_configure_developer,
        _reverse_proxy_origins_from_env,
        extract_json_from_response,
        gemini_modal_cn_proxy_first_context,
    )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    keys = _collect_developer_api_keys()

    def ordered_keys() -> list[tuple[int, str]]:
        if not keys:
            return []
        if key_hint is None or key_hint < 0 or key_hint >= len(keys):
            return list(enumerate(keys))
        rot = keys[key_hint:] + keys[:key_hint]
        return [(keys.index(k), k) for k in rot]

    last_err: Optional[BaseException] = None

    with gemini_modal_cn_proxy_first_context(region_cn):
        proxies = _reverse_proxy_origins_from_env()
        proxy_first = bool(_gemini_modal_cn_proxy_first.get()) and bool(proxies)
        if proxy_first:
            endpoints: list[Optional[str]] = list(proxies) + [None]
        elif proxies:
            endpoints = [None] + list(proxies)
        else:
            endpoints = [None]

        ok_keys = ordered_keys()

        if file_uri and keys:
            name = _files_name_from_uri(file_uri)
            if name:
                for ep in endpoints:
                    for _idx, api_key in ok_keys:
                        try:
                            with _gemini_dev_lock:
                                _genai_configure_developer(api_key, api_endpoint=ep)
                                vf = genai.get_file(name)
                                _wait_genai_file_active(genai, vf)
                                model = genai.GenerativeModel(model_name)
                                response = model.generate_content(
                                    [ANALYSIS_PROMPT, vf],
                                    generation_config={"temperature": 0.3, "max_output_tokens": 4096},
                                )
                                if not response.candidates:
                                    raise RuntimeError("empty Gemini candidates")
                                text = (response.text or "").strip()
                            parsed = extract_json_from_response(text)
                            return _build_product(parsed)
                        except Exception as e:
                            last_err = e
                            logger.warning("[vision-classic] uri path fail: %s", e)
                            continue

        if tmp_path and keys:
            display = filename or "swing.mp4"
            mt = mime_type or "video/mp4"
            for ep in endpoints:
                for _idx, api_key in ok_keys:
                    try:
                        with _gemini_dev_lock:
                            _genai_configure_developer(api_key, api_endpoint=ep)
                            uploaded = genai.upload_file(path=tmp_path, mime_type=mt, display_name=display)
                            _wait_genai_file_active(genai, uploaded)
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(
                                [ANALYSIS_PROMPT, uploaded],
                                generation_config={"temperature": 0.3, "max_output_tokens": 4096},
                            )
                            if not response.candidates:
                                raise RuntimeError("empty Gemini candidates")
                            text = (response.text or "").strip()
                        parsed = extract_json_from_response(text)
                        return _build_product(parsed)
                    except Exception as e:
                        last_err = e
                        logger.warning("[vision-classic] upload path fail: %s", e)
                        continue

        if tmp_path and (os.getenv("QWEN_API_KEY") or "").strip():
            try:
                parsed = _qwen_classic_sync(tmp_path, mime_type, filename)
                return _build_product(parsed)
            except VisionClassicLiteError:
                raise
            except Exception as e:
                last_err = e

    msg = str(last_err) if last_err else "AI 服务不可用"
    raise VisionClassicLiteError(msg, 503)
