"""
Club detection via AI Vision.

Sends a keyframe to Gemini or Qwen (based on region) and asks the model to
identify the golf club type visible in the image.  Returns a structured dict
with club_type, club_group, and confidence.

Uses the same Gemini stack as gemini_service.py:
    GEMINI_BACKEND=vertex  → Vertex AI (GCP project + ADC)
    otherwise             → Google AI Studio API (GEMINI_API_KEY)
    then Qwen fallback if Gemini fails and QWEN_API_KEY is set.
"""

import base64
import json
import logging
import os
import re

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLUB_GROUP_MAP: dict[str, str] = {
    "1W": "WOOD", "3W": "WOOD", "5W": "WOOD",
    "3I": "IRON", "4I": "IRON", "5I": "IRON", "6I": "IRON",
    "7I": "IRON", "8I": "IRON", "9I": "IRON",
    "PW": "WEDGE", "AW": "WEDGE", "SW": "WEDGE", "LW": "WEDGE",
    "PT": "PUTTER",
}

ALL_CLUB_TYPES = list(CLUB_GROUP_MAP.keys())

CLUB_DETECT_PROMPT = """You are an expert golf equipment analyst.
Look at this golf swing image carefully and identify the type of golf club the player is using.

你是一名高尔夫器材专家。请仔细观察图片中球员使用的球杆，判断球杆型号。

Possible club types (球杆型号):
- WOOD (木杆): 1W, 3W, 5W
- IRON (铁杆): 3I, 4I, 5I, 6I, 7I, 8I, 9I
- WEDGE (挖起杆): PW, AW, SW, LW
- PUTTER (推杆): PT

Respond with ONLY this JSON (no markdown, no backticks):
{
  "club_type": "<one of: 1W, 3W, 5W, 3I, 4I, 5I, 6I, 7I, 8I, 9I, PW, AW, SW, LW, PT>",
  "confidence": <float 0.0 to 1.0>
}

Identification tips / 识别要点:
- Wood clubs have large, rounded heads (木杆杆头大而圆)
- Irons have thin, flat blade-like heads (铁杆杆头薄而平)
- Wedges look similar to short irons but with more loft (挖起杆类似短铁杆但角度更大)
- Putters have a flat face and are used on the green (推杆平面杆头，用于果岭)
- If the club head is not clearly visible, estimate from shaft length and swing posture
  (如果杆头不清晰，可从杆身长度和挥杆姿势推断)
- If you truly cannot determine the club type, use "7I" with low confidence
  (如果确实无法判断，使用 "7I" 并给出低置信度)"""


def _frame_to_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def _parse_club_response(text: str) -> dict:
    """Extract club_type, club_group, confidence from AI text response."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            data = json.loads(match.group())
            club_type = str(data.get("club_type", "")).upper().strip()
            confidence = float(data.get("confidence", 0.0))

            if club_type in CLUB_GROUP_MAP:
                return {
                    "club_type": club_type,
                    "club_group": CLUB_GROUP_MAP[club_type],
                    "confidence": round(min(max(confidence, 0.0), 1.0), 2),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    logger.warning("Could not parse club detection response: %s", text[:200])
    return {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0}


async def _detect_club_gemini(img_b64: str) -> dict:
    """Call Gemini (Developer API or Vertex) to identify the club."""
    from services.gemini_service import developer_key_label, run_gemini_vision

    text, key_slot = await run_gemini_vision(
        CLUB_DETECT_PROMPT, [img_b64], max_tokens=256, temperature=0.2,
    )
    out = _parse_club_response(text)
    if key_slot is not None:
        out["ai_key"] = developer_key_label(key_slot)
    return out


async def _detect_club_qwen(img_b64: str) -> dict:
    """Call Qwen-VL via DashScope to identify the club."""
    qwen_key = os.getenv("QWEN_API_KEY", "")
    if not qwen_key:
        raise RuntimeError("QWEN_API_KEY not configured")

    from openai import AsyncOpenAI

    qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model = "qwen-vl-max-latest"
    from services.gemini_service import QWEN_TIMEOUT_S

    client = AsyncOpenAI(
        api_key=qwen_key,
        base_url=qwen_base_url,
        timeout=QWEN_TIMEOUT_S,
    )

    resp = await client.chat.completions.create(
        model=qwen_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CLUB_DETECT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        max_tokens=256,
        temperature=0.2,
    )

    raw = resp.choices[0].message.content or ""
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    return _parse_club_response(raw)


async def detect_club(frame: np.ndarray, region: str = "global") -> dict:
    """
    Identify the golf club in a video frame using AI Vision.
    Always tries Gemini first; falls back to Qwen if Gemini fails and
    QWEN_API_KEY is available.

    Args:
        frame:  BGR image (np.ndarray) — typically an impact or address keyframe.
        region: Passed through but no longer gates provider selection.

    Returns:
        {
            "club_type":  str   — e.g. "7I", or "UNKNOWN" on failure,
            "club_group": str   — one of WOOD / IRON / WEDGE / PUTTER,
            "confidence": float — 0.0 – 1.0,
        }
    """
    _unknown = {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0, "ai_provider": "none"}
    try:
        img_b64 = _frame_to_base64(frame)
    except Exception as e:
        logger.error("Club detection frame encoding failed: %s", e)
        return _unknown

    # Gemini first
    try:
        out = await _detect_club_gemini(img_b64)
        out["ai_provider"] = "gemini"
        logger.info("[ai] club_detect provider=gemini")
        return out
    except Exception as e:
        logger.warning("Gemini club detection failed: %s", e)

    # Qwen fallback
    if os.getenv("QWEN_API_KEY", ""):
        try:
            out = await _detect_club_qwen(img_b64)
            out["ai_provider"] = "qwen"
            logger.info("[ai] club_detect provider=qwen (fallback)")
            return out
        except Exception as e2:
            logger.warning("Qwen club detection fallback also failed: %s", e2)

    return _unknown
