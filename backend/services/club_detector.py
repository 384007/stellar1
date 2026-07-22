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

import asyncio
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

CLUB_DETECT_PROMPT = """You are an expert golf equipment and stance analyst.

FIRST: Determine if this image shows a person holding or swinging a golf club.
If NO golf club is visible, return: {"club_type": "UNKNOWN", "confidence": 0, "hand": "R"}
Only if you can clearly see a golf club, identify the club type and handedness.

首先判断图中是否有人持有或挥动高尔夫球杆。如果看不到球杆，直接返回 UNKNOWN。

Possible club types (球杆型号):
- UNKNOWN: no golf club visible (看不到球杆)
- WOOD (木杆): 1W, 3W, 5W
- IRON (铁杆): 3I, 4I, 5I, 6I, 7I, 8I, 9I
- WEDGE (挖起杆): PW, AW, SW, LW
- PUTTER (推杆): PT

Respond with ONLY this JSON (no markdown, no backticks):
{
  "club_type": "<UNKNOWN or one of: 1W, 3W, 5W, 3I, 4I, 5I, 6I, 7I, 8I, 9I, PW, AW, SW, LW, PT>",
  "confidence": <float 0.0 to 1.0>,
  "hand": "<R or L>"
}

Identification tips / 识别要点:
- Wood clubs have large, rounded heads (木杆杆头大而圆)
- Irons have thin, flat blade-like heads (铁杆杆头薄而平)
- Wedges look similar to short irons but with more loft (挖起杆类似短铁杆但角度更大)
- Putters have a flat face and are used on the green (推杆平面杆头，用于果岭)
- If the club head is not clearly visible but a person is swinging, estimate from shaft length
- Handedness: R = right-handed, L = left-handed; default R if uncertain (无法判断时默认 R)"""

CLUB_DETECT_MULTIFRAME_PROMPT = """You are an expert golf equipment and stance analyst.

You are given THREE JPEG images from the SAME golf swing clip, in chronological order:
Image 1 ≈ 25% through the video, Image 2 ≈ 40%, Image 3 ≈ 60%.
Use ALL images together (majority / clearest evidence) to decide club type and handedness.

FIRST: If no golf club is clearly visible in any frame, return UNKNOWN.

首先判断三帧中是否有人持有或挥动高尔夫球杆；若均看不清球杆，返回 UNKNOWN。

Possible club types (球杆型号):
- UNKNOWN: no golf club visible
- WOOD (木杆): 1W, 3W, 5W
- IRON (铁杆): 3I, 4I, 5I, 6I, 7I, 8I, 9I
- WEDGE (挖起杆): PW, AW, SW, LW
- PUTTER (推杆): PT

Respond with ONLY this JSON (no markdown, no backticks):
{
  "club_type": "<UNKNOWN or one of: 1W, 3W, 5W, 3I, 4I, 5I, 6I, 7I, 8I, 9I, PW, AW, SW, LW, PT>",
  "confidence": <float 0.0 to 1.0>,
  "hand": "<R or L>"
}

Identification tips / 识别要点: same as single-image — woods large round heads, irons thin blades,
wedges high loft, putters flat face; handedness R/L, default R if uncertain."""


def _frame_to_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def encode_bgr_frame_jpeg_b64(frame: np.ndarray, *, quality: int = 85) -> str:
    """Public JPEG base64 for pipelines that merge club frames into a single Gemini call."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return base64.b64encode(buf.tobytes()).decode()


def _parse_club_response(text: str) -> dict:
    """Extract club_type, club_group, confidence, hand from AI text response."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            data = json.loads(match.group())
            club_type = str(data.get("club_type", "")).upper().strip()
            confidence = float(data.get("confidence", 0.0))
            hand_raw = str(data.get("hand", "R")).upper().strip()
            hand = "L" if hand_raw == "L" else "R"

            if club_type == "UNKNOWN" or club_type not in CLUB_GROUP_MAP:
                return {
                    "club_type": "UNKNOWN",
                    "club_group": "IRON",
                    "confidence": round(min(max(confidence, 0.0), 1.0), 2),
                    "hand": hand,
                }
            return {
                "club_type": club_type,
                "club_group": CLUB_GROUP_MAP[club_type],
                "confidence": round(min(max(confidence, 0.0), 1.0), 2),
                "hand": hand,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    logger.warning("Could not parse club detection response: %s", text[:200])
    return {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0, "hand": "R"}


def aggregate_club_detect_frames(frame_results: list[dict]) -> dict:
    """
    Merge 1–3 per-frame ``detect_club`` outputs (same voting rules as legacy browser).

    Strips ``ai_provider`` from the merged HTTP response.
    """
    cleaned: list[dict] = []
    for r in frame_results:
        out = {k: v for k, v in r.items() if k != "ai_provider"}
        cleaned.append(out)

    valid = [c for c in cleaned if c.get("club_type") and str(c["club_type"]) != "UNKNOWN"]
    if valid:
        votes: dict[str, dict[str, float | int | str]] = {}
        for r in valid:
            ct = str(r["club_type"])
            if ct not in votes:
                votes[ct] = {
                    "count": 0,
                    "totalConf": 0.0,
                    "group": str(r.get("club_group") or CLUB_GROUP_MAP.get(ct, "IRON")),
                }
            votes[ct]["count"] = int(votes[ct]["count"]) + 1  # type: ignore[assignment]
            votes[ct]["totalConf"] = float(votes[ct]["totalConf"]) + float(r.get("confidence") or 0.0)  # type: ignore[assignment]
        sorted_items = sorted(
            votes.items(),
            key=lambda x: (-int(x[1]["count"]), -float(x[1]["totalConf"])),  # type: ignore[arg-type]
        )
        winner_ct, w = sorted_items[0]
        count = int(w["count"])
        total_conf = float(w["totalConf"])
        avg_conf = total_conf / max(count, 1)
        hand_votes = {"R": 0, "L": 0}
        for r in cleaned:
            h = "L" if r.get("hand") == "L" else "R"
            hand_votes[h] += 1
        hand = "L" if hand_votes["L"] > hand_votes["R"] else "R"
        return {
            "club_type": winner_ct,
            "club_group": str(w.get("group") or CLUB_GROUP_MAP.get(winner_ct, "IRON")),
            "confidence": round(min(max(avg_conf, 0.0), 1.0), 4),
            "hand": hand,
        }

    first = cleaned[0] if cleaned else {}
    hand = "L" if first.get("hand") == "L" else "R"
    return {
        "club_type": "UNKNOWN",
        "club_group": "IRON",
        "confidence": 0.0,
        "hand": hand,
    }


def read_bgr_frame_at_percent(video_path: str, pct: float) -> np.ndarray | None:
    """Read one BGR frame at relative position ``pct`` in ``[0, 1]`` (best-effort)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 1:
            ok, fr = cap.read()
            return fr if ok and fr is not None else None
        p = min(1.0, max(0.0, float(pct)))
        idx = int(round((n - 1) * p))
        idx = max(0, min(idx, n - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok2, fr2 = cap.read()
        return fr2 if ok2 else None
    finally:
        cap.release()


def _three_jpeg_b64s_from_frames(frames: list[np.ndarray]) -> list[str]:
    """Up to three base64 JPEG payloads; pad to length 3 by repeating last (for one multimodal call)."""
    imgs: list[str] = []
    for fr in frames[:3]:
        imgs.append(_frame_to_base64(fr))
    while len(imgs) < 3:
        imgs.append(imgs[-1])
    return imgs[:3]


async def _detect_club_multiframe_gemini(images_b64: list[str]) -> dict:
    from services.gemini_service import run_gemini_vision

    text, _key_slot = await run_gemini_vision(
        CLUB_DETECT_MULTIFRAME_PROMPT,
        images_b64[:3],
        max_tokens=384,
        temperature=0.2,
    )
    return _parse_club_response(text)


async def _detect_club_multiframe_qwen(images_b64: list[str]) -> dict:
    qwen_key = os.getenv("QWEN_API_KEY", "")
    if not qwen_key:
        raise RuntimeError("QWEN_API_KEY not configured")

    from openai import AsyncOpenAI

    from services.gemini_service import QWEN_TIMEOUT_S

    qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model = "qwen-vl-max-latest"
    client = AsyncOpenAI(
        api_key=qwen_key,
        base_url=qwen_base_url,
        timeout=QWEN_TIMEOUT_S,
    )
    content: list[dict] = [{"type": "text", "text": CLUB_DETECT_MULTIFRAME_PROMPT}]
    for b64 in images_b64[:3]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    resp = await client.chat.completions.create(
        model=qwen_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=384,
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    return _parse_club_response(raw)


def _club_out_public(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "ai_provider"}


async def detect_club_multiframe_bgr(frames: list[np.ndarray], region: str = "global") -> dict:
    """
    One multimodal vision call for 1–3 BGR frames (Gemini / Qwen multiframe prompt).

    Used by ``/analyze/club-detect-batch`` and heuristic Lite A so a single Modal HTTP does not
    trigger three separate ``detect_club`` (single-image) round-trips.
    """
    _unknown = {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0, "hand": "R"}
    clean = [f for f in frames if f is not None and getattr(f, "size", 0) > 0]
    if not clean:
        return dict(_unknown)

    try:
        imgs = _three_jpeg_b64s_from_frames(clean[:3])
    except Exception as e:
        logger.error("Club multiframe encode failed: %s", e)
        return dict(_unknown)

    try:
        out = await _detect_club_multiframe_gemini(imgs)
        logger.info("[ai] club_detect_multiframe provider=gemini")
        hand = out.get("hand")
        if hand not in ("R", "L"):
            out["hand"] = "R"
        return _club_out_public(out)
    except Exception as e:
        logger.warning("Gemini multiframe club detection failed: %s", e)

    if os.getenv("QWEN_API_KEY", ""):
        try:
            out = await _detect_club_multiframe_qwen(imgs)
            logger.info("[ai] club_detect_multiframe provider=qwen")
            hand = out.get("hand")
            if hand not in ("R", "L"):
                out["hand"] = "R"
            return _club_out_public(out)
        except Exception as e2:
            logger.warning("Qwen multiframe club detection failed: %s", e2)

    # Last resort: single middle frame, one vision call (not three sequential).
    mid = clean[len(clean) // 2]
    out = await detect_club(mid, region)
    return _club_out_public(out)


async def detect_club_three_frames_from_video(video_path: str, region: str = "global") -> dict:
    """
    Sample 25% / 40% / 60% of the clip, **one** multimodal vision call (Gemini or Qwen), not three.

    Keeps Lite / Plus / Pro to **one** club-related vision round-trip on top of the product's own AI call.
    """
    _unknown = {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0, "hand": "R"}
    if not video_path or not os.path.isfile(video_path):
        return dict(_unknown)

    pcts = (0.25, 0.4, 0.6)
    frames: list[np.ndarray] = []
    for p in pcts:
        fr = await asyncio.to_thread(read_bgr_frame_at_percent, video_path, p)
        if fr is not None and fr.size > 0:
            frames.append(fr)
    if not frames:
        fr = await asyncio.to_thread(read_bgr_frame_at_percent, video_path, 0.4)
        if fr is not None and fr.size > 0:
            frames = [fr, fr, fr]
    if not frames:
        return dict(_unknown)

    return await detect_club_multiframe_bgr(frames, region=region)


async def _detect_club_gemini(img_b64: str) -> dict:
    """Call Gemini (Developer API or Vertex) to identify the club."""
    from services.gemini_service import run_gemini_vision

    text, _key_slot = await run_gemini_vision(
        CLUB_DETECT_PROMPT, [img_b64], max_tokens=256, temperature=0.2,
    )
    return _parse_club_response(text)


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
            "hand": str — "R" or "L",
            "ai_provider": str — internal diagnostic only; strip before HTTP responses,
        }
    """
    _unknown = {
        "club_type": "UNKNOWN",
        "club_group": "IRON",
        "confidence": 0.0,
        "hand": "R",
        "ai_provider": "none",
    }
    try:
        img_b64 = _frame_to_base64(frame)
    except Exception as e:
        logger.error("Club detection frame encoding failed: %s", e)
        return _unknown

    # Gemini first
    try:
        out = await _detect_club_gemini(img_b64)
        out["ai_provider"] = "gemini"
        if "hand" not in out:
            out["hand"] = "R"
        logger.info("[ai] club_detect provider=gemini")
        return out
    except Exception as e:
        logger.warning("Gemini club detection failed: %s", e)

    # Qwen fallback
    if os.getenv("QWEN_API_KEY", ""):
        try:
            out = await _detect_club_qwen(img_b64)
            out["ai_provider"] = "qwen"
            if "hand" not in out:
                out["hand"] = "R"
            logger.info("[ai] club_detect provider=qwen (fallback)")
            return out
        except Exception as e2:
            logger.warning("Qwen club detection fallback also failed: %s", e2)

    return _unknown
