"""Stellar Pro: legacy Pro–parity club vision on impact keyframe (not used for keyframe selection)."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Any

import cv2
import numpy as np

from services.club_detector import detect_club, detect_club_three_frames_from_video
from services.shot_predictor import calibrate_prediction

logger = logging.getLogger(__name__)

STELLAR_PRO_CLUB_DETECT_TIMEOUT_S = 45.0
STELLAR_PRO_CLUB_VIDEO_TIMEOUT_S = 180.0


def keyframe_jpeg_b64_to_bgr(b64: str) -> np.ndarray | None:
    """Decode data-URL or raw base64 JPEG to BGR ndarray for club_detector."""
    raw = (b64 or "").strip()
    if not raw:
        return None
    if raw.startswith("data:"):
        parts = raw.split(",", 1)
        raw = parts[-1] if len(parts) == 2 else raw
    try:
        buf = base64.b64decode(raw, validate=False)
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        return None


def mirror_hand_club_flags_to_top_level(raw: dict[str, Any]) -> None:
    """Match legacy /analyze/pro: hand_*/club_* warnings on response root."""
    p = raw.get("prediction")
    if not isinstance(p, dict):
        return
    if p.get("hand_assumed"):
        raw["hand_assumed"] = p["hand_assumed"]
        raw["hand_warning"] = p.get("hand_warning", "")
    else:
        raw.pop("hand_assumed", None)
        raw.pop("hand_warning", None)
    if p.get("club_assumed"):
        raw["club_assumed"] = p["club_assumed"]
        raw["club_warning"] = p.get("club_warning", "")
    else:
        raw.pop("club_assumed", None)
        raw.pop("club_warning", None)


async def apply_impact_club_vision_to_result(
    raw: dict[str, Any],
    keyframes: list[dict[str, Any]],
    *,
    region: str,
    source_video_path: str | None = None,
) -> None:
    """Merge club vision into prediction + detected_club (3-frame video or impact JPEG fallback)."""
    club_info: dict[str, Any] | None = None
    if source_video_path and os.path.isfile(source_video_path):
        logger.info("[STELLAR_PRO][CLUB_VISION] stage=start path=video_three_frame")
        try:
            club_info = await asyncio.wait_for(
                detect_club_three_frames_from_video(source_video_path, region=region),
                timeout=STELLAR_PRO_CLUB_VIDEO_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[STELLAR_PRO][CLUB_VISION] stage=timeout video after %ss",
                STELLAR_PRO_CLUB_VIDEO_TIMEOUT_S,
            )
        except Exception as exc:
            logger.warning("[STELLAR_PRO][CLUB_VISION] video stage=failed err=%s", exc)

    if club_info is None:
        imp = next((k for k in keyframes if str(k.get("phase")) == "impact"), None)
        if not imp:
            logger.warning("[STELLAR_PRO][CLUB_VISION] no impact keyframe")
            return
        frame = keyframe_jpeg_b64_to_bgr(str(imp.get("image_base64") or ""))
        if frame is None:
            logger.warning("[STELLAR_PRO][CLUB_VISION] impact image decode failed")
            return

        logger.info("[STELLAR_PRO][CLUB_VISION] stage=start path=impact_jpeg")
        try:
            club_info = await asyncio.wait_for(
                detect_club(frame, region),
                timeout=STELLAR_PRO_CLUB_DETECT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[STELLAR_PRO][CLUB_VISION] stage=timeout after %ss",
                STELLAR_PRO_CLUB_DETECT_TIMEOUT_S,
            )
            return
        except Exception as exc:
            logger.warning("[STELLAR_PRO][CLUB_VISION] stage=failed err=%s", exc)
            return

    if club_info is None:
        return

    ct = str(club_info.get("club_type") or "").upper().strip()
    cg = str(club_info.get("club_group") or "").upper().strip()
    conf = float(club_info.get("confidence") or 0.0)
    logger.info(
        "[STELLAR_PRO][CLUB_VISION] stage=done club=%s group=%s conf=%.3f",
        ct,
        cg,
        conf,
    )

    raw["detected_club"] = {
        "club_type": ct or "UNKNOWN",
        "club_group": cg or "IRON",
        "confidence": round(conf, 4),
    }

    if not ct or ct == "UNKNOWN":
        return

    pred = dict(raw.get("prediction") or {})
    pred["club_type"] = ct
    pred["club_group"] = cg
    pred["club_detection_confidence"] = round(conf, 4)
    pred.pop("club_assumed", None)
    pred.pop("club_warning", None)
    try:
        pred = calibrate_prediction(pred, club_type=ct, club_group=cg if cg else None)
    except Exception as exc:
        logger.warning("[STELLAR_PRO][CLUB_VISION] calibrate_prediction err=%s", exc)
    raw["prediction"] = pred
