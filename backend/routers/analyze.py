import logging
import os
import tempfile
import base64

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional

from routers.auth import get_current_user
from services.json_sanitize import log_non_finite_if_any, safe_float, sanitize_json_floats

router = APIRouter()
logger = logging.getLogger(__name__)


class LiteAnalyzeRequest(BaseModel):
    video_url: str
    user_id: Optional[str] = None


class RecalculatePredictionRequest(BaseModel):
    pose_data: dict
    all_frame_angles: list[dict] = []
    swing_duration: float = 1.2
    club_type: Optional[str] = None
    club_group: Optional[str] = None
    hand: Optional[str] = None
    hand_confidence: Optional[float] = None
    preferred_ball_speed: Optional[float] = None


def _extract_frames_safe(tmp_path: str, max_frames: int = 12):
    """Pose extraction only; smart keyframes are built later in /lite (never uniform 5-frame strip for AI)."""
    from services.pose_service import extract_poses_from_video

    poses, pose_bundle = extract_poses_from_video(tmp_path, max_frames=max_frames)
    return (poses if poses else []), pose_bundle


def _extract_keyframe_images_safe(tmp_path: str, num_frames: int = 5) -> list[str]:
    """Extract raw frame images as base64 without pose detection."""
    try:
        import cv2
        from services.video_utils import get_video_rotation, apply_rotation
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        rotation = get_video_rotation(tmp_path)
        import numpy as np
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        images = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = apply_rotation(frame, rotation)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            images.append(base64.b64encode(buf.tobytes()).decode())
        cap.release()
        return images
    except Exception as e:
        print(f"[analyze] Frame extraction failed: {e}")
        return []


@router.post("/lite")
async def analyze_lite(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Lite API with backend-only pipeline internals and public response packing."""
    from services.lite_api_pack_service import pack_lite_public_response
    from services.lite_independent_pipeline import run_lite_independent_pipeline

    tmp_path = None
    try:
        content_type = request.headers.get("content-type", "")
        if "multipart" in content_type:
            form = await request.form()
            uploaded_file = form.get("file")
            if not uploaded_file or not hasattr(uploaded_file, "read"):
                raise HTTPException(status_code=400, detail="No file provided")
            file_bytes = await uploaded_file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Empty file")
            filename = getattr(uploaded_file, "filename", "video.mp4") or "video.mp4"
            suffix = ".mov" if ".mov" in filename.lower() else ".mp4"
            if ".webm" in filename.lower():
                suffix = ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
        else:
            body = await request.json()
            video_url = body.get("video_url", "")
            if not video_url:
                raise HTTPException(status_code=400, detail="No video_url provided")
            if video_url.startswith("blob:"):
                raise HTTPException(
                    status_code=400,
                    detail="Blob URLs cannot be fetched by the server. Please upload the file directly.",
                )
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.get(video_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to download video")
            suffix = ".mov" if ".mov" in video_url.lower() else ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

        region = "CN" if request.headers.get("CF-IPCountry", "").upper() == "CN" else "global"
        internal_result = await run_lite_independent_pipeline(tmp_path, region=region)
        if len(internal_result.get("keyframes") or []) != 8:
            raise HTTPException(status_code=422, detail="Lite analysis failed to produce complete keyframes")
        public_result = pack_lite_public_response(internal_result)
        log_non_finite_if_any(logger, public_result, "analyze_lite")
        return sanitize_json_floats(public_result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/recalculate")
async def recalculate_prediction(
    payload: RecalculatePredictionRequest,
    current_user: Optional[dict] = Depends(get_current_user),
):
    try:
        from services.shot_predictor import predict_shot

        pose_data = payload.pose_data or {}
        all_angles = payload.all_frame_angles or []
        hand = (payload.hand or "UNKNOWN").upper().strip()
        if hand not in ("R", "L", "UNKNOWN"):
            hand = "UNKNOWN"

        pref_speed = payload.preferred_ball_speed
        if pref_speed is not None:
            ps = safe_float(pref_speed, 0.0)
            pref_speed = ps if ps > 0 else None

        prediction = predict_shot(
            pose_data=pose_data,
            swing_duration=safe_float(payload.swing_duration, 1.2),
            all_frame_angles=all_angles,
            club_type=payload.club_type,
            club_group=payload.club_group,
            hand=hand,
            hand_confidence=safe_float(payload.hand_confidence, 0.0),
            preferred_ball_speed=pref_speed,
        )
        recalc_out = {"prediction": prediction}
        log_non_finite_if_any(logger, recalc_out, "recalculate")
        return sanitize_json_floats(recalc_out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recalculate failed: {str(e)}")
