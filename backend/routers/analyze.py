import asyncio
import base64
import json
import logging
import os
import tempfile

import httpx
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException, Depends, Request, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from collections.abc import AsyncIterator
from typing import Any, Optional

from routers.auth import get_current_user
from services.json_sanitize import log_non_finite_if_any, safe_float, sanitize_json_floats
from services.lite_singleflight import (
    begin_lite_analyze,
    complete_lite_analyze_failure,
    complete_lite_analyze_success,
)
from services.video_upload_suffix import temp_suffix_for_uploaded_video, temp_suffix_from_url

router = APIRouter()
logger = logging.getLogger(__name__)

_ID_HEADER = "x-stellar-idempotency-key"

# SSE comment lines every N seconds while Lite pipeline runs — keeps proxies from closing “idle” TCP during ~2–3m compute.
LITE_SSE_KEEPALIVE_S = float(os.getenv("STELLAR_LITE_SSE_KEEPALIVE_S", "12"))


async def _lite_analyze_sse_stream(tmp_path: str, region: str, request_id: str) -> AsyncIterator[bytes]:
    from services.lite_api_pack_service import pack_lite_public_response
    from services.lite_independent_pipeline import run_lite_independent_pipeline

    completed_normally = False
    last_exc: Optional[BaseException] = None
    task: Optional[asyncio.Task] = None
    try:
        yield b": lite-start\n\n"
        task = asyncio.create_task(run_lite_independent_pipeline(tmp_path, region=region))
        while True:
            try:
                internal_result = await asyncio.wait_for(asyncio.shield(task), timeout=LITE_SSE_KEEPALIVE_S)
                break
            except asyncio.TimeoutError:
                yield b": lite-analyze\n\n"

        if len(internal_result.get("keyframes") or []) != 8:
            body = {
                "ok": False,
                "status": 422,
                "detail": "Lite analysis failed to produce complete keyframes",
                "code": "LITE_KEYFRAMES_INCOMPLETE",
                "request_id": request_id,
            }
            yield f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8")
            return

        public_result = pack_lite_public_response(internal_result)
        log_non_finite_if_any(logger, public_result, "analyze_lite")
        out = sanitize_json_floats(public_result)
        await complete_lite_analyze_success(request_id, out)
        completed_normally = True
        payload = {"ok": True, "result": out}
        yield f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8")
    except asyncio.CancelledError:
        last_exc = asyncio.CancelledError()
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except Exception:
                pass
        raise
    except Exception as e:
        last_exc = e
        logger.exception("[analyze_lite] stream pipeline_failed request_id=%r", request_id)
        msg = str(e)
        low = msg.lower()
        is_quotaish = (
            "429" in msg
            or "resource exhausted" in low
            or "resource_exhausted" in low
            or "quota" in low
            or "rate limit" in low
            or "too many requests" in low
        )
        if is_quotaish:
            body = {
                "ok": False,
                "status": 503,
                "detail": msg[:4000],
                "code": "LITE_AI_QUOTA_OR_RATE_LIMIT",
                "request_id": request_id,
            }
        else:
            body = {
                "ok": False,
                "status": 500,
                "detail": f"Analysis failed: {msg}"[:4000],
                "code": "LITE_PIPELINE_FAILED",
                "request_id": request_id,
            }
        yield f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if request_id and not completed_normally:
            try:
                await complete_lite_analyze_failure(request_id, exc=last_exc)
            except Exception:
                pass


def _coerce_idempotency_field(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _resolve_lite_idempotency_key(
    request: Request,
    form_request_id: Optional[Any],
    json_request_id: Optional[Any],
) -> str:
    header_raw = request.headers.get(_ID_HEADER) or request.headers.get("X-Stellar-Idempotency-Key")
    header_v = _coerce_idempotency_field(header_raw)
    form_v = _coerce_idempotency_field(form_request_id)
    json_v = _coerce_idempotency_field(json_request_id)
    body_v = form_v or json_v
    if not header_v or not body_v:
        logger.warning("[lite_singleflight] missing_key header=%r form/json=%r", bool(header_v), bool(body_v))
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Missing idempotency key",
                "code": "LITE_IDEMPOTENCY_KEY_REQUIRED",
            },
        )
    if header_v != body_v:
        logger.warning(
            "[lite_singleflight] mismatch header_request_id=%s body_request_id=%s",
            header_v,
            body_v,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Idempotency key mismatch",
                "code": "LITE_IDEMPOTENCY_MISMATCH",
            },
        )
    return header_v


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
    """Lite API — long runs return ``text/event-stream`` with SSE keepalives, final ``data:`` JSON envelope."""
    tmp_path = None
    request_id: str | None = None

    def _lite_409(rid: str) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "Lite analyze already running",
                "code": "LITE_ANALYZE_ALREADY_RUNNING",
                "request_id": rid,
            },
        )

    try:
        content_type = request.headers.get("content-type", "")
        if "multipart" in content_type:
            form = await request.form()
            uploaded_file = form.get("file")
            form_rid = form.get("request_id")
            request_id = _resolve_lite_idempotency_key(request, form_rid, None)
            if not uploaded_file or not hasattr(uploaded_file, "read"):
                raise HTTPException(status_code=400, detail="No file provided")
            file_bytes = await uploaded_file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Empty file")
            filename = getattr(uploaded_file, "filename", "video.mp4") or "video.mp4"
            suffix = temp_suffix_for_uploaded_video(filename)

            status, cached = await begin_lite_analyze(request_id)
            if status == "cached":
                return cached
            if status == "busy":
                return _lite_409(request_id)

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
        else:
            body = await request.json()
            video_url = body.get("video_url", "")
            request_id = _resolve_lite_idempotency_key(request, None, body.get("request_id"))
            if not video_url:
                raise HTTPException(status_code=400, detail="No video_url provided")
            if video_url.startswith("blob:"):
                raise HTTPException(
                    status_code=400,
                    detail="Blob URLs cannot be fetched by the server. Please upload the file directly.",
                )

            status, cached = await begin_lite_analyze(request_id)
            if status == "cached":
                return cached
            if status == "busy":
                return _lite_409(request_id)

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.get(video_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to download video")
            suffix = temp_suffix_from_url(video_url)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

        region = "CN" if request.headers.get("CF-IPCountry", "").upper() == "CN" else "global"
        assert tmp_path is not None and request_id is not None
        return StreamingResponse(
            _lite_analyze_sse_stream(tmp_path, region, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except HTTPException as he:
        if he.status_code == 400 and isinstance(he.detail, dict):
            return JSONResponse(status_code=400, content=he.detail)
        raise


@router.post("/club-detect")
async def analyze_club_detect_multipart(
    request: Request,
    frame: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """
    Club + handedness from a single JPEG frame.
    Provider / keys stay in ``gemini_service``; response is product-only.
    """
    from services.club_detector import detect_club

    raw = await frame.read()
    if not raw:
        return JSONResponse(
            content={
                "club_type": "UNKNOWN",
                "club_group": "IRON",
                "confidence": 0.0,
                "hand": "R",
            }
        )
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(
            content={
                "club_type": "UNKNOWN",
                "club_group": "IRON",
                "confidence": 0.0,
                "hand": "R",
            }
        )
    region = "CN" if (request.headers.get("CF-IPCountry") or "").upper() == "CN" else "global"
    out = await detect_club(img, region)
    hand = out.get("hand")
    if hand not in ("R", "L"):
        hand = "R"
    return JSONResponse(
        content={
            "club_type": str(out.get("club_type") or "UNKNOWN"),
            "club_group": str(out.get("club_group") or "IRON"),
            "confidence": float(out.get("confidence") or 0.0),
            "hand": hand,
        }
    )


@router.post("/club-detect-batch")
async def analyze_club_detect_batch_multipart(
    request: Request,
    frame_0: UploadFile = File(...),
    frame_1: UploadFile = File(...),
    frame_2: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """
    Three JPEG frames in **one** HTTP request — **one** multimodal club vision call (not 3× single-frame).

    One Modal/ASGI invocation; one Gemini/Qwen round-trip for the three frames together.
    """
    from services.club_detector import detect_club_multiframe_bgr

    async def _decode_one(up: UploadFile) -> Optional[np.ndarray]:
        raw = await up.read()
        if not raw:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    imgs: list[np.ndarray] = []
    for uf in (frame_0, frame_1, frame_2):
        im = await _decode_one(uf)
        if im is not None:
            imgs.append(im)

    if not imgs:
        return JSONResponse(
            content={
                "club_type": "UNKNOWN",
                "club_group": "IRON",
                "confidence": 0.0,
                "hand": "R",
            }
        )

    region = "CN" if (request.headers.get("CF-IPCountry") or "").upper() == "CN" else "global"
    merged = await detect_club_multiframe_bgr(imgs, region)
    hand = merged.get("hand")
    if hand not in ("R", "L"):
        hand = "R"
    return JSONResponse(
        content={
            "club_type": str(merged.get("club_type") or "UNKNOWN"),
            "club_group": str(merged.get("club_group") or "IRON"),
            "confidence": float(merged.get("confidence") or 0.0),
            "hand": hand,
        }
    )


@router.post("/vision-classic")
async def vision_classic_multipart(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Classic Lite vision (Gemini Files + Qwen fallback) — product JSON only."""
    from services.vision_classic_lite_service import VisionClassicLiteError, run_vision_classic_lite_sync

    tmp_path: Optional[str] = None
    try:
        ct = (request.headers.get("content-type") or "").lower()
        if "multipart" not in ct:
            raise HTTPException(status_code=400, detail="Expected multipart form data")
        form = await request.form()
        uploaded = form.get("file")
        file_uri_raw = form.get("file_uri")
        mime_raw = form.get("mime_type")
        key_raw = form.get("gemini_key_index")

        file_uri = str(file_uri_raw).strip() if file_uri_raw else None
        if file_uri == "":
            file_uri = None
        mime_type = str(mime_raw or "video/mp4") or "video/mp4"
        key_hint: Optional[int] = None
        if key_raw is not None and str(key_raw).strip() != "":
            try:
                key_hint = int(str(key_raw))
            except ValueError:
                key_hint = None

        file_bytes: Optional[bytes] = None
        filename = "video.mp4"
        if uploaded is not None and hasattr(uploaded, "read"):
            raw = await uploaded.read()
            if raw:
                file_bytes = raw
                filename = getattr(uploaded, "filename", None) or "video.mp4"

        if file_bytes:
            suffix = temp_suffix_for_uploaded_video(filename)
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            try:
                os.write(fd, file_bytes)
            finally:
                os.close(fd)

        region_cn = (request.headers.get("CF-IPCountry") or "").upper() == "CN"
        out = await asyncio.to_thread(
            run_vision_classic_lite_sync,
            tmp_path,
            file_uri,
            mime_type,
            filename,
            region_cn,
            key_hint,
        )
        log_non_finite_if_any(logger, out, "vision_classic")
        return JSONResponse(content=sanitize_json_floats(out))
    except VisionClassicLiteError as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.message})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[vision-classic] failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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
