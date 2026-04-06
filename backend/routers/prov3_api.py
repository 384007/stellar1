"""Pro v3 API — single HTTP surface under ``/pro-v3``.

* ``POST /pro-v3/analyze`` — product (keyframes + optional Gemini report + media URLs)
* ``GET /pro-v3/media/{analysis_id}/{filename}`` — persisted originals / playback / contact sheet
* ``POST /pro-v3/keyframes/*`` — preprocess / extract / refine / analyze (raw pipeline, no Gemini)
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from lib.prov3.keyframes.types import ExtractRequest, RefineRequest
from routers.auth import get_current_user
from services.internal.prov3_ffmpeg import FFmpegNotFoundError
from services.prov3_analyze_control import (
    PROV3_ANALYZE_CANCELLED,
    prov3_analyze_in_flight_count,
    prov3_begin_analyze,
    prov3_check_cancelled,
    prov3_finish_analyze,
    prov3_request_cancel,
)
from services.pro_prov3_analyze_service import run_pro_video_analyze_via_prov3
from services.pro_prov3_gemini_enrich import enrich_pro_prov3_response
from services.prov3_keyframe_a_extractor_service import run_a_extract
from services.prov3_keyframe_b_refiner_service import run_b_refine
from services.prov3_keyframe_orchestrator_service import run_keyframe_analyze
from services.prov3_keyframe_preprocess_service import run_preprocess

logger = logging.getLogger(__name__)

_PRO_MEDIA_ROOT = Path(os.getenv("STELLAR_PRO_V3_MEDIA_ROOT") or "/tmp/stellar_prov3_media").resolve()

_MODAL_PRO_ANALYZE_LOCK = asyncio.Lock()


def _modal_pro_single_flight_enabled() -> bool:
    if (os.getenv("STELLAR_RUNTIME") or "").strip().lower() == "modal":
        return True
    if (os.getenv("MODAL_REGION") or "").strip():
        return True
    v3 = (os.getenv("STELLAR_MODAL_PRO_V3_ONLY") or "").strip().lower()
    return v3 in ("1", "true", "yes")


def _single_flight_disabled_by_env() -> bool:
    """Set ``STELLAR_PROV3_ANALYZE_SINGLE_FLIGHT=0`` to allow concurrent ``POST /pro-v3/analyze`` (OOM risk on Modal)."""
    v = (os.getenv("STELLAR_PROV3_ANALYZE_SINGLE_FLIGHT") or "").strip().lower()
    return v in ("0", "false", "no", "off")


def prov3_analyze_single_flight_active() -> bool:
    """Used by ``/health`` and ``_run_pro_analyze`` — when True, a second analyze gets HTTP 409 while one holds the lock."""
    if _single_flight_disabled_by_env():
        return False
    return _modal_pro_single_flight_enabled()


router_pro_v3 = APIRouter(prefix="/pro-v3", tags=["pro-v3"])
router_keyframes = APIRouter(prefix="/keyframes", tags=["pro-v3-keyframes"])

router = APIRouter()


@router_keyframes.post("/preprocess")
async def prov3_keyframes_preprocess(file: UploadFile = File(...), screen_mode: bool = Form(default=False)):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="prov3_preprocess_") as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty file")
        input_path.write_bytes(payload)
        try:
            result = run_preprocess(str(input_path), tmpdir, screen_mode=screen_mode)
        except FFmpegNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.model_dump()


@router_keyframes.post("/extract")
async def prov3_keyframes_extract(req: ExtractRequest):
    result = run_a_extract(
        analysis_id=req.analysis_id,
        analysis_video=req.analysis_video,
        preprocess_meta=req.preprocess_meta,
        analysis_frames=req.analysis_frames,
    )
    return result.model_dump()


@router_keyframes.post("/refine")
async def prov3_keyframes_refine(req: RefineRequest):
    result = run_b_refine(
        analysis_id=req.analysis_id,
        analysis_video=req.analysis_video,
        preprocess_meta=req.preprocess_meta,
        analysis_frames=req.analysis_frames,
        enhanced_local_frames=req.enhanced_local_frames,
        keyframes=req.keyframes,
        confidence=req.confidence,
        fail_reasons=req.fail_reasons,
    )
    return result.model_dump()


@router_keyframes.post("/analyze")
async def prov3_keyframes_analyze(file: UploadFile = File(...), screen_mode: bool = Form(default=False)):
    """预处理 + A/B 关键帧-only JSON（无 Gemini、无媒体落盘）。产品完整分析请用 ``POST /pro-v3/analyze``。"""
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="prov3_analyze_") as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty file")
        input_path.write_bytes(payload)
        try:
            result = run_keyframe_analyze(str(input_path), tmpdir, screen_mode=screen_mode)
        except FFmpegNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.model_dump(exclude={"analysis_video", "analysis_fps", "source_fps"})


def _pro_analyze_ingress_echo(route: str, request: Request) -> None:
    host = (request.headers.get("host") or "").lower()
    modal_host = ".modal.run" in host
    modal_env = bool(os.getenv("MODAL_REGION")) or (os.getenv("STELLAR_RUNTIME") or "").lower() == "modal"
    runtime = (os.getenv("STELLAR_RUNTIME") or "").lower() or "unknown"
    msg = (
        f"[stellar-ingress] route={route} method={request.method} path={request.url.path} "
        f"host={host!r} runtime={runtime} modal_host={int(modal_host)} modal_env={int(modal_env)}"
    )
    logger.info("%s", msg)


def _safe_analysis_media_dir(analysis_id: str) -> Path:
    safe_id = "".join(ch for ch in (analysis_id or "") if ch.isalnum() or ch in ("-", "_"))
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid analysis id")
    out = (_PRO_MEDIA_ROOT / safe_id).resolve()
    if _PRO_MEDIA_ROOT not in out.parents and out != _PRO_MEDIA_ROOT:
        raise HTTPException(status_code=400, detail="Invalid media path")
    return out


async def _pro_media_file_handler(analysis_id: str, filename: str) -> FileResponse:
    media_dir = _safe_analysis_media_dir(analysis_id)
    safe_name = Path(filename).name
    target = (media_dir / safe_name).resolve()
    if media_dir not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(str(target))


@router_pro_v3.get("/media/{analysis_id}/{filename}")
async def pro_v3_media_file(analysis_id: str, filename: str):
    return await _pro_media_file_handler(analysis_id, filename)


@router_pro_v3.post("/analyze/cancel")
async def prov3_analyze_cancel_http():
    """Request cooperative cancellation of the current ``POST /pro-v3/analyze`` on this worker."""
    prov3_request_cancel()
    return {
        "ok": True,
        "in_flight": prov3_analyze_in_flight_count(),
        "cancel_requested": True,
    }


async def _run_pro_analyze(
    request: Request,
    file: UploadFile,
    rough_impact_time_s: Optional[float],
    screen_mode: bool,
    current_user: Optional[dict],
    *,
    media_prefix: str,
    ingress_tag: str,
) -> dict:
    if current_user and not current_user.get("is_pro"):
        raise HTTPException(status_code=403, detail="Pro membership required")

    _pro_analyze_ingress_echo(ingress_tag, request)
    rid = (request.headers.get("x-request-id") or "").strip() or secrets.token_hex(4)
    client = getattr(request.client, "host", None) or ""
    logger.info(
        "[PRO_PROV3][API] rid=%s path=%s screen_mode=%s client=%s",
        rid,
        request.url.path,
        "true" if screen_mode else "false",
        client,
    )

    if prov3_analyze_single_flight_active():
        try:
            await asyncio.wait_for(_MODAL_PRO_ANALYZE_LOCK.acquire(), timeout=0)
        except asyncio.TimeoutError:
            logger.warning(
                "[PRO_PROV3][LOCK] rid=%s reject 409 — another analyze holds the lock (or stuck job). "
                "If you did not start two analyses, the previous request may still be running after a client timeout.",
                rid,
            )
            raise HTTPException(
                status_code=409,
                detail=(
                    "当前已有视频在分析中，请稍后再试。"
                    "若您只提交了一次，可能是上一次分析仍在后台处理（例如前端已超时但服务端未结束），请等待几分钟后再试。"
                ),
            ) from None
        logger.info("[PRO_PROV3][LOCK] rid=%s acquired", rid)
        try:
            return await _run_pro_analyze_body(
                request,
                file,
                rough_impact_time_s,
                screen_mode,
                media_prefix,
            )
        finally:
            _MODAL_PRO_ANALYZE_LOCK.release()
            logger.info("[PRO_PROV3][LOCK] rid=%s released", rid)

    return await _run_pro_analyze_body(
        request,
        file,
        rough_impact_time_s,
        screen_mode,
        media_prefix,
    )


async def _run_pro_analyze_body(
    request: Request,
    file: UploadFile,
    rough_impact_time_s: Optional[float],
    screen_mode: bool,
    media_prefix: str,
) -> dict:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    mp = media_prefix.rstrip("/")

    prov3_begin_analyze()
    try:
        with tempfile.TemporaryDirectory(prefix="stellar_pro_analyze_") as tmpdir:
            input_path = str(Path(tmpdir) / f"input{suffix}")
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir, exist_ok=True)
            body = await file.read()
            if not body:
                raise HTTPException(status_code=400, detail="Empty file")
            with open(input_path, "wb") as f:
                f.write(body)

            try:
                result = await asyncio.to_thread(
                    run_pro_video_analyze_via_prov3,
                    input_path,
                    work_dir,
                    screen_mode=screen_mode,
                    rough_impact_time_s=rough_impact_time_s,
                    cancel_check=prov3_check_cancelled,
                )
                result = await enrich_pro_prov3_response(result, region="global")
                analysis_id = str(result.get("analysis_id") or "").strip()
                if not analysis_id:
                    raise RuntimeError("pro_analyze failed: missing analysis_id")
                media_dir = _safe_analysis_media_dir(analysis_id)
                media_dir.mkdir(parents=True, exist_ok=True)

                base = str(request.base_url).rstrip("/")

                original_name = f"original{suffix}"
                original_path = media_dir / original_name
                with open(original_path, "wb") as f:
                    f.write(body)
                original_video_url = f"{base}{mp}/media/{analysis_id}/{original_name}"

                playback_src = Path(str(result.get("playback_video_url") or result.get("video_url") or ""))
                playback_video_url = ""
                if playback_src.exists() and playback_src.is_file():
                    playback_name = f"playback{playback_src.suffix or '.mp4'}"
                    playback_dst = media_dir / playback_name
                    shutil.copy2(playback_src, playback_dst)
                    playback_video_url = f"{base}{mp}/media/{analysis_id}/{playback_name}"

                analysis_src = Path(str(result.get("_prov3_motion", {}).get("analysis_video") or ""))
                analysis_video_url = ""
                if analysis_src.exists() and analysis_src.is_file():
                    analysis_name = f"analysis_timeline{analysis_src.suffix or '.mp4'}"
                    analysis_dst = media_dir / analysis_name
                    shutil.copy2(analysis_src, analysis_dst)
                    analysis_video_url = f"{base}{mp}/media/{analysis_id}/{analysis_name}"

                keyframe_images = list(result.get("keyframe_images") or [])
                keyframe_url_by_file: dict[str, str] = {}
                persisted_rows: list[dict] = []
                for row in keyframe_images:
                    fp = Path(str(row.get("file_path") or ""))
                    fn = Path(str(row.get("file_name") or "")).name
                    if not fn or not fp.exists() or not fp.is_file():
                        continue
                    dst = media_dir / fn
                    shutil.copy2(fp, dst)
                    url = f"{base}{mp}/media/{analysis_id}/{fn}"
                    keyframe_url_by_file[fn] = url
                    new_row = dict(row)
                    new_row.pop("file_path", None)
                    new_row["keyframe_image_url"] = url
                    persisted_rows.append(new_row)
                if persisted_rows:
                    result["keyframe_images"] = persisted_rows

                event_to_file = {
                    "Address": "address.jpg",
                    "Toe-up": "toe_up.jpg",
                    "Mid-backswing": "mid_backswing.jpg",
                    "Top": "top.jpg",
                    "Mid-downswing": "mid_downswing.jpg",
                    "Impact": "impact.jpg",
                    "Mid-follow-through": "mid_follow_through.jpg",
                    "Finish": "finish.jpg",
                }
                phase_to_event = {
                    "address": "Address",
                    "takeaway": "Toe-up",
                    "backswing": "Mid-backswing",
                    "top": "Top",
                    "downswing": "Mid-downswing",
                    "impact": "Impact",
                    "follow_through": "Mid-follow-through",
                    "finish": "Finish",
                }
                for kf in list(result.get("keyframes") or []):
                    phase = str(kf.get("phase") or "")
                    event = phase_to_event.get(phase, "")
                    fn = event_to_file.get(event, "")
                    if fn and fn in keyframe_url_by_file:
                        kf["keyframe_image_url"] = keyframe_url_by_file[fn]
                        kf["keyframe_image_source"] = "analysis_video"
                    if "keyframe_image_path" in kf:
                        kf.pop("keyframe_image_path", None)
                for kf in list(result.get("official_phase_keyframes") or []):
                    phase = str(kf.get("phase") or "")
                    event = phase_to_event.get(phase, "")
                    fn = event_to_file.get(event, "")
                    if fn and fn in keyframe_url_by_file:
                        kf["keyframe_image_url"] = keyframe_url_by_file[fn]
                        kf["keyframe_image_source"] = "analysis_video"
                    if "keyframe_image_path" in kf:
                        kf.pop("keyframe_image_path", None)
                for kf in list(result.get("preview_keyframes") or []):
                    phase = str(kf.get("phase") or "")
                    event = phase_to_event.get(phase, "")
                    fn = event_to_file.get(event, "")
                    if fn and fn in keyframe_url_by_file:
                        kf["keyframe_image_url"] = keyframe_url_by_file[fn]
                        kf["keyframe_image_source"] = "analysis_video"
                    if "keyframe_image_path" in kf:
                        kf.pop("keyframe_image_path", None)

                screen_cropped_video_url = ""
                screen_src = Path(str(result.get("screen_cropped_video_url") or ""))
                if screen_src.exists() and screen_src.is_file():
                    screen_name = f"screen_cropped{screen_src.suffix or '.mp4'}"
                    screen_dst = media_dir / screen_name
                    shutil.copy2(screen_src, screen_dst)
                    screen_cropped_video_url = f"{base}{mp}/media/{analysis_id}/{screen_name}"

                contact_src = Path(str(result.get("contact_sheet_url") or ""))
                if contact_src.exists() and contact_src.is_file():
                    contact_name = f"contact_sheet{contact_src.suffix or '.jpg'}"
                    contact_dst = media_dir / contact_name
                    shutil.copy2(contact_src, contact_dst)
                    result["contact_sheet_url"] = f"{base}{mp}/media/{analysis_id}/{contact_name}"

                result["original_video_url"] = original_video_url
                result["video_url"] = original_video_url
                result["playback_video_url"] = playback_video_url or original_video_url
                if analysis_video_url:
                    result["analysis_video_url"] = analysis_video_url
                if screen_cropped_video_url:
                    result["screen_cropped_video_url"] = screen_cropped_video_url
                result["screen_mode"] = bool(screen_mode)
                result["pro_http_path"] = mp
                result.pop("_prov3_motion", None)
                result.pop("prov3", None)

                logger.info("[PRO_PROV3][MEDIA] original_video_url=%s", result["original_video_url"])
                logger.info("[PRO_PROV3][MEDIA] playback_video_url=%s", result["playback_video_url"])
                logger.info("[PRO_PROV3][MEDIA] video_url=%s", result["video_url"])
                return result
            except FFmpegNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except RuntimeError as exc:
                if str(exc) == PROV3_ANALYZE_CANCELLED:
                    raise HTTPException(status_code=422, detail="分析已取消") from exc
                logger.warning(
                    "[PRO_PROV3][API] analyze RuntimeError (422) file=%s: %s",
                    file.filename,
                    exc,
                )
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception(
                    "[PRO_PROV3][API] analyze failed (500) file=%s suffix=%s: %s",
                    file.filename,
                    suffix,
                    exc,
                )
                raise HTTPException(status_code=500, detail=f"pro_analyze failed: {exc}") from exc
    finally:
        prov3_finish_analyze()


@router_pro_v3.post("/analyze")
async def pro_v3_analyze(
    request: Request,
    file: UploadFile = File(...),
    rough_impact_time_s: Optional[float] = Form(default=None),
    screen_mode: bool = Form(default=False),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Stellar Pro — ``POST /pro-v3/analyze``."""
    return await _run_pro_analyze(
        request,
        file,
        rough_impact_time_s,
        screen_mode,
        current_user,
        media_prefix="/pro-v3",
        ingress_tag="PRO_PROV3",
    )


# Mount after all routes are registered (include_router snapshots routes at call time).
router_pro_v3.include_router(router_keyframes)
router.include_router(router_pro_v3)
