"""Pro v3 API — single HTTP surface under ``/pro-v3``.

* ``POST /pro-v3/analyze`` — product (keyframes + optional Gemini report + media URLs)
* ``GET /pro-v3/media/{analysis_id}/{filename}`` — persisted originals / playback / contact sheet
* ``POST /pro-v3/keyframes/*`` — preprocess / extract / refine / analyze (raw pipeline, no Gemini)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from lib.prov3.keyframes.types import ExtractRequest, RefineRequest
from routers.auth import get_current_user
from services.internal.prov3_ffmpeg import FFmpegNotFoundError
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
    logger.info("[PRO_PROV3][API] path=%s screen_mode=%s", request.url.path, "true" if screen_mode else "false")

    if _modal_pro_single_flight_enabled():
        try:
            await asyncio.wait_for(_MODAL_PRO_ANALYZE_LOCK.acquire(), timeout=0)
        except asyncio.TimeoutError:
            logger.warning("[PRO_PROV3][API] reject concurrent analyze (Modal single-flight)")
            raise HTTPException(
                status_code=409,
                detail="当前已有视频在分析中，请稍后再试。",
            ) from None
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
            if screen_cropped_video_url:
                result["screen_cropped_video_url"] = screen_cropped_video_url
            result["screen_mode"] = bool(screen_mode)
            result["pro_http_path"] = mp

            logger.info("[PRO_PROV3][MEDIA] original_video_url=%s", result["original_video_url"])
            logger.info("[PRO_PROV3][MEDIA] playback_video_url=%s", result["playback_video_url"])
            logger.info("[PRO_PROV3][MEDIA] video_url=%s", result["video_url"])
            return result
        except FFmpegNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"pro_analyze failed: {exc}") from exc


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
