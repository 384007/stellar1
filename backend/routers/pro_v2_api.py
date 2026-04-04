"""Pro HTTP API — `/pro-v2/analyze` is the stable frontend path; the engine is Pro v3 keyframes only."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from routers.auth import get_current_user
from services.internal.prov3_ffmpeg import FFmpegNotFoundError
from services.pro_prov3_analyze_service import run_pro_video_analyze_via_prov3
from services.pro_prov3_gemini_enrich import enrich_pro_prov3_response

router = APIRouter(prefix="/pro-v2", tags=["pro-v2"])
logger = logging.getLogger(__name__)

_PRO_V2_MEDIA_ROOT = Path(os.getenv("STELLAR_PRO_V2_MEDIA_ROOT", "/tmp/stellar_pro_v2_media")).resolve()


def _pro_analyze_ingress_echo(route: str, request: Request) -> None:
    """One-line ingress trace (replaces plus_analyze helper — no Plus dependency)."""
    host = (request.headers.get("host") or "").lower()
    modal_host = ".modal.run" in host
    modal_env = bool(os.getenv("MODAL_REGION")) or (os.getenv("STELLAR_RUNTIME") or "").lower() == "modal"
    runtime = (os.getenv("STELLAR_RUNTIME") or "").lower() or "unknown"
    msg = (
        f"[stellar-ingress] route={route} method={request.method} path={request.url.path} "
        f"host={host!r} runtime={runtime} modal_host={int(modal_host)} modal_env={int(modal_env)}"
    )
    print(msg, flush=True)
    print(msg, flush=True, file=sys.stderr)
    logger.info("%s", msg)


def _safe_analysis_media_dir(analysis_id: str) -> Path:
    safe_id = "".join(ch for ch in (analysis_id or "") if ch.isalnum() or ch in ("-", "_"))
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid analysis id")
    out = (_PRO_V2_MEDIA_ROOT / safe_id).resolve()
    if _PRO_V2_MEDIA_ROOT not in out.parents and out != _PRO_V2_MEDIA_ROOT:
        raise HTTPException(status_code=400, detail="Invalid media path")
    return out


@router.get("/media/{analysis_id}/{filename}")
async def pro_v2_media_file(analysis_id: str, filename: str):
    media_dir = _safe_analysis_media_dir(analysis_id)
    safe_name = Path(filename).name
    target = (media_dir / safe_name).resolve()
    if media_dir not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(str(target))


@router.post("/analyze")
async def pro_v2_analyze(
    request: Request,
    file: UploadFile = File(...),
    rough_impact_time_s: Optional[float] = Form(default=None),
    screen_mode: bool = Form(default=False),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Stellar Pro — Pro v3 keyframe pipeline (eight phases, thumbnails, optional contact sheet)."""
    if current_user and not current_user.get("is_pro"):
        raise HTTPException(status_code=403, detail="Pro membership required")

    _pro_analyze_ingress_echo("PRO_PROV3", request)
    logger.info("[PRO_PROV3][API] screen_mode=%s", "true" if screen_mode else "false")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"

    with tempfile.TemporaryDirectory(prefix="stellar_pro_v2_") as tmpdir:
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

            original_name = f"original{suffix}"
            original_path = media_dir / original_name
            with open(original_path, "wb") as f:
                f.write(body)
            original_video_url = str(request.base_url).rstrip("/") + f"/pro-v2/media/{analysis_id}/{original_name}"

            playback_src = Path(str(result.get("playback_video_url") or result.get("video_url") or ""))
            playback_video_url = ""
            if playback_src.exists() and playback_src.is_file():
                playback_name = f"playback{playback_src.suffix or '.mp4'}"
                playback_dst = media_dir / playback_name
                shutil.copy2(playback_src, playback_dst)
                playback_video_url = str(request.base_url).rstrip("/") + f"/pro-v2/media/{analysis_id}/{playback_name}"

            screen_cropped_video_url = ""
            screen_src = Path(str(result.get("screen_cropped_video_url") or ""))
            if screen_src.exists() and screen_src.is_file():
                screen_name = f"screen_cropped{screen_src.suffix or '.mp4'}"
                screen_dst = media_dir / screen_name
                shutil.copy2(screen_src, screen_dst)
                screen_cropped_video_url = str(request.base_url).rstrip("/") + f"/pro-v2/media/{analysis_id}/{screen_name}"

            contact_src = Path(str(result.get("contact_sheet_url") or ""))
            if contact_src.exists() and contact_src.is_file():
                contact_name = f"contact_sheet{contact_src.suffix or '.jpg'}"
                contact_dst = media_dir / contact_name
                shutil.copy2(contact_src, contact_dst)
                result["contact_sheet_url"] = (
                    str(request.base_url).rstrip("/") + f"/pro-v2/media/{analysis_id}/{contact_name}"
                )

            result["original_video_url"] = original_video_url
            result["video_url"] = original_video_url
            result["playback_video_url"] = playback_video_url or original_video_url
            if screen_cropped_video_url:
                result["screen_cropped_video_url"] = screen_cropped_video_url
            result["screen_mode"] = bool(screen_mode)

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
