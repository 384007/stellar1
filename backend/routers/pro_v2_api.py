"""Pro v2 HTTP API — frontend should call this route only (not legacy Pro orchestrators)."""

from __future__ import annotations

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from routers.auth import get_current_user
from routers.plus_analyze import _stellar_modal_upload_echo
from services.pro_v2_video_analysis_service import run_pro_v2_video_analysis

router = APIRouter(prefix="/pro-v2", tags=["pro-v2"])
logger = logging.getLogger(__name__)

_PRO_V2_MEDIA_ROOT = Path(os.getenv("STELLAR_PRO_V2_MEDIA_ROOT", "/tmp/stellar_pro_v2_media")).resolve()


def _pro_v2_client_region(request: Request) -> str:
    hint = (request.headers.get("X-Stellar-Network-Hint") or "").strip().lower()
    if hint in ("cn", "china", "mainland"):
        return "CN"
    if (request.headers.get("CF-IPCountry") or "").upper() == "CN":
        return "CN"
    return "global"


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
    """Stellar Pro v2: 240fps → motion swing window → dense scan → 8 motion keyframes → OpenCV impact → gate → AI report."""
    if current_user and not current_user.get("is_pro"):
        raise HTTPException(status_code=403, detail="Pro membership required")

    _stellar_modal_upload_echo("PRO_V2", request)
    logger.info("[PRO_V2][API] screen_mode=%s", "true" if screen_mode else "false")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    region = _pro_v2_client_region(request)

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
            result = await run_pro_v2_video_analysis(
                input_path,
                work_dir,
                rough_impact_time_s=rough_impact_time_s,
                screen_mode=screen_mode,
                region=region,
            )
            analysis_id = str(result.get("analysis_id") or "").strip()
            if not analysis_id:
                raise RuntimeError("pro_v2_analyze failed: missing analysis_id")
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

            logger.info("[PRO_V2][MEDIA] original_video_url=%s", result["original_video_url"])
            logger.info("[PRO_V2][MEDIA] playback_video_url=%s", result["playback_video_url"])
            logger.info("[PRO_V2][MEDIA] video_url=%s", result["video_url"])
            return result
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"pro_v2_analyze failed: {exc}") from exc
