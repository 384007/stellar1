"""Pro v2 HTTP API — frontend should call this route only (not legacy Pro orchestrators)."""

from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from routers.auth import get_current_user
from routers.plus_analyze import _stellar_modal_upload_echo
from services.pro_v2_video_analysis_service import run_pro_v2_video_analysis

router = APIRouter(prefix="/pro-v2", tags=["pro-v2"])
logger = logging.getLogger(__name__)


def _pro_v2_client_region(request: Request) -> str:
    hint = (request.headers.get("X-Stellar-Network-Hint") or "").strip().lower()
    if hint in ("cn", "china", "mainland"):
        return "CN"
    if (request.headers.get("CF-IPCountry") or "").upper() == "CN":
        return "CN"
    return "global"


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
            return await run_pro_v2_video_analysis(
                input_path,
                work_dir,
                rough_impact_time_s=rough_impact_time_s,
                screen_mode=screen_mode,
                region=region,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"pro_v2_analyze failed: {exc}") from exc
