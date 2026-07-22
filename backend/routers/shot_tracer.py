from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from services.shot_tracer_reconstruct_service import run_shot_tracer_reconstruct

router = APIRouter(prefix="/shot-tracer", tags=["shot-tracer"])

_UPLOAD_CHUNK_SIZE = 1024 * 1024


def _tmp_suffix(name: str | None, default: str = ".mp4") -> str:
    if not name:
        return default
    suffix = Path(name).suffix
    return suffix if suffix else default


async def _save_upload_chunked(upload: UploadFile, tmp_paths: list[str]) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=_tmp_suffix(upload.filename)) as f:
        tmp_paths.append(f.name)
        while True:
            chunk = await upload.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
        return f.name


@router.post("/reconstruct")
async def reconstruct_shot_tracer(
    file: UploadFile = File(...),
    front_view: UploadFile | None = File(None),
    side_view: UploadFile | None = File(None),
    calibration_json: str | None = Form(None),
    mode: str = Form("single_video"),
    debug: int = Query(0),
):
    allowed_modes = {"single_video", "dual_camera", "high_speed", "3d_scene"}
    if mode not in allowed_modes:
        raise HTTPException(status_code=400, detail=f"invalid mode: {mode}")

    tmp_paths: list[str] = []
    try:
        main_path = await _save_upload_chunked(file, tmp_paths)
        front_path = await _save_upload_chunked(front_view, tmp_paths) if front_view else None
        side_path = await _save_upload_chunked(side_view, tmp_paths) if side_view else None

        return await asyncio.to_thread(
            run_shot_tracer_reconstruct,
            video_path=main_path,
            front_view_path=front_path,
            side_view_path=side_path,
            calibration_json=calibration_json,
            mode=mode,
            include_debug=bool(debug),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"shot tracer reconstruct failed: {e}") from e
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass
