from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from services.shot_tracer_reconstruct_service import run_shot_tracer_reconstruct

router = APIRouter(prefix="/shot-tracer", tags=["shot-tracer"])


def _tmp_suffix(name: str | None, default: str = ".mp4") -> str:
    if not name:
        return default
    p = Path(name)
    return p.suffix if p.suffix else default


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
        def save_upload(upload: UploadFile) -> str:
            with tempfile.NamedTemporaryFile(delete=False, suffix=_tmp_suffix(upload.filename)) as f:
                f.write(upload.file.read())
                tmp_paths.append(f.name)
                return f.name

        main_path = save_upload(file)
        front_path = save_upload(front_view) if front_view else None
        side_path = save_upload(side_view) if side_view else None

        data = run_shot_tracer_reconstruct(
            video_path=main_path,
            front_view_path=front_path,
            side_view_path=side_path,
            calibration_json=calibration_json,
            mode=mode,
            include_debug=bool(debug),
        )
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"shot tracer reconstruct failed: {e}")
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
