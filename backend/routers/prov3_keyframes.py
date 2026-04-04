from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.internal.prov3_ffmpeg import FFmpegNotFoundError

from lib.prov3.keyframes.types import ExtractRequest, RefineRequest
from services.prov3_keyframe_a_extractor_service import run_a_extract
from services.prov3_keyframe_b_refiner_service import run_b_refine
from services.prov3_keyframe_orchestrator_service import run_keyframe_analyze
from services.prov3_keyframe_preprocess_service import run_preprocess

router = APIRouter(prefix="/api/prov3/keyframes", tags=["prov3-keyframes"])


@router.post("/preprocess")
async def prov3_preprocess(file: UploadFile = File(...), screen_mode: bool = Form(default=False)):
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


@router.post("/extract")
async def prov3_extract(req: ExtractRequest):
    result = run_a_extract(
        analysis_id=req.analysis_id,
        analysis_video=req.analysis_video,
        preprocess_meta=req.preprocess_meta,
        analysis_frames=req.analysis_frames,
    )
    return result.model_dump()


@router.post("/refine")
async def prov3_refine(req: RefineRequest):
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


@router.post("/analyze")
async def prov3_analyze(file: UploadFile = File(...), screen_mode: bool = Form(default=False)):
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
