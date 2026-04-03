from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from routers.auth import get_current_user
from routers.plus_analyze import _stellar_modal_upload_echo
from services.pro_analysis_chain_service import ProAnalysisChainSettings
from services.stellar_pro_video_analysis_service import run_stellar_pro_video_analysis

router = APIRouter(prefix='/stellar-pro', tags=['stellar-pro'])


@router.post('/analyze')
async def stellar_pro_analyze(
    request: Request,
    file: UploadFile = File(...),
    rough_impact_time_s: Optional[float] = Form(default=None),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Pro 唯一入口：240fps → Pose → 动作窗 → 动作中心选帧 → OpenCV impact → Gemini 文案。"""
    if current_user and not current_user.get('is_pro'):
        raise HTTPException(status_code=403, detail='Pro membership required')

    _stellar_modal_upload_echo('STELLAR_PRO', request)

    suffix = Path(file.filename or 'video.mp4').suffix or '.mp4'
    region = 'CN' if (request.headers.get('CF-IPCountry') or '').upper() == 'CN' else 'global'

    with tempfile.TemporaryDirectory(prefix='stellar_pro_analyze_') as tmpdir:
        input_path = str(Path(tmpdir) / f'input{suffix}')
        work_dir = os.path.join(tmpdir, 'work')
        os.makedirs(work_dir, exist_ok=True)
        body = await file.read()
        if not body:
            raise HTTPException(status_code=400, detail='Empty file')
        with open(input_path, 'wb') as f:
            f.write(body)

        try:
            return await run_stellar_pro_video_analysis(
                input_path,
                work_dir,
                rough_impact_time_s=rough_impact_time_s,
                region=region,
                chain_settings=ProAnalysisChainSettings(),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'stellar_pro_analyze failed: {exc}') from exc
