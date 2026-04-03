from __future__ import annotations

from pathlib import Path
from typing import Any

from services.pro_contact_sheet_service import build_pro_keyframe_contact_sheet
from services.pro_minimal_public_result_service import pack_pro_minimal_public_result
from services.stellar_pro_role_log import (
    ROLE_CONTACT_SHEET,
    ROLE_PRO_PUBLIC_PACK,
    log_stage_done,
    log_stage_failed,
    log_stage_start,
)


def finalize_stellar_pro_result(
    raw_result: dict[str, Any],
    *,
    work_dir: str,
    contact_sheet_filename: str = 'contact_sheet.jpg',
) -> dict[str, Any]:
    """Finalize backend Pro output into the minimal frontend payload.

    Steps:
    1. generate contact sheet if keyframes are available
    2. attach contact_sheet_url/path into raw_result
    3. return minimal public result only
    """
    result = dict(raw_result or {})
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    keyframes = list(result.get('keyframes') or [])
    if keyframes:
        try:
            log_stage_start(ROLE_CONTACT_SHEET, keyframes=len(keyframes))
            out_path = str(work / contact_sheet_filename)
            build_pro_keyframe_contact_sheet(keyframes, out_path)
            result['contact_sheet_url'] = out_path
            log_stage_done(ROLE_CONTACT_SHEET, output=out_path)
        except Exception as exc:
            log_stage_failed(ROLE_CONTACT_SHEET, reason=type(exc).__name__, detail=str(exc))

    log_stage_start(ROLE_PRO_PUBLIC_PACK)
    public_result = pack_pro_minimal_public_result(result)
    log_stage_done(ROLE_PRO_PUBLIC_PACK, keyframes=len(list(public_result.get('keyframes') or [])))
    return public_result
