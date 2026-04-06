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

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from lib.prov3.keyframes.types import ExtractRequest, RefineRequest
from lib.prov3.r2_media import (
    prov3_async_job_result_key,
    prov3_async_job_status_key,
    prov3_r2_media_fully_configured,
    prov3_r2_object_key,
    prov3_r2_public_url_for_key,
    r2_download_object_to_path,
    r2_head_object_exists,
    r2_put_json_object,
    upload_prov3_media_directory_to_r2_and_verify,
)
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
from services.gemini_service import gemini_modal_cn_proxy_first_context
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


def _prov3_durable_media_required() -> bool:
    """Product prov3 responses must not reference ephemeral worker disk only. Opt out for local dev."""
    v = (os.getenv("STELLAR_PROV3_REQUIRE_R2") or "").strip().lower()
    return v not in ("0", "false", "no", "off")


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


def _prov3_cn_network_hint_from_request(request: Request) -> bool:
    """China mainland hint for Modal Gemini routing: client header and/or Cloudflare country (forwarded by Edge)."""
    if request.headers.get("x-stellar-network-hint", "").strip().lower() == "cn":
        return True
    cc = (request.headers.get("cf-ipcountry") or request.headers.get("CF-IPCountry") or "").strip().upper()
    return cc == "CN"


class Prov3AnalyzeStartBody(BaseModel):
    """Start a background Pro v3 analyze from a video already stored in R2 (same-origin upload)."""

    source_r2_key: str = Field(..., min_length=6, max_length=512)
    screen_mode: bool = False
    rough_impact_time_s: Optional[float] = None


def _validate_user_video_r2_key(source_r2_key: str, user_id: str) -> str:
    k = (source_r2_key or "").strip().lstrip("/")
    if not k or ".." in k:
        raise HTTPException(status_code=400, detail="Invalid source_r2_key")
    uid = user_id.strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token")
    prefix = f"videos/{uid}/"
    if not k.startswith(prefix):
        raise HTTPException(status_code=403, detail="Video key does not belong to this account")
    rest = k[len(prefix) :]
    if not rest or "/" in rest:
        raise HTTPException(status_code=400, detail="Invalid source_r2_key shape")
    return k


async def _prov3_write_async_job_status(job_id: str, payload: dict) -> None:
    key = prov3_async_job_status_key(job_id)
    await asyncio.to_thread(r2_put_json_object, key, payload)


async def _prov3_analyze_async_worker(
    job_id: str,
    source_r2_key: str,
    screen_mode: bool,
    rough_impact_time_s: Optional[float],
    api_base_url: str,
    media_prefix: str,
    lock_acquired: bool,
    user_id: str,
    *,
    cn_network_hint: bool = False,
) -> None:
    """Runs after ``/analyze/start`` returns; updates R2 job records; releases Modal single-flight lock."""
    try:
        await _prov3_write_async_job_status(
            job_id,
            {
                "status": "running",
                "job_id": job_id,
                "user_id": user_id,
            },
        )
        suffix = Path(source_r2_key).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="stellar_pro_async_dl_") as dl_tmp:
            dl_path = Path(dl_tmp) / f"input{suffix}"
            try:
                await asyncio.to_thread(r2_download_object_to_path, source_r2_key, dl_path)
            except FileNotFoundError as exc:
                await _prov3_write_async_job_status(
                    job_id,
                    {
                        "status": "failed",
                        "job_id": job_id,
                        "user_id": user_id,
                        "detail": str(exc) or "Video missing in R2",
                    },
                )
                return
            body_bytes = dl_path.read_bytes()
        if not body_bytes:
            await _prov3_write_async_job_status(
                job_id,
                {
                    "status": "failed",
                    "job_id": job_id,
                    "user_id": user_id,
                    "detail": "Downloaded video is empty",
                },
            )
            return

        fname = Path(source_r2_key).name or f"upload{suffix}"
        result = await _run_pro_analyze_body(
            api_base_url,
            body_bytes,
            fname,
            rough_impact_time_s,
            screen_mode,
            media_prefix,
            cn_network_hint=cn_network_hint,
        )
        rkey = prov3_async_job_result_key(job_id)
        await asyncio.to_thread(r2_put_json_object, rkey, {"result": result})
        aid = str(result.get("analysis_id") or "")
        await _prov3_write_async_job_status(
            job_id,
            {
                "status": "completed",
                "job_id": job_id,
                "user_id": user_id,
                "analysis_id": aid,
            },
        )
        logger.info("[PRO_PROV3][ASYNC] job_id=%s analysis_id=%s completed", job_id, aid)
    except HTTPException as exc:
        det = exc.detail
        detail_for_failed = det if isinstance(det, str) else str(det)
        await _prov3_write_async_job_status(
            job_id,
            {
                "status": "failed",
                "job_id": job_id,
                "user_id": user_id,
                "detail": detail_for_failed,
                "http_status": int(exc.status_code),
            },
        )
        logger.warning("[PRO_PROV3][ASYNC] job_id=%s HTTPException %s", job_id, exc.status_code)
    except Exception as exc:
        await _prov3_write_async_job_status(
            job_id,
            {
                "status": "failed",
                "job_id": job_id,
                "user_id": user_id,
                "detail": str(exc)[:2000],
            },
        )
        logger.exception("[PRO_PROV3][ASYNC] job_id=%s failed: %s", job_id, exc)
    finally:
        if lock_acquired:
            try:
                _MODAL_PRO_ANALYZE_LOCK.release()
                logger.info("[PRO_PROV3][ASYNC] job_id=%s lock released", job_id)
            except RuntimeError:
                pass


def _safe_analysis_media_dir(analysis_id: str) -> Path:
    safe_id = "".join(ch for ch in (analysis_id or "") if ch.isalnum() or ch in ("-", "_"))
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid analysis id")
    out = (_PRO_MEDIA_ROOT / safe_id).resolve()
    if _PRO_MEDIA_ROOT not in out.parents and out != _PRO_MEDIA_ROOT:
        raise HTTPException(status_code=400, detail="Invalid media path")
    return out


def _inject_keyframe_urls(rows: list[dict], keyframe_url_by_file: dict[str, str]) -> None:
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
    for kf in rows:
        phase = str(kf.get("phase") or "").strip().lower()
        event = str(kf.get("event_name") or "").strip() or phase_to_event.get(phase, "")
        fn = event_to_file.get(event, "")
        if fn and fn in keyframe_url_by_file:
            kf["keyframe_image_url"] = keyframe_url_by_file[fn]
            kf["keyframe_image_source"] = "analysis_video"
        if "keyframe_image_path" in kf:
            kf.pop("keyframe_image_path", None)


_MIN_PROV3_JPEG_BYTES = 256


def _prov3_strip_keyframe_b64(rows: list) -> None:
    for r in rows:
        if isinstance(r, dict):
            r.pop("image_base64", None)


def _prov3_assert_row_media_on_disk(row: dict, media_dir: Path) -> None:
    url = str(row.get("keyframe_image_url") or "").strip()
    if not url:
        raise RuntimeError("prov3_media_gate:missing_keyframe_image_url")
    fn = Path(url.split("?")[0]).name
    low = fn.lower()
    if not low.endswith((".jpg", ".jpeg")):
        raise RuntimeError(f"prov3_media_gate:keyframe_not_jpeg:{fn}")
    dest = (media_dir / fn).resolve()
    if media_dir not in dest.parents and dest != media_dir:
        raise RuntimeError(f"prov3_media_gate:bad_keyframe_path:{fn}")
    if not dest.is_file() or dest.stat().st_size < _MIN_PROV3_JPEG_BYTES:
        raise RuntimeError(f"prov3_media_gate:keyframe_file_missing_or_tiny:{fn}")


def _prov3_assert_timeline_video_on_disk(media_dir: Path) -> None:
    ok = False
    for p in media_dir.iterdir():
        if p.is_file() and p.name.startswith("analysis_timeline") and p.stat().st_size > 4096:
            ok = True
            break
    if not ok:
        raise RuntimeError("prov3_media_gate:analysis_timeline_video_missing_on_disk")


def _rewrite_prov3_result_urls_to_r2(result: dict, r2_by_fn: dict[str, str]) -> None:
    """Swap Modal ``/pro-v3/media/…`` URLs for R2 public URLs after upload."""
    if not r2_by_fn:
        return
    for key in (
        "original_video_url",
        "video_url",
        "playback_video_url",
        "analysis_video_url",
        "screen_cropped_video_url",
        "contact_sheet_url",
    ):
        url = result.get(key)
        if not isinstance(url, str) or not url.strip():
            continue
        fn = Path(url.split("?")[0]).name
        if fn in r2_by_fn:
            result[key] = r2_by_fn[fn]
    for list_key in ("keyframes", "official_phase_keyframes", "preview_keyframes", "keyframe_images"):
        rows = result.get(list_key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            u = row.get("keyframe_image_url")
            if not isinstance(u, str):
                continue
            fn = Path(u.split("?")[0]).name
            if fn in r2_by_fn:
                row["keyframe_image_url"] = r2_by_fn[fn]


def _prov3_validate_product_media_or_raise(result: dict, media_dir: Path) -> None:
    """Every user-visible keyframe must be a persisted true-240 timeline JPG with a public URL.

    On ``low_trust``, ``keyframes`` / ``official_phase_keyframes`` may be empty while
    ``preview_keyframes`` still holds the persisted timeline JPGs — gate those rows too.
    """
    _prov3_assert_timeline_video_on_disk(media_dir)
    if not str(result.get("analysis_video_url") or "").strip():
        raise RuntimeError("prov3_media_gate:missing_analysis_video_url")
    has_any_rows = any(
        bool(list(result.get(k) or []))
        for k in ("keyframes", "official_phase_keyframes", "preview_keyframes")
    )
    if not has_any_rows:
        raise RuntimeError("prov3_media_gate:empty_display_keyframes")
    for key in ("keyframes", "official_phase_keyframes", "preview_keyframes"):
        rows = list(result.get(key) or [])
        if not rows:
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"prov3_media_gate:invalid_row:{key}:{i}")
            _prov3_assert_row_media_on_disk(row, media_dir)
    _prov3_strip_keyframe_b64(list(result.get("keyframes") or []))
    _prov3_strip_keyframe_b64(list(result.get("official_phase_keyframes") or []))
    _prov3_strip_keyframe_b64(list(result.get("preview_keyframes") or []))


def _log_prov3_frontend_media_playbook(result: dict, *, analysis_id: str, context: str) -> None:
    """Log why Cloudflare/Next 「分析结果页」 may show no video or keyframes — for Modal ops debugging.

    Typical causes: non-https video strings, missing ``keyframe_image_url`` on rows, low-trust empty official strip,
    or browser cannot fetch Modal/R2 URLs (CORS / wrong ``NEXT_PUBLIC_MODAL_BACKEND_URL`` / expired worker URLs).
    """
    def _url_kind(u: str) -> str:
        s = (u or "").strip()
        if not s:
            return "empty"
        if s.startswith("https://"):
            return "https"
        if s.startswith("http://"):
            return "http"
        if s.startswith("/"):
            return "path_only"
        return "non_http"

    def _count_rows_with_url(key: str) -> tuple[int, int]:
        rows = list(result.get(key) or [])
        n = len(rows)
        with_u = sum(
            1 for r in rows if isinstance(r, dict) and str(r.get("keyframe_image_url") or "").strip()
        )
        return n, with_u

    pv = str(result.get("playback_video_url") or "")
    av = str(result.get("analysis_video_url") or "")
    vu = str(result.get("video_url") or "")
    lo = str(result.get("low_trust_preview_only") or "")
    fs = str(result.get("final_status") or "")
    trust = str(result.get("analysis_trust") or result.get("trust_level") or "")

    o_n, o_u = _count_rows_with_url("official_phase_keyframes")
    p_n, p_u = _count_rows_with_url("preview_keyframes")
    k_n, k_u = _count_rows_with_url("keyframes")

    lines = [
        f"[PRO_PROV3][UI_PLAYBOOK] analysis_id={analysis_id} context={context}",
        "  若设置(Cloudflare)分析页无视频/关键帧图，优先核对本单 JSON 是否满足前端契约:",
        "  (1) 视频: playback_video_url / video_url / analysis_video_url 须为浏览器可请求的绝对 URL(生产多为 https+R2)；",
        "  (2) 关键帧: 展示行需含 keyframe_image_url(时间线 JPG)；高信任看 official_phase_keyframes，低信任看 preview_keyframes；",
        "  (3) 路由 /pro/[id] 会先用 session 再 IndexedDB；仅含本机路径或非 http 时易空白。",
        f"  本单: final_status={fs!r} low_trust_preview_only={lo!r} trust={trust!r}",
        f"  行数 official={o_n}(url={o_u}) preview={p_n}(url={p_u}) keyframes={k_n}(url={k_u})",
        f"  video_url[{_url_kind(vu)} len={len(vu)}]",
        f"  playback[{_url_kind(pv)} len={len(pv)}]",
        f"  analysis_video[{_url_kind(av)} len={len(av)}]",
        "  排查: STELLAR_PROV3_R2_PUBLIC_BASE / R2 上传、Edge 回写、前端 NEXT_PUBLIC_MODAL_BACKEND_URL；"
        "若 kind=path_only/non_http 则前端无法直接播放。",
        "  提示: original/playback 常为 iPhone .mov，Chrome 等可能无法解码；前端应优先用 analysis_video_url(时间线 .mp4) 作页内播放。",
    ]
    logger.info("\n".join(lines))


async def _pro_media_file_handler(analysis_id: str, filename: str) -> FileResponse | RedirectResponse:
    media_dir = _safe_analysis_media_dir(analysis_id)
    safe_name = Path(filename).name
    target = (media_dir / safe_name).resolve()
    if media_dir not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if target.exists() and target.is_file():
        return FileResponse(str(target))
    if prov3_r2_media_fully_configured():
        try:
            key = prov3_r2_object_key(analysis_id, safe_name)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename") from None
        if await asyncio.to_thread(r2_head_object_exists, key):
            try:
                pub = prov3_r2_public_url_for_key(key)
            except Exception:
                pub = ""
            if pub:
                return RedirectResponse(url=pub, status_code=302)
    raise HTTPException(status_code=404, detail="Media not found")


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
            body_bytes = await file.read()
            return await _run_pro_analyze_body(
                str(request.base_url).rstrip("/"),
                body_bytes,
                file.filename or "video.mp4",
                rough_impact_time_s,
                screen_mode,
                media_prefix,
                cn_network_hint=_prov3_cn_network_hint_from_request(request),
            )
        finally:
            _MODAL_PRO_ANALYZE_LOCK.release()
            logger.info("[PRO_PROV3][LOCK] rid=%s released", rid)

    body_bytes = await file.read()
    return await _run_pro_analyze_body(
        str(request.base_url).rstrip("/"),
        body_bytes,
        file.filename or "video.mp4",
        rough_impact_time_s,
        screen_mode,
        media_prefix,
        cn_network_hint=_prov3_cn_network_hint_from_request(request),
    )


async def _run_pro_analyze_body(
    api_base_url: str,
    body: bytes,
    original_filename: str,
    rough_impact_time_s: Optional[float],
    screen_mode: bool,
    media_prefix: str,
    *,
    cn_network_hint: bool = False,
) -> dict:
    suffix = Path(original_filename or "video.mp4").suffix or ".mp4"
    mp = media_prefix.rstrip("/")
    base = api_base_url.rstrip("/")

    prov3_begin_analyze()
    try:
        if _prov3_durable_media_required() and not prov3_r2_media_fully_configured():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pro v3 requires durable media (Cloudflare R2) on this runtime: set R2_ENDPOINT, "
                    "R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET, and STELLAR_PROV3_R2_PUBLIC_BASE. "
                    "For local/dev without R2, set STELLAR_PROV3_REQUIRE_R2=0 (non-durable media URLs)."
                ),
            )
        with tempfile.TemporaryDirectory(prefix="stellar_pro_analyze_") as tmpdir:
            input_path = str(Path(tmpdir) / f"input{suffix}")
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir, exist_ok=True)
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
                # enrich_pro_prov3_response may pop _prov3_motion (e.g. low_trust Gemini skip or pass+Gemini).
                # Snapshot so analysis_timeline copy + media gate always see the true-240 file path.
                prov3_motion_snapshot = dict(result.get("_prov3_motion") or {})
                with gemini_modal_cn_proxy_first_context(cn_network_hint):
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
                original_video_url = f"{base}{mp}/media/{analysis_id}/{original_name}"

                playback_src = Path(str(result.get("playback_video_url") or result.get("video_url") or ""))
                playback_video_url = ""
                if playback_src.exists() and playback_src.is_file():
                    playback_name = f"playback{playback_src.suffix or '.mp4'}"
                    playback_dst = media_dir / playback_name
                    shutil.copy2(playback_src, playback_dst)
                    playback_video_url = f"{base}{mp}/media/{analysis_id}/{playback_name}"

                analysis_src = Path(
                    str(
                        prov3_motion_snapshot.get("analysis_video")
                        or (result.get("_prov3_motion") or {}).get("analysis_video")
                        or ""
                    ).strip()
                )
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

                _inject_keyframe_urls(list(result.get("keyframes") or []), keyframe_url_by_file)
                _inject_keyframe_urls(list(result.get("official_phase_keyframes") or []), keyframe_url_by_file)
                _inject_keyframe_urls(list(result.get("preview_keyframes") or []), keyframe_url_by_file)

                result["original_video_url"] = original_video_url
                result["video_url"] = original_video_url
                result["playback_video_url"] = playback_video_url or original_video_url
                if analysis_video_url:
                    result["analysis_video_url"] = analysis_video_url

                _prov3_validate_product_media_or_raise(result, media_dir)

                screen_cropped_video_url = ""
                screen_src = Path(str(result.get("screen_cropped_video_url") or ""))
                if screen_src.exists() and screen_src.is_file():
                    screen_name = f"screen_cropped{screen_src.suffix or '.mp4'}"
                    screen_dst = media_dir / screen_name
                    shutil.copy2(screen_src, screen_dst)
                    screen_cropped_video_url = f"{base}{mp}/media/{analysis_id}/{screen_name}"

                contact_sheet_filename: str | None = None
                contact_src = Path(str(result.get("contact_sheet_url") or ""))
                if contact_src.exists() and contact_src.is_file():
                    contact_name = f"contact_sheet{contact_src.suffix or '.jpg'}"
                    contact_dst = media_dir / contact_name
                    shutil.copy2(contact_src, contact_dst)
                    contact_sheet_filename = contact_name
                    result["contact_sheet_url"] = f"{base}{mp}/media/{analysis_id}/{contact_name}"

                if screen_cropped_video_url:
                    result["screen_cropped_video_url"] = screen_cropped_video_url

                if prov3_r2_media_fully_configured():
                    try:
                        r2_required: set[str] = set(keyframe_url_by_file.keys())
                        r2_required.add(original_name)
                        if playback_video_url:
                            r2_required.add(playback_dst.name)
                        if analysis_video_url:
                            r2_required.add(analysis_dst.name)
                        if screen_cropped_video_url:
                            r2_required.add(screen_dst.name)
                        if contact_sheet_filename:
                            r2_required.add(contact_sheet_filename)

                        def _upload_verify() -> dict[str, str]:
                            return upload_prov3_media_directory_to_r2_and_verify(
                                media_dir,
                                analysis_id,
                                r2_required,
                            )

                        r2_by_fn = await asyncio.to_thread(_upload_verify)
                        _rewrite_prov3_result_urls_to_r2(result, r2_by_fn)
                    except Exception as exc:
                        logger.exception("[PRO_PROV3][R2] durable media upload failed: %s", exc)
                        raise HTTPException(
                            status_code=503,
                            detail="Pro media storage unavailable; check R2 configuration.",
                        ) from exc

                result["screen_mode"] = bool(screen_mode)
                result["pro_http_path"] = mp
                result.pop("_prov3_motion", None)
                result.pop("prov3", None)

                logger.info("[PRO_PROV3][MEDIA] original_video_url=%s", result["original_video_url"])
                logger.info("[PRO_PROV3][MEDIA] playback_video_url=%s", result["playback_video_url"])
                logger.info("[PRO_PROV3][MEDIA] video_url=%s", result["video_url"])
                logger.info(
                    "[PRO_PROV3][MEDIA] analysis_video_url=%s",
                    str(result.get("analysis_video_url") or ""),
                )
                _log_prov3_frontend_media_playbook(
                    result,
                    analysis_id=analysis_id,
                    context="pro_v3_analyze_body_ok",
                )
                return result
            except FFmpegNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except RuntimeError as exc:
                if str(exc) == PROV3_ANALYZE_CANCELLED:
                    raise HTTPException(status_code=422, detail="分析已取消") from exc
                logger.warning(
                    "[PRO_PROV3][API] analyze RuntimeError (422) file=%s: %s",
                    original_filename,
                    exc,
                )
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception(
                    "[PRO_PROV3][API] analyze failed (500) file=%s suffix=%s: %s",
                    original_filename,
                    suffix,
                    exc,
                )
                raise HTTPException(status_code=500, detail=f"pro_analyze failed: {exc}") from exc
    finally:
        prov3_finish_analyze()


@router_pro_v3.post("/analyze/start")
async def pro_v3_analyze_start(
    request: Request,
    body: Prov3AnalyzeStartBody,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Queue a full Pro v3 analyze from R2 (``videos/{user_id}/…``); returns ``job_id`` immediately.

    Poll job status via the same-origin app route that reads R2 (``/api/prov3/analyze/job/…``).
    One in-flight analyze per Modal worker when single-flight is enabled (same lock as sync ``/analyze``).

    The heavy work is scheduled with ``BackgroundTasks`` (not ``asyncio.create_task``): on Modal/ASGI the
    request scope can tear down before a detached task runs, leaving R2 stuck at ``accepted`` and the UI
    polling forever.
    """
    if not current_user or not current_user.get("is_pro"):
        raise HTTPException(status_code=403, detail="Pro membership required")
    user_id = str(current_user.get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    _pro_analyze_ingress_echo("PRO_PROV3_ASYNC_START", request)
    rid = (request.headers.get("x-request-id") or "").strip() or secrets.token_hex(4)
    cn_hint = _prov3_cn_network_hint_from_request(request)
    logger.info(
        "[PRO_PROV3][ASYNC][API] rid=%s path=%s screen_mode=%s cn_network_hint=%s",
        rid,
        request.url.path,
        "true" if body.screen_mode else "false",
        int(cn_hint),
    )

    if not prov3_r2_media_fully_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Async Pro analyze requires R2 for job status. Configure R2 credentials, "
                "or use synchronous POST /pro-v3/analyze for local testing."
            ),
        )

    key = _validate_user_video_r2_key(body.source_r2_key, user_id)
    if not await asyncio.to_thread(r2_head_object_exists, key):
        raise HTTPException(
            status_code=404,
            detail="Uploaded video not found in storage; upload again via the app and retry.",
        )

    lock_acquired = False
    if prov3_analyze_single_flight_active():
        try:
            await asyncio.wait_for(_MODAL_PRO_ANALYZE_LOCK.acquire(), timeout=0)
            lock_acquired = True
        except asyncio.TimeoutError:
            logger.warning("[PRO_PROV3][ASYNC][LOCK] rid=%s reject 409", rid)
            raise HTTPException(
                status_code=409,
                detail=(
                    "当前已有视频在分析中，请稍后再试。"
                    "若您只提交了一次，可能是上一次分析仍在后台处理，请等待几分钟后在历史记录中查看。"
                ),
            ) from None
        logger.info("[PRO_PROV3][ASYNC][LOCK] rid=%s acquired", rid)

    job_id = secrets.token_urlsafe(16)
    api_base = str(request.base_url).rstrip("/")
    await _prov3_write_async_job_status(
        job_id,
        {"status": "accepted", "job_id": job_id, "user_id": user_id},
    )
    background_tasks.add_task(
        _prov3_analyze_async_worker,
        job_id,
        key,
        body.screen_mode,
        body.rough_impact_time_s,
        api_base,
        "/pro-v3",
        lock_acquired,
        user_id,
        cn_network_hint=cn_hint,
    )
    logger.info("[PRO_PROV3][ASYNC][API] rid=%s job_id=%s background_tasks scheduled", rid, job_id)
    return {"job_id": job_id, "status": "accepted"}


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
