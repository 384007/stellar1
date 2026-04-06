"""Durable Pro v3 product media via Cloudflare R2 (S3-compatible API).

When fully configured, analyze uploads originals / timeline / keyframe JPGs to R2 and
returns public ``https://…/prov3-media/{analysis_id}/{filename}`` URLs instead of
ephemeral ``GET /pro-v3/media/…`` on the worker disk.

Env (reuse bucket creds from ``.env.example``):

- ``R2_ENDPOINT``, ``R2_ACCESS_KEY``, ``R2_SECRET_KEY``, ``R2_BUCKET`` — required for upload
- ``STELLAR_PROV3_R2_PUBLIC_BASE`` — public origin for objects (no trailing slash), e.g.
  ``https://pub-xxxx.r2.dev`` or your R2 custom domain that maps to the same bucket root
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
from pathlib import Path

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_R2_KEY_PREFIX = "prov3-media"
PROV3_ASYNC_JOB_PREFIX = "prov3-async-jobs"


def prov3_r2_media_fully_configured() -> bool:
    b = (os.getenv("STELLAR_PROV3_R2_PUBLIC_BASE") or "").strip().rstrip("/")
    return bool(
        b
        and (os.getenv("R2_ENDPOINT") or "").strip()
        and (os.getenv("R2_ACCESS_KEY") or "").strip()
        and (os.getenv("R2_SECRET_KEY") or "").strip()
        and (os.getenv("R2_BUCKET") or "").strip()
    )


def _safe_segment(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum() or ch in ("-", "_"))


def prov3_r2_object_key(analysis_id: str, filename: str) -> str:
    aid = _safe_segment(analysis_id)
    fn = Path(str(filename).replace("\\", "/")).name
    if not aid or not fn or fn in (".", ".."):
        raise ValueError("invalid analysis_id or filename for R2 key")
    return f"{_R2_KEY_PREFIX}/{aid}/{fn}"


def prov3_r2_public_url_for_key(key: str) -> str:
    base = (os.getenv("STELLAR_PROV3_R2_PUBLIC_BASE") or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("STELLAR_PROV3_R2_PUBLIC_BASE is not set")
    return f"{base}/{key.lstrip('/')}"


def _s3_client():
    import boto3  # lazy: optional when R2 unused

    return boto3.client(
        "s3",
        endpoint_url=(os.getenv("R2_ENDPOINT") or "").strip(),
        aws_access_key_id=(os.getenv("R2_ACCESS_KEY") or "").strip(),
        aws_secret_access_key=(os.getenv("R2_SECRET_KEY") or "").strip(),
        region_name="auto",
    )


def _guess_content_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(path.name)
    if mt:
        return mt
    low = path.suffix.lower()
    if low in (".jpg", ".jpeg"):
        return "image/jpeg"
    if low == ".png":
        return "image/png"
    if low in (".mp4", ".m4v"):
        return "video/mp4"
    if low == ".webm":
        return "video/webm"
    return "application/octet-stream"


def upload_prov3_media_directory_to_r2_and_verify(
    media_dir: Path,
    analysis_id: str,
    required_filenames: set[str],
) -> dict[str, str]:
    """Upload every file in ``media_dir``, verify S3 head for each required name, return ``{fn: public_url}``."""
    out = upload_prov3_media_directory_to_r2(media_dir, analysis_id)
    missing = required_filenames - set(out.keys())
    if missing:
        raise RuntimeError(f"prov3_media_gate:r2_upload_incomplete:{sorted(missing)}")
    for fn in out:
        key = prov3_r2_object_key(analysis_id, fn)
        if not r2_head_object_exists(key):
            raise RuntimeError(f"prov3_media_gate:r2_head_missing_after_put:{fn}")
    return out


def upload_prov3_media_directory_to_r2(media_dir: Path, analysis_id: str) -> dict[str, str]:
    """Upload every file under ``media_dir``; return ``{filename: public_url}``."""
    if not prov3_r2_media_fully_configured():
        raise RuntimeError("R2 prov3 media is not fully configured")
    bucket = (os.getenv("R2_BUCKET") or "").strip()
    client = _s3_client()
    out: dict[str, str] = {}
    for p in sorted(media_dir.iterdir(), key=lambda x: x.name):
        if not p.is_file():
            continue
        key = prov3_r2_object_key(analysis_id, p.name)
        extra: dict = {"ContentType": _guess_content_type(p)}
        if re.search(r"\.(jpe?g|png|gif|webp)$", p.name, re.I):
            extra["CacheControl"] = "public, max-age=31536000, immutable"
        elif re.search(r"\.(mp4|webm|mov)$", p.name, re.I):
            extra["CacheControl"] = "public, max-age=86400"
        client.upload_file(str(p), bucket, key, ExtraArgs=extra)
        out[p.name] = prov3_r2_public_url_for_key(key)
        logger.info("[PRO_PROV3][R2] put s3://%s/%s (%s bytes)", bucket, key, p.stat().st_size)
    return out


def r2_head_object_exists(key: str) -> bool:
    if not prov3_r2_media_fully_configured():
        return False
    bucket = (os.getenv("R2_BUCKET") or "").strip()
    client = _s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def prov3_async_job_status_key(job_id: str) -> str:
    jid = _safe_segment(job_id)
    if not jid:
        raise ValueError("invalid job_id")
    return f"{PROV3_ASYNC_JOB_PREFIX}/{jid}/status.json"


def prov3_async_job_result_key(job_id: str) -> str:
    jid = _safe_segment(job_id)
    if not jid:
        raise ValueError("invalid job_id")
    return f"{PROV3_ASYNC_JOB_PREFIX}/{jid}/result.json"


def r2_put_json_object(key: str, data: dict) -> None:
    if not prov3_r2_media_fully_configured():
        raise RuntimeError("R2 not configured")
    bucket = (os.getenv("R2_BUCKET") or "").strip()
    client = _s3_client()
    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
        CacheControl="no-store",
    )


def r2_get_json_object_if_exists(key: str) -> dict | None:
    if not prov3_r2_media_fully_configured():
        return None
    bucket = (os.getenv("R2_BUCKET") or "").strip()
    client = _s3_client()
    try:
        r = client.get_object(Bucket=bucket, Key=key)
        raw = r["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except ClientError as exc:
        code = (exc.response.get("Error") or {}).get("Code") or ""
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    except json.JSONDecodeError:
        return None


def r2_download_object_to_path(key: str, dest: Path) -> None:
    if not prov3_r2_media_fully_configured():
        raise RuntimeError("R2 not configured")
    bucket = (os.getenv("R2_BUCKET") or "").strip()
    client = _s3_client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(dest))
    except ClientError as exc:
        code = (exc.response.get("Error") or {}).get("Code") or ""
        if code in ("404", "NoSuchKey", "NotFound"):
            raise FileNotFoundError(f"R2 object not found: {key}") from exc
        raise
