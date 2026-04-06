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

import logging
import mimetypes
import os
import re
from pathlib import Path
logger = logging.getLogger(__name__)

_R2_KEY_PREFIX = "prov3-media"


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
