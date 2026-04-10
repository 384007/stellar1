"""Temp-file suffix + media hints for uploaded or URL-fetched videos (any common container/codec)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

# Preserve real extensions so ffmpeg/OpenCV demux correctly; unknown ext → .mp4 hint.
_VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".mp4",
        ".m4v",
        ".mov",
        ".qt",
        ".webm",
        ".mkv",
        ".avi",
        ".wmv",
        ".flv",
        ".3gp",
        ".3g2",
        ".ts",
        ".mts",
        ".m2ts",
        ".mpg",
        ".mpeg",
        ".vob",
        ".ogv",
        ".f4v",
        ".asf",
        ".divx",
        ".xvid",
        ".rm",
        ".rmvb",
        ".mxf",
        ".nut",
    }
)


def temp_suffix_for_uploaded_video(filename: str | None) -> str:
    """Suffix for ``NamedTemporaryFile`` / ``mkstemp`` from client filename (or URL basename)."""
    suf = Path((filename or "").strip() or "video.mp4").suffix.lower()
    return suf if suf else ".mp4"


def temp_suffix_from_url(video_url: str) -> str:
    try:
        path = unquote(urlparse(video_url).path)
        base = path.rstrip("/").rsplit("/", 1)[-1]
        return temp_suffix_for_uploaded_video(base)
    except Exception:
        return ".mp4"


def looks_like_video_mime(mime_type: str | None) -> bool:
    m = (mime_type or "").strip().lower().split(";")[0].strip()
    return m.startswith("video/")


def is_likely_video_filename(filename: str | None) -> bool:
    suf = Path(filename or "").suffix.lower()
    return suf in _VIDEO_SUFFIXES
