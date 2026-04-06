"""FFmpeg / ffprobe helpers for Pro v3 — no system package required on PATH.

Resolution order for ``ffmpeg``:
1. ``STELLAR_FFMPEG_BINARY`` if executable file exists
2. ``shutil.which("ffmpeg")``
3. ``imageio_ffmpeg.get_ffmpeg_exe()`` (wheel-bundled static build, includes ``minterpolate`` on supported platforms)

``ffprobe`` is optional: same env / PATH / sibling of ffmpeg; if still missing, metadata uses OpenCV
(``opencv-python-headless`` already in requirements). Phones only upload video; processing stays server-side.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ffmpeg -i prints "Duration: HH:MM:SS.xx" for many inputs where ffprobe JSON omits stream/format duration (low-bitrate / odd encodes).
_DURATION_LINE_RE = re.compile(
    r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def _parse_hms_duration_token(val: str) -> float:
    s = (val or "").strip().replace(",", ".")
    if not s or s.upper() == "N/A":
        return 0.0
    parts = s.split(":")
    if len(parts) == 3:
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return 0.0
    try:
        x = float(s)
        return x if x > 0 else 0.0
    except ValueError:
        return 0.0


def _duration_from_tags(tags: Any) -> float:
    if not isinstance(tags, dict):
        return 0.0
    best = 0.0
    for key, raw in tags.items():
        lk = str(key).lower()
        if not isinstance(raw, str):
            continue
        if "duration" not in lk and lk not in ("length", "media_length", "playback_length"):
            continue
        best = max(best, _parse_hms_duration_token(raw))
    return best


def _duration_from_ffmpeg_stderr(text: str) -> float:
    m = _DURATION_LINE_RE.search(text or "")
    if not m:
        return 0.0
    try:
        hh, mm, ss = m.group(1), m.group(2), m.group(3)
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except ValueError:
        return 0.0


def _ffprobe_supplement_duration_from_tags(path: str, pb: str) -> float:
    cmd = [
        pb,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags",
        "-show_entries",
        "format_tags",
        "-of",
        "json",
        path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
    except subprocess.TimeoutExpired:
        logger.warning("[prov3][ffprobe] tag probe timeout path=%s", path)
        return 0.0
    if out.returncode != 0:
        return 0.0
    try:
        data = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return 0.0
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    d = 0.0
    if streams and isinstance(streams[0], dict):
        d = max(d, _duration_from_tags(streams[0].get("tags")))
    if isinstance(fmt, dict):
        d = max(d, _duration_from_tags(fmt.get("tags")))
    if d > 0:
        logger.info("[prov3][ffprobe] duration from container/stream tags: %.3fs", d)
    return d


def _duration_from_ffmpeg_input_banner(path: str) -> float:
    try:
        exe = ffmpeg_bin()
    except FFmpegNotFoundError:
        return 0.0
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-nostdin", "-i", path],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("[prov3][ffprobe] ffmpeg -i banner probe timeout path=%s", path)
        return 0.0
    blob = (proc.stderr or "") + (proc.stdout or "")
    d = _duration_from_ffmpeg_stderr(blob)
    if d > 0:
        logger.info("[prov3][ffprobe] duration from ffmpeg demuxer banner: %.3fs", d)
    return d


def _coalesce_duration_seconds(
    path: str,
    pb: str | None,
    dur_s: float,
    stream0: dict[str, Any] | None,
    fmt: dict[str, Any] | None,
) -> float:
    """Fill missing/zero duration for low-quality or thin-metadata uploads (enables minterpolate gates)."""
    if dur_s > 0:
        return dur_s
    if stream0:
        d_tags = _duration_from_tags(stream0.get("tags"))
        if d_tags > 0:
            logger.info("[prov3][ffprobe] duration from primary probe stream tags: %.3fs", d_tags)
            return d_tags
    if fmt:
        d_tags = _duration_from_tags(fmt.get("tags"))
        if d_tags > 0:
            logger.info("[prov3][ffprobe] duration from primary probe format tags: %.3fs", d_tags)
            return d_tags
    if pb:
        d2 = _ffprobe_supplement_duration_from_tags(path, pb)
        if d2 > 0:
            return d2
    d3 = _duration_from_ffmpeg_input_banner(path)
    if d3 > 0:
        return d3
    try:
        ocv = ffprobe_video_meta_opencv(path)
        oc_d = float(ocv.get("duration_s") or 0.0)
    except Exception as exc:
        logger.warning("[prov3][ffprobe] OpenCV duration fallback failed: %s", exc)
        return 0.0
    if oc_d > 0:
        logger.info("[prov3][ffprobe] duration from OpenCV frame_count/fps: %.3fs", oc_d)
        return oc_d
    return 0.0


class FFmpegNotFoundError(RuntimeError):
    pass


_ffmpeg_exe: str | None = None
_ffprobe_resolved: bool = False
_ffprobe_exe: str | None = None


def _env_executable(name: str) -> str | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_file() and os.access(str(p), os.X_OK):
        return str(p.resolve())
    logger.warning("[%s] not executable or missing: %s", name, raw)
    return None


def _imageio_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "imageio-ffmpeg not installed — Pro v3 needs it when ffmpeg is not on PATH "
            "(pip install imageio-ffmpeg)"
        )
        return None
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.warning("imageio_ffmpeg.get_ffmpeg_exe failed: %s", exc)
        return None
    if exe and Path(exe).is_file():
        return str(Path(exe).resolve())
    return None


def ffmpeg_bin() -> str:
    """Return a working ``ffmpeg`` path (system, env, or imageio-ffmpeg bundle)."""
    global _ffmpeg_exe
    if _ffmpeg_exe is not None:
        return _ffmpeg_exe

    for candidate in (
        _env_executable("STELLAR_FFMPEG_BINARY"),
        shutil.which("ffmpeg"),
        _imageio_ffmpeg_exe(),
    ):
        if candidate:
            _ffmpeg_exe = candidate
            logger.info("[prov3][ffmpeg] using %s", _ffmpeg_exe)
            return _ffmpeg_exe

    raise FFmpegNotFoundError(
        "ffmpeg not found — set STELLAR_FFMPEG_BINARY, install ffmpeg on PATH, "
        "or `pip install imageio-ffmpeg` for a bundled binary."
    )


def _sibling_ffprobe(ffmpeg_path: str) -> str | None:
    parent = Path(ffmpeg_path).resolve().parent
    for name in ("ffprobe", "ffprobe.exe"):
        p = parent / name
        if p.is_file() and os.access(str(p), os.X_OK):
            return str(p)
    return None


def _resolve_ffprobe() -> str | None:
    """Return ffprobe path, or None to use OpenCV metadata fallback."""
    global _ffprobe_resolved, _ffprobe_exe
    if _ffprobe_resolved:
        return _ffprobe_exe
    _ffprobe_resolved = True

    for candidate in (
        _env_executable("STELLAR_FFPROBE_BINARY"),
        shutil.which("ffprobe"),
    ):
        if candidate:
            _ffprobe_exe = candidate
            logger.info("[prov3][ffprobe] using %s", _ffprobe_exe)
            return _ffprobe_exe

    try:
        sib = _sibling_ffprobe(ffmpeg_bin())
    except FFmpegNotFoundError:
        sib = None
    if sib:
        _ffprobe_exe = sib
        logger.info("[prov3][ffprobe] using sibling %s", _ffprobe_exe)
        return _ffprobe_exe

    _ffprobe_exe = None
    logger.info("[prov3][ffprobe] not found — using OpenCV for stream metadata")
    return None


def ffprobe_video_meta_opencv(path: str) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("opencv: cannot open video for metadata")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    dur_s = 0.0
    if fps > 1e-6 and nb_frames > 0:
        dur_s = nb_frames / fps
    if fps <= 0.0:
        fps = 30.0
    if dur_s <= 0:
        dur_s = _duration_from_ffmpeg_input_banner(path)

    return {
        "width": w,
        "height": h,
        "fps": float(fps),
        "duration_s": float(dur_s),
        "nb_frames": nb_frames,
    }


def ffprobe_video_meta(path: str) -> dict[str, Any]:
    """Return first video stream metadata: fps, duration_s, width, height, nb_frames."""
    pb = _resolve_ffprobe()
    if not pb:
        return ffprobe_video_meta_opencv(path)

    cmd = [
        pb,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
        "-show_entries",
        "stream_tags",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "format_tags",
        "-of",
        "json",
        path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if out.returncode != 0:
        logger.warning(
            "ffprobe failed (%s), falling back to OpenCV: %s",
            out.returncode,
            (out.stderr or out.stdout or "")[:500],
        )
        return ffprobe_video_meta_opencv(path)

    data = json.loads(out.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        return ffprobe_video_meta_opencv(path)

    s0 = streams[0]
    w = int(s0.get("width") or 0)
    h = int(s0.get("height") or 0)

    def _parse_rate(r: str | None) -> float:
        if not r or r in ("0/0", "N/A"):
            return 0.0
        try:
            a, b = r.split("/")
            return float(a) / max(float(b), 1e-9)
        except (ValueError, ZeroDivisionError):
            return 0.0

    fps = _parse_rate(s0.get("r_frame_rate")) or _parse_rate(s0.get("avg_frame_rate"))

    def _safe_stream_duration(raw: Any) -> float:
        if raw is None or raw == "":
            return 0.0
        s = str(raw).strip()
        if s.upper() in ("N/A", "NAN"):
            return 0.0
        try:
            x = float(s)
            return x if x > 0 else 0.0
        except (ValueError, TypeError):
            return 0.0

    dur_s = max(_safe_stream_duration(s0.get("duration")), _safe_stream_duration(fmt.get("duration")))
    dur_s = _coalesce_duration_seconds(path, pb, dur_s, s0, fmt)
    nb = s0.get("nb_frames")
    try:
        nb_frames = int(nb) if nb and str(nb).isdigit() else 0
    except (TypeError, ValueError):
        nb_frames = 0

    if fps <= 0.0 and dur_s > 0 and nb_frames > 0:
        fps = nb_frames / dur_s
    if fps <= 0.0:
        fps = 30.0

    return {
        "width": w,
        "height": h,
        "fps": float(fps),
        "duration_s": float(dur_s),
        "nb_frames": nb_frames,
    }


_ffmpeg_active: set[subprocess.Popen[Any]] = set()
_ffmpeg_active_lock = threading.Lock()
_sigterm_hook_installed = False
_previous_sigterm: Any = None


def _kill_tracked_ffmpeg() -> None:
    with _ffmpeg_active_lock:
        procs = list(_ffmpeg_active)
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass


def _sigterm_forward(signum: int, frame: Any) -> None:
    _kill_tracked_ffmpeg()
    prev = _previous_sigterm
    if callable(prev):
        prev(signum, frame)
        return
    if prev == signal.SIG_IGN:
        return
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def ensure_sigterm_kills_ffmpeg() -> None:
    """Install SIGTERM hook once on the **main thread** (required by Python).

    Pro analyze runs ffmpeg inside ``asyncio.to_thread``; those workers cannot call
    ``signal.signal``. Call ``ensure_sigterm_kills_ffmpeg()`` from FastAPI ``startup``
    so container SIGTERM still terminates blocking ffmpeg children.
    """
    global _sigterm_hook_installed, _previous_sigterm
    if _sigterm_hook_installed or os.name == "nt":
        return
    if threading.current_thread() is not threading.main_thread():
        return
    _sigterm_hook_installed = True
    _previous_sigterm = signal.signal(signal.SIGTERM, _sigterm_forward)


def run_ffmpeg(
    args: list[str],
    *,
    timeout_s: int = 900,
    label: str = "ffmpeg",
    loglevel: str = "error",
    stats_period_s: int | None = None,
) -> None:
    ensure_sigterm_kills_ffmpeg()
    pre: list[str] = [ffmpeg_bin(), "-hide_banner", "-loglevel", loglevel]
    if stats_period_s is not None and int(stats_period_s) > 0:
        pre.extend(["-stats_period", str(int(stats_period_s))])
    pre.extend(["-y"])
    cmd = pre + args
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with _ffmpeg_active_lock:
        _ffmpeg_active.add(proc)
    stdout = stderr = ""
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=60)
        except Exception:
            stdout, stderr = "", ""
        err = (stderr or stdout or "").strip()
        raise RuntimeError(f"{label} timed out after {timeout_s}s: {err[:2000]}")
    finally:
        with _ffmpeg_active_lock:
            _ffmpeg_active.discard(proc)
    if proc.returncode != 0:
        err = (stderr or stdout or "").strip()
        raise RuntimeError(f"{label} failed (exit {proc.returncode}): {err[:2000]}")


def ffmpeg_has_filter(name: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return name in (out.stdout or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
