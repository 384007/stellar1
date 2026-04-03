from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any  # noqa: TC003

logger = logging.getLogger(__name__)


class FFmpegProcessError(RuntimeError):
    """Raised when ffmpeg/ffprobe returns a non-zero exit code."""


class FFmpegBinaryMissingError(RuntimeError):
    """Raised when ffmpeg or ffprobe is not installed in the backend environment."""


def _which_or_raise(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegBinaryMissingError(f"{name} not found on PATH")
    return path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def verify_ffmpeg() -> dict[str, str]:
    ffmpeg_bin = _which_or_raise("ffmpeg")
    ffprobe_bin = _which_or_raise("ffprobe")
    logger.info("[ROLE=FFMPEG_PREP] ffmpeg=%s", ffmpeg_bin)
    logger.info("[ROLE=FFMPEG_PREP] ffprobe=%s", ffprobe_bin)
    return {"ffmpeg": ffmpeg_bin, "ffprobe": ffprobe_bin}


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    logger.info("[ROLE=FFMPEG_PREP] run=%s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegProcessError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def probe_video(video_path: str | Path) -> dict[str, Any]:
    ffprobe_bin = _which_or_raise("ffprobe")
    video_path = str(Path(video_path))
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        video_path,
    ]
    proc = _run(cmd)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    return streams[0] if streams else {}


def build_full_240fps_playback(
    input_path: str | Path,
    output_path: str | Path,
    *,
    fast: bool = True,
    crf: int = 28,
    preset: str = "ultrafast",
) -> str:
    """Build a full-length 240fps playback file.

    fast=True: uses fps=240 (frame duplication / frame-rate conversion) for quick frontend playback tests.
    fast=False: uses motion-compensated interpolation via minterpolate and is much slower.
    """
    ffmpeg_bin = _which_or_raise("ffmpeg")
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    vf = (
        "fps=240"
        if fast
        else "minterpolate=fps=240:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    )
    mode = "fast_playback" if fast else "motion_interpolated"
    logger.info("[ROLE=FFMPEG_PREP] mode=%s input=%s output=%s", mode, input_path, output_path)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]
    _run(cmd)
    meta = probe_video(output_path)
    logger.info(
        "[ROLE=FFMPEG_PREP] full240_done avg_frame_rate=%s duration=%s output=%s",
        meta.get("avg_frame_rate"),
        meta.get("duration"),
        output_path,
    )
    return output_path


def build_impact_240fps_window(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_s: float,
    duration_s: float = 0.22,
    crf: int = 30,
    preset: str = "ultrafast",
) -> str:
    """Build a short motion-interpolated 240fps clip around impact for precise analysis."""
    ffmpeg_bin = _which_or_raise("ffmpeg")
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[ROLE=FFMPEG_PREP] mode=impact_window_240 input=%s output=%s start_s=%.3f duration_s=%.3f",
        input_path,
        output_path,
        start_s,
        duration_s,
    )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        input_path,
        "-vf",
        "minterpolate=fps=240:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]
    _run(cmd)
    meta = probe_video(output_path)
    logger.info(
        "[ROLE=FFMPEG_PREP] impact240_done avg_frame_rate=%s duration=%s output=%s",
        meta.get("avg_frame_rate"),
        meta.get("duration"),
        output_path,
    )
    return output_path


def suggest_impact_window(impact_time_s: float, *, pre_s: float = 0.10, duration_s: float = 0.22) -> tuple[float, float]:
    start_s = max(0.0, float(impact_time_s) - float(pre_s))
    return start_s, float(duration_s)


def build_frontend_playback_from_analysis(
    analysis_path: str | Path,
    output_path: str | Path,
    *,
    crf: int = 32,
    preset: str = "ultrafast",
) -> str:
    """Browser-oriented re-encode from the 240fps analysis file (smaller / faststart)."""
    ffmpeg_bin = _which_or_raise("ffmpeg")
    analysis_path = str(Path(analysis_path))
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[ROLE=FFMPEG_PREP] mode=frontend_from_analysis input=%s output=%s crf=%s",
        analysis_path,
        output_path,
        crf,
    )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        analysis_path,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]
    _run(cmd)
    return output_path


def trim_video_segment_h264(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    crf: int = 28,
    preset: str = "ultrafast",
) -> str:
    """Cut a time range without fps conversion — use when source is already analysis 240fps."""
    ffmpeg_bin = _which_or_raise("ffmpeg")
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[ROLE=FFMPEG_PREP] mode=trim_segment input=%s output=%s start_s=%.3f duration_s=%.3f",
        input_path,
        output_path,
        start_s,
        duration_s,
    )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]
    _run(cmd)
    return output_path


def ffprobe_fps(meta: dict[str, Any]) -> float:
    ar = meta.get("avg_frame_rate") or meta.get("r_frame_rate") or "30/1"
    if isinstance(ar, (int, float)):
        return float(ar)
    s = str(ar)
    if "/" in s:
        a, _, b = s.partition("/")
        try:
            den = float(b)
            return float(a) / den if den else 30.0
        except (ValueError, ZeroDivisionError):
            return 30.0
    try:
        return float(s)
    except ValueError:
        return 30.0


def estimated_frame_count(meta: dict[str, Any]) -> int:
    dur = safe_duration_s(meta)
    fps = ffprobe_fps(meta)
    nb = meta.get("nb_frames")
    if nb not in (None, "", "N/A"):
        try:
            v = int(float(str(nb)))
            if v > 0:
                return v
        except (ValueError, TypeError):
            pass
    return max(1, int(round(dur * fps)))


def safe_duration_s(meta: dict[str, Any]) -> float:
    d = meta.get("duration")
    try:
        return max(0.01, float(d))
    except (TypeError, ValueError):
        return 0.01
