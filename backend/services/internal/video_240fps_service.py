from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

from services.internal.prov3_ffmpeg import (
    FFmpegNotFoundError,
    ffmpeg_has_filter,
    ffprobe_stream_codec_meta,
    ffprobe_video_meta,
    run_ffmpeg,
)
from services.internal.video_cleanup_service import prov3_cleanup_max_dims, prov3_cleanup_vf

logger = logging.getLogger(__name__)

TARGET_FPS = 240
_MINTERPOLATE_TIMEOUT_S = 3600


def _probe_input_meta(video_path: str) -> Dict[str, object]:
    try:
        m = ffprobe_video_meta(video_path)
        return dict(m) if isinstance(m, dict) else {}
    except Exception as exc:
        logger.warning("[prov3] 240fps: ffprobe input failed (%s), proceeding anyway", exc)
        return {}


def _cleanup_metadata_dict(
    input_video: str,
    screen_mode: bool,
    meta_in: Dict[str, object],
) -> Dict[str, object]:
    return {
        "source_fps": float(meta_in.get("fps") or 30.0),
        "stabilized": True,
        "denoised": True,
        "cropped_single_swing": True,
        "screen_mode_corrected": bool(screen_mode),
        "input_size_bytes": os.path.getsize(input_video),
    }


def _cache_valid_true240(out_path: str) -> bool:
    try:
        if not os.path.isfile(out_path) or os.path.getsize(out_path) <= 0:
            return False
        om = ffprobe_video_meta(out_path)
        out_fps = float(om.get("fps") or 0.0)
        return out_fps > 0 and abs(out_fps - TARGET_FPS) < 0.6
    except Exception:
        return False


def _strict_true240_input_fast_path(video_path: str, meta_in: Dict[str, object]) -> bool:
    """Strict gate: only skip minterpolate when input already matches true240 + h264 + yuv420p + target box."""
    codec = ffprobe_stream_codec_meta(video_path)
    if not codec or not (codec.get("codec_name") or "").strip():
        return False

    fps = float(meta_in.get("fps") or 0.0)
    if abs(fps - TARGET_FPS) > 1.8:
        return False

    r_fps = float(codec.get("r_fps") or 0.0)
    avg_fps = float(codec.get("avg_fps") or 0.0)
    if r_fps > 1.0 and avg_fps > 1.0:
        base = max(r_fps, avg_fps)
        if abs(r_fps - avg_fps) > 0.02 * base:
            return False

    cname = str(codec.get("codec_name") or "").lower()
    if cname not in ("h264",):
        return False

    pix = str(codec.get("pix_fmt") or "").lower()
    if pix != "yuv420p":
        return False

    w = int(meta_in.get("width") or 0)
    h = int(meta_in.get("height") or 0)
    if w <= 0 or h <= 0 or (w % 2) or (h % 2):
        return False

    max_w, max_h = prov3_cleanup_max_dims()
    if w > max_w or h > max_h:
        return False

    dur = float(meta_in.get("duration_s") or 0.0)
    nb = int(meta_in.get("nb_frames") or 0)
    if dur > 0.5 and nb > 10:
        expected = dur * TARGET_FPS
        if abs(float(nb) - expected) / max(expected, 1.0) > 0.025:
            return False

    return True


def run_prov3_cleanup_and_true240(
    input_video: str,
    work_dir: str,
    *,
    screen_mode: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Single-pass cleanup (scale/denoise/[sar]) + optional minterpolate → ``analysis_240fps.mp4``.

    Replaces separate ``cleanup_video`` + ``build_analysis_timeline`` for the product preprocess path:
    one full libx264 encode instead of two. Returns the same logical metadata as those two steps
    (cleanup side-car dict + timeline dict) without writing ``analysis_cleaned.mp4``.
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(work_dir) / "analysis_240fps.mp4")

    meta_in = _probe_input_meta(input_video)
    dur = float(meta_in.get("duration_s") or 0.0)
    w = int(meta_in.get("width") or 0)
    h = int(meta_in.get("height") or 0)
    logger.info(
        "[prov3] true240 input meta: duration_s=%.2f size=%dx%d (merged cleanup+240)",
        dur,
        w,
        h,
    )

    if _cache_valid_true240(out_path):
        logger.info("[prov3] true240_reuse_cached=1 path=%s", out_path)
        return _cleanup_metadata_dict(input_video, screen_mode, meta_in), {
            "analysis_video": out_path,
            "analysis_fps": TARGET_FPS,
        }

    skip_mci = _strict_true240_input_fast_path(input_video, meta_in)
    if skip_mci:
        logger.info("[prov3] true240_fast_path=1 skip_minterpolate=1 (strict input gate passed)")
        vf = prov3_cleanup_vf(screen_mode=screen_mode)
    else:
        if not ffmpeg_has_filter("minterpolate"):
            raise RuntimeError(
                "Pro v3 true240_required but ffmpeg filter 'minterpolate' is unavailable. "
                "True240 cannot proceed and no fallback is allowed."
            )
        mci = f"minterpolate=fps={TARGET_FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        vf = f"{prov3_cleanup_vf(screen_mode=screen_mode)},{mci}"

    if dur <= 0:
        logger.warning(
            "[prov3] 240fps: input duration still unknown after probes — minterpolate may be slow",
        )

    logger.info(
        "[prov3] true240_started=1 merged_encode=1 minterpolate=%s timeout_s=%s",
        "0" if skip_mci else "1",
        _MINTERPOLATE_TIMEOUT_S,
    )

    try:
        run_ffmpeg(
            [
                "-i",
                input_video,
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                out_path,
            ],
            label="prov3_cleanup_true240",
            timeout_s=_MINTERPOLATE_TIMEOUT_S,
            loglevel="info",
            stats_period_s=2,
            stream_progress_logs=True,
            progress_stats_period_s=2,
        )
    except FFmpegNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"prov3_cleanup_true240: ffmpeg failed: {exc}") from exc

    om = ffprobe_video_meta(out_path)
    out_fps = float(om.get("fps") or 0.0)
    if out_fps <= 0:
        raise RuntimeError("prov3_cleanup_true240: output fps probe failed")
    if abs(out_fps - TARGET_FPS) > 0.5:
        raise RuntimeError(f"prov3_cleanup_true240: output fps mismatch {out_fps:.3f} != {TARGET_FPS}")
    logger.info("[prov3] true240_completed=1 output_fps=%.3f merged_encode=1", out_fps)

    return _cleanup_metadata_dict(input_video, screen_mode, meta_in), {
        "analysis_video": out_path,
        "analysis_fps": TARGET_FPS,
    }


def build_analysis_timeline(video_path: str, work_dir: str) -> Dict[str, object]:
    """Build Pro v3 single-authority analysis timeline: true-240 via minterpolate only.

    Used when a **cleaned** intermediate already exists (tests, debugging). The product path uses
    ``run_prov3_cleanup_and_true240`` to avoid a duplicate full transcode.
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(work_dir) / "analysis_240fps.mp4")
    logger.info("[prov3] true240_required=1 (standalone build_analysis_timeline)")

    meta = {}
    try:
        meta = ffprobe_video_meta(video_path)
    except Exception as exc:
        logger.warning("[prov3] 240fps: ffprobe input failed (%s), proceeding anyway", exc)
    dur = float(meta.get("duration_s") or 0.0)
    w = int(meta.get("width") or 0)
    h = int(meta.get("height") or 0)
    logger.info(
        "[prov3] 240fps input meta: duration_s=%.2f size=%dx%d",
        dur,
        w,
        h,
    )

    if dur <= 0:
        logger.warning(
            "[prov3] 240fps: input duration still unknown after probes — using minterpolate anyway (may be slow)",
        )

    if _cache_valid_true240(out_path):
        logger.info("[prov3] true240_reuse_cached=1 path=%s", out_path)
        return {"analysis_video": out_path, "analysis_fps": TARGET_FPS}

    _has_mci = ffmpeg_has_filter("minterpolate")
    if not _has_mci:
        raise RuntimeError(
            "Pro v3 true240_required but ffmpeg filter 'minterpolate' is unavailable. "
            "True240 cannot proceed and no fallback is allowed."
        )

    vf = f"minterpolate=fps={TARGET_FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    logger.info("[prov3] true240_started=1 timeout_s=%s standalone=1", _MINTERPOLATE_TIMEOUT_S)

    try:
        run_ffmpeg(
            [
                "-i",
                video_path,
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                out_path,
            ],
            label="prov3_240fps",
            timeout_s=_MINTERPOLATE_TIMEOUT_S,
            loglevel="info",
            stats_period_s=2,
            stream_progress_logs=True,
            progress_stats_period_s=2,
        )
    except FFmpegNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"prov3_240fps: ffmpeg failed: {exc}") from exc

    om = ffprobe_video_meta(out_path)
    out_fps = float(om.get("fps") or 0.0)
    if out_fps <= 0:
        raise RuntimeError("prov3_240fps: output fps probe failed")
    if abs(out_fps - TARGET_FPS) > 0.5:
        raise RuntimeError(f"prov3_240fps: output fps mismatch {out_fps:.3f} != {TARGET_FPS}")
    logger.info("[prov3] true240_completed=1 output_fps=%.3f", out_fps)

    return {
        "analysis_video": out_path,
        "analysis_fps": TARGET_FPS,
    }
