from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

from services.internal.prov3_ffmpeg import (
    FFmpegNotFoundError,
    ffmpeg_has_filter,
    ffprobe_video_meta,
    run_ffmpeg,
)

logger = logging.getLogger(__name__)

TARGET_FPS = 240

try:
    _MINTERPOLATE_TIMEOUT_S = max(60, int(os.getenv("STELLAR_PROV3_MINTERPOLATE_TIMEOUT_S") or "3600"))
except ValueError:
    _MINTERPOLATE_TIMEOUT_S = 3600


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def build_analysis_timeline(video_path: str, work_dir: str) -> Dict[str, object]:
    """Pro v3 **第二步**：在 ``cleanup_video`` 产物上生成 **恒定 240fps** 分析用 MP4（SwingNet / 抽帧用）。

    **True 240 (MCI):** whenever ``minterpolate`` exists and fast-path is off, **always** use it — no
    input-duration cap and no “unknown duration → dup” fallback. Very long or pathological inputs may hit
    ``STELLAR_PROV3_MINTERPOLATE_TIMEOUT_S`` (default 3600s) and fail loudly.

    * ``STELLAR_PROV3_USE_FAST_240FPS=1`` — use ``fps=240`` dup/sample only (no minterpolate).
    * ``STELLAR_RUNTIME=modal`` — MCI forbidden unless ``STELLAR_PROV3_ALLOW_MINTERPOLATE_ON_MODAL=1`` (dup only).

    If true MCI is required (fast off, modal allows MCI) but ``minterpolate`` is missing from ffmpeg,
    raises ``RuntimeError`` instead of silently duping.

    Duration is still probed for logging / diagnostics (``ffprobe_video_meta`` fallbacks for thin metadata).
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(work_dir) / "analysis_240fps.mp4")

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

    _on_modal = (os.getenv("STELLAR_RUNTIME") or "").strip().lower() == "modal"
    _allow_mci_on_modal = _env_truthy("STELLAR_PROV3_ALLOW_MINTERPOLATE_ON_MODAL")
    _modal_skip_mci = _on_modal and not _allow_mci_on_modal
    if _modal_skip_mci and ffmpeg_has_filter("minterpolate"):
        logger.warning(
            "[prov3] 240fps: STELLAR_RUNTIME=modal — skipping minterpolate "
            "(set STELLAR_PROV3_ALLOW_MINTERPOLATE_ON_MODAL=1 to enable MCI; prefer cpu>=2 and a long function timeout).",
        )

    _fast = _env_truthy("STELLAR_PROV3_USE_FAST_240FPS")
    _has_mci = ffmpeg_has_filter("minterpolate")
    _want_mci = not _fast and not _modal_skip_mci

    if _want_mci and not _has_mci:
        raise RuntimeError(
            "Pro v3 true 240 requires ffmpeg filter 'minterpolate' (not available on this ffmpeg build). "
            "Use a full ffmpeg build (see Dockerfile / modal image) or set STELLAR_PROV3_USE_FAST_240FPS=1 "
            "to allow fps duplication only."
        )

    use_mci = _want_mci and _has_mci

    if use_mci:
        vf = f"minterpolate=fps={TARGET_FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        logger.info(
            "[prov3] 240fps pipeline: minterpolate (motion-compensated) — timeout %ss",
            _MINTERPOLATE_TIMEOUT_S,
        )
    else:
        vf = f"fps={TARGET_FPS}"
        if not ffmpeg_has_filter("minterpolate"):
            logger.warning(
                "[prov3] 240fps pipeline: fps=%s only (minterpolate missing on this ffmpeg build)",
                TARGET_FPS,
            )
        else:
            logger.info("[prov3] 240fps pipeline: fps=%s (fast dup/sample, no minterpolate)", TARGET_FPS)

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
        )
    except FFmpegNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"prov3_240fps: ffmpeg failed: {exc}") from exc

    try:
        om = ffprobe_video_meta(out_path)
        out_fps = float(om.get("fps") or TARGET_FPS)
    except Exception:
        out_fps = float(TARGET_FPS)

    return {
        "analysis_video": out_path,
        "analysis_fps": int(round(out_fps)) if out_fps > 1 else TARGET_FPS,
    }
