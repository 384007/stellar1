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

# Beyond this duration (seconds), default to fps=dup instead of minterpolate — MCI is O(frames²)-ish and can stall Modal for tens of minutes.
_MINTERPOLATE_MAX_DURATION_S = float(os.getenv("STELLAR_PROV3_MINTERPOLATE_MAX_DURATION_S") or "45")


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def build_analysis_timeline(video_path: str, work_dir: str) -> Dict[str, object]:
    """Pro v3 **第二步**：在 ``cleanup_video`` 产物上生成 **恒定 240fps** 分析用 MP4（SwingNet / 抽帧用）。

    Uses motion-compensated interpolation when ``minterpolate`` is available; otherwise ``fps=240``
    (frame duplication / sampling — still a valid 240 Hz time base).

    **Note:** MCI minterpolate is CPU-heavy and produces **no ffmpeg logs** until the pass finishes
    (``run_ffmpeg`` uses ``-loglevel error``). Long clips can sit on this line for many minutes — not a deadlock.

    * ``STELLAR_PROV3_USE_FAST_240FPS=1`` — always use ``fps=240`` (no minterpolate).
    * ``STELLAR_PROV3_MINTERPOLATE_MAX_DURATION_S`` — above this input duration, skip minterpolate (default 45).
    * ``STELLAR_RUNTIME=modal`` — skip minterpolate unless ``STELLAR_PROV3_ALLOW_MINTERPOLATE_ON_MODAL=1``.
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

    unknown_dur = dur <= 0
    if unknown_dur:
        logger.warning(
            "[prov3] 240fps: input duration unknown — using fps=%s fast path (avoid long minterpolate stall)",
            TARGET_FPS,
        )
    # Modal: MCI is CPU-heavy; forbid unless explicitly allowed (Modal Pro worker uses cpu>=2 + 3600s timeout).
    _on_modal = (os.getenv("STELLAR_RUNTIME") or "").strip().lower() == "modal"
    _allow_mci_on_modal = _env_truthy("STELLAR_PROV3_ALLOW_MINTERPOLATE_ON_MODAL")
    _modal_skip_mci = _on_modal and not _allow_mci_on_modal
    if _modal_skip_mci and ffmpeg_has_filter("minterpolate"):
        logger.warning(
            "[prov3] 240fps: STELLAR_RUNTIME=modal — skipping minterpolate "
            "(set STELLAR_PROV3_ALLOW_MINTERPOLATE_ON_MODAL=1 to enable MCI; prefer cpu>=2 and a long function timeout).",
        )
    use_mci = (
        ffmpeg_has_filter("minterpolate")
        and not _env_truthy("STELLAR_PROV3_USE_FAST_240FPS")
        and not _modal_skip_mci
        and not unknown_dur
        and dur <= _MINTERPOLATE_MAX_DURATION_S
    )
    if (
        not unknown_dur
        and dur > _MINTERPOLATE_MAX_DURATION_S
        and ffmpeg_has_filter("minterpolate")
        and not _env_truthy("STELLAR_PROV3_USE_FAST_240FPS")
    ):
        logger.warning(
            "[prov3] 240fps: duration %.1fs > %.1fs — using fps=%s dup (skip minterpolate). "
            "Raise STELLAR_PROV3_MINTERPOLATE_MAX_DURATION_S if you accept long CPU time.",
            dur,
            _MINTERPOLATE_MAX_DURATION_S,
            TARGET_FPS,
        )

    if use_mci:
        vf = f"minterpolate=fps={TARGET_FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        logger.info(
            "[prov3] 240fps pipeline: minterpolate (motion-compensated) — CPU-heavy, no log until ffmpeg exits (timeout 1200s)",
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
            timeout_s=1200,
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
