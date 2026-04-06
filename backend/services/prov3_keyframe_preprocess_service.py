from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable, Dict

from lib.prov3.keyframes.types import PreprocessMeta, PreprocessResult
from services.internal.frame_enhance_service import generate_analysis_frames
from services.internal.video_240fps_service import run_prov3_cleanup_and_true240

logger = logging.getLogger(__name__)


def run_preprocess(
    input_video: str,
    work_dir: str,
    *,
    screen_mode: bool = False,
    cancel_check: Callable[[], None] | None = None,
) -> PreprocessResult:
    """Pro v3 统一预处理（**屏幕模式与普通上传同一套顺序**）。

    客户端无论「拍屏录」还是「直接上传挥杆文件」，都是 **multipart 上传** 到
    ``POST /pro-v3/analyze``（``screen_mode=true/false``）或分步 ``POST /pro-v3/keyframes/*``。

    后端顺序（**单次编码：清洗 + 恒定 240fps 分析轨**，避免先 ``analysis_cleaned`` 再 ``analysis_240fps`` 的重复整段转码）：

    1. **run_prov3_cleanup_and_true240** — 同一 ffmpeg 通路：缩放、轻去噪、H.264（``screen_mode`` 时 ``setsar=1``），在需要时接 ``minterpolate``（MCI）至 **240fps**；输入已严格满足 true240 条件时跳过插帧但仍走同一清洗编码。可选复用工作目录内已有效的 ``analysis_240fps.mp4``。
    2. **generate_analysis_frames** — 在 240 分析视频上均匀抽帧 → ``analysis_frames`` / 增强局部帧。

    之后 A/B（SwingNet）只吃 **第 1 步产出的 240 分析视频** 与第 2 步的帧列表，不再区分「是否拍屏」分支。
    """
    analysis_id = f"prov3_{uuid.uuid4().hex[:12]}"
    local_dir = str(Path(work_dir) / analysis_id)

    logger.info(
        "[prov3][preprocess] analysis_id=%s screen_mode=%s pipeline=cleanup_true240_merged->frames",
        analysis_id,
        screen_mode,
    )
    if cancel_check:
        cancel_check()
    cleanup, timeline = run_prov3_cleanup_and_true240(
        input_video,
        local_dir,
        screen_mode=screen_mode,
    )
    afps = int(timeline.get("analysis_fps", 240))
    if afps != 240:
        raise RuntimeError(f"true240_required: analysis_fps_mismatch={afps}")
    if cancel_check:
        cancel_check()
    frames = generate_analysis_frames(
        str(timeline["analysis_video"]),
        local_dir,
        analysis_fps=afps,
    )

    meta = PreprocessMeta(
        source_fps=float(cleanup.get("source_fps", 30.0)),
        analysis_fps=int(timeline.get("analysis_fps", 240)),
        stabilized=bool(cleanup.get("stabilized", True)),
        denoised=bool(cleanup.get("denoised", True)),
        cropped_single_swing=bool(cleanup.get("cropped_single_swing", True)),
        screen_mode_corrected=bool(cleanup.get("screen_mode_corrected", False)),
    )

    return PreprocessResult(
        analysis_id=analysis_id,
        analysis_video=str(timeline["analysis_video"]),
        preprocess_meta=meta,
        analysis_frames=frames["analysis_frames"],
        enhanced_local_frames=frames["enhanced_local_frames"],
    )
