from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable, Dict

from lib.prov3.keyframes.types import PreprocessMeta, PreprocessResult
from services.internal.frame_enhance_service import generate_analysis_frames
from services.internal.video_240fps_service import build_analysis_timeline
from services.internal.video_cleanup_service import cleanup_video

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

    后端顺序（**先得到清洗后的源时间轴视频，再强制 240Hz 分析时间轴**）：

    1. **cleanup_video** — 缩放、轻去噪、H.264；``screen_mode`` 时额外 ``setsar=1``（拍屏常见 SAR 问题）。
    2. **build_analysis_timeline** — 在清洗产物上生成 **恒定 240fps** 分析用 MP4（默认始终 ``minterpolate`` / MCI；缺滤镜且未开快轨则报错；快轨为 ``fps=240`` dup）。
    3. **generate_analysis_frames** — 在 240 分析视频上均匀抽帧 → ``analysis_frames`` / 增强局部帧。

    之后 A/B（SwingNet）只吃 **第 2 步产出的 240 分析视频** 与第 3 步的帧列表，不再区分「是否拍屏」分支。
    """
    analysis_id = f"prov3_{uuid.uuid4().hex[:12]}"
    local_dir = str(Path(work_dir) / analysis_id)

    logger.info(
        "[prov3][preprocess] analysis_id=%s screen_mode=%s pipeline=cleanup->240fps->frames",
        analysis_id,
        screen_mode,
    )
    if cancel_check:
        cancel_check()
    cleanup = cleanup_video(input_video, local_dir, screen_mode=screen_mode)
    if cancel_check:
        cancel_check()
    timeline = build_analysis_timeline(str(cleanup["analysis_video"]), local_dir)
    afps = int(timeline.get("analysis_fps", 240))
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
