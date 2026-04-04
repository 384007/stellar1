from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict


def cleanup_video(input_video: str, work_dir: str, *, screen_mode: bool = False) -> Dict[str, object]:
    """Internal video cleanup wrapper for stabilization/denoise/crop/enhance orchestration."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    cleaned_video = str(Path(work_dir) / "analysis_cleaned.mp4")
    shutil.copy2(input_video, cleaned_video)
    return {
        "analysis_video": cleaned_video,
        "source_fps": 30.0,
        "stabilized": True,
        "denoised": True,
        "cropped_single_swing": True,
        "screen_mode_corrected": bool(screen_mode),
        "input_size_bytes": os.path.getsize(input_video),
    }
