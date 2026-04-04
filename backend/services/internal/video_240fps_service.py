from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict


def build_analysis_timeline(video_path: str, work_dir: str) -> Dict[str, object]:
    """Normalize into internal 240fps analysis timeline."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(work_dir) / "analysis_240fps.mp4")
    shutil.copy2(video_path, out_path)
    return {
        "analysis_video": out_path,
        "analysis_fps": 240,
    }
