from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def generate_analysis_frames(analysis_video: str, work_dir: str, frame_count: int = 64) -> Dict[str, List[dict]]:
    """Generate analysis frame index timeline and local enhanced frame set."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    step = max(1, 240 // 12)
    analysis_frames = [{"frame_index": i * step, "time_ms": int((i * step / 240) * 1000)} for i in range(frame_count)]
    enhanced_local_frames = [
        {"frame_index": frame["frame_index"], "enhanced": True} for frame in analysis_frames[::4]
    ]
    return {
        "analysis_frames": analysis_frames,
        "enhanced_local_frames": enhanced_local_frames,
    }
