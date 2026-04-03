"""Stellar Pro pipeline — compatibility shim for `stellar_pro_video_analysis_service`.

Canonical orchestrator: `run_stellar_pro_video_analysis` (motion windows → motion keyframe engine
→ OpenCV impact refine → final gate → contact sheet → text-only AI report → minimal JSON).
"""

from __future__ import annotations

from typing import Any

from services.pro_analysis_chain_service import ProAnalysisChainSettings
from services.stellar_pro_video_analysis_service import run_stellar_pro_video_analysis


async def run_stellar_pro_pipeline_async(
    input_video_path: str,
    work_dir: str,
    *,
    rough_impact_time_s: float | None = None,
    region: str = "global",
    chain_settings: ProAnalysisChainSettings | None = None,
    keyframe_width: int = 320,
) -> dict[str, Any]:
    """Delegates to the single Stellar Pro video analysis orchestrator."""
    return await run_stellar_pro_video_analysis(
        input_video_path,
        work_dir,
        rough_impact_time_s=rough_impact_time_s,
        region=region,
        chain_settings=chain_settings,
        keyframe_width=keyframe_width,
    )
