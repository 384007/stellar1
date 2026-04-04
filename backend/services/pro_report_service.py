"""Pro Stage 7: AI writes coaching copy only — no images, no frame selection."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from services.gemini_service import PRO_AI_TIMEOUT_S, analyze_stellar_pro_report_only

logger = logging.getLogger(__name__)


def build_keyframe_metrics_payload(keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    """Numeric phase timeline for the report prompt (no base64)."""
    return {
        "motion_engine": "stellar_pro_v3",
        "phases": [
            {
                "phase": k.get("phase"),
                "timestamp_s": k.get("timestamp"),
                "frame_index": k.get("source_frame_index", k.get("frame_index")),
                "pose_idx": k.get("source_pose_idx"),
            }
            for k in (keyframes or [])
        ],
    }


async def run_pro_ai_report(
    poses: list[dict],
    keyframes: list[dict[str, Any]],
    *,
    impact_meta: dict[str, Any] | None = None,
    region: str = "global",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    logger.info("[STELLAR_PRO][REPORT] stage=start poses=%s kf=%s", len(poses), len(keyframes))

    mid = max(0, len(poses) // 2)
    rep = poses[mid]
    pose_data: dict[str, Any] = {
        "angles": rep.get("angles") or {},
        "all_frame_angles": [p.get("angles") or {} for p in poses],
        "frame_count": len(poses),
        "impact_refine_meta": impact_meta or {},
    }
    kfm = build_keyframe_metrics_payload(keyframes)

    ai = await asyncio.wait_for(
        analyze_stellar_pro_report_only(pose_data, kfm, region=region),
        timeout=PRO_AI_TIMEOUT_S + 60.0,
    )
    wall = round(time.perf_counter() - t0, 3)
    out: dict[str, Any] = {
        "summary": (ai.get("summary") or "").strip() or None,
        "summary_zh": (ai.get("summary_zh") or ai.get("summary") or "").strip() or None,
        "total_score": int(ai.get("total_score") or 0),
        "issues": ai.get("issues") or [],
        "issues_zh": ai.get("issues_zh") or [],
        "suggestions": ai.get("suggestions") or [],
        "suggestions_zh": ai.get("suggestions_zh") or [],
    }
    if isinstance(ai.get("scores"), dict):
        out["scores"] = ai["scores"]
    if isinstance(ai.get("advanced_metrics"), dict):
        out["advanced_metrics"] = ai["advanced_metrics"]
    if isinstance(ai.get("training_plan"), dict):
        out["training_plan"] = ai["training_plan"]
    logger.info(
        "[STELLAR_PRO][REPORT] stage=done wall_s=%s total_score=%s",
        wall,
        out.get("total_score"),
    )
    return out
