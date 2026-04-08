"""Lite product pipeline entry — delegates to ``lite_orchestrator_service`` (preprocess → A → B)."""

from __future__ import annotations

from typing import Any

from services.lite_orchestrator_service import run_lite_orchestrator


async def run_lite_independent_pipeline(video_path: str, *, region: str = "global") -> dict[str, Any]:
    return await run_lite_orchestrator(video_path, region=region)
