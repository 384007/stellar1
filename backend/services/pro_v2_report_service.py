"""Pro v2 — AI writes report only (short phase-tagged issues/suggestions; contract in PRO_V2_REPORT_PROMPT)."""

from __future__ import annotations

import logging
from typing import Any

from services.gemini_service import analyze_pro_v2_report_only

logger = logging.getLogger(__name__)


async def write_pro_v2_ai_report(
    motion_context: dict[str, Any],
    *,
    region: str = "global",
) -> dict[str, Any]:
    """Text-only Gemini report from motion_context JSON."""
    out = await analyze_pro_v2_report_only(motion_context, region=region)
    summary_en = str(out.get("summary") or "").strip()
    summary_zh = str(out.get("summary_zh") or "").strip()
    en_words = len(summary_en.split())
    zh_chars = len(summary_zh.replace(" ", ""))
    if en_words < 300 or zh_chars < 380:
        logger.warning(
            "[PRO_V2][REPORT] short_summary warning en_words=%s zh_chars=%s",
            en_words,
            zh_chars,
        )
    logger.info(
        "[PRO_V2][REPORT] total_score=%s provider=%s",
        out.get("total_score"),
        out.get("ai_provider"),
    )
    return out
