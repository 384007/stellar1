"""Pro v2 — AI review of 6 core keyframes (screen mode trust gate)."""

from __future__ import annotations

import logging
from typing import Any

from services.gemini_service import analyze_pro_v2_core_keyframe_review

logger = logging.getLogger(__name__)

# Picker phases → API core_frame_scores keys
_PICKER_TO_CORE: dict[str, str] = {
    "takeaway": "takeaway",
    "backswing": "backswing_mid",
    "top": "top",
    "downswing": "early_downswing",
    "impact": "impact",
    "follow_through": "release",
}

CORE_PHASE_ORDER: tuple[str, ...] = (
    "takeaway",
    "backswing_mid",
    "top",
    "early_downswing",
    "impact",
    "release",
)


def _empty_scores() -> dict[str, dict[str, Any]]:
    return {
        k: {"score": 0, "pass_90": False, "confidence": 0.0}
        for k in CORE_PHASE_ORDER
    }


def _parse_score_entry(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"score": 0, "pass_90": False, "confidence": 0.0}
    try:
        sc = int(round(float(raw.get("score", 0))))
    except (TypeError, ValueError):
        sc = 0
    sc = max(0, min(100, sc))
    # Product gate: pass_90 is strictly score >= 90 (do not trust model-only boolean).
    pass90 = sc >= 90
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return {"score": sc, "pass_90": pass90, "confidence": round(conf, 4)}


def build_ordered_core_images(keyframes: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Returns (b64_list, picker_phases_in_order) for Gemini."""
    by_phase: dict[str, dict[str, Any]] = {}
    for k in keyframes:
        p = str(k.get("phase") or "").strip()
        if p:
            by_phase[p] = k
    images: list[str] = []
    picker_order: list[str] = []
    for picker_phase, core_key in _PICKER_TO_CORE.items():
        row = by_phase.get(picker_phase)
        b64 = str((row or {}).get("image_base64") or "").strip()
        if not b64:
            logger.warning("[PRO_V2][KF_REVIEW] missing image for phase=%s (core=%s)", picker_phase, core_key)
        images.append(b64)
        picker_order.append(picker_phase)
    return images, picker_order


def merge_ai_core_scores(
    ai_raw: dict[str, Any],
    *,
    review_round: int,
) -> tuple[dict[str, dict[str, Any]], bool, list[str]]:
    """Normalize AI JSON → core_frame_scores, all_pass_90, retry_reasons."""
    base = _empty_scores()
    cfs = ai_raw.get("core_frame_scores")
    if isinstance(cfs, dict):
        alias = {
            "backswing": "backswing_mid",
            "downswing": "early_downswing",
            "follow_through": "release",
        }
        for k, v in cfs.items():
            kk = str(k).strip()
            kk = alias.get(kk, kk)
            if kk in base:
                base[kk] = _parse_score_entry(v)
    reasons = ai_raw.get("retry_reasons")
    rlist: list[str] = []
    if isinstance(reasons, list):
        rlist = [str(x).strip().upper() for x in reasons if str(x).strip()]
    all_pass = all(base[k]["pass_90"] for k in CORE_PHASE_ORDER)
    retry_ai = ai_raw.get("retry_required")
    if isinstance(retry_ai, bool):
        retry_required = retry_ai
    else:
        retry_required = not all_pass
    if not all_pass and not rlist:
        for ck in CORE_PHASE_ORDER:
            if not base[ck]["pass_90"]:
                rlist.append(f"{ck.upper()}_BELOW_90")
    logger.info(
        "[PRO_V2][KF_REVIEW] review_round=%s all_core_pass_90=%s retry_required=%s reasons=%s",
        review_round,
        all_pass,
        retry_required,
        rlist[:8],
    )
    return base, all_pass, rlist


async def run_core_keyframe_review_round(
    keyframes: list[dict[str, Any]],
    *,
    review_round: int,
) -> dict[str, Any]:
    """One AI vision pass over 6 core frames."""
    images, picker_order = build_ordered_core_images(keyframes)
    if not any(images):
        logger.error("[PRO_V2][KF_REVIEW] no_images — forcing retry_required")
        sc, _, reasons = merge_ai_core_scores(
            {"core_frame_scores": {}, "retry_required": True, "retry_reasons": ["NO_CORE_IMAGES"]},
            review_round=review_round,
        )
        return {
            "review_round": review_round,
            "core_frame_scores": sc,
            "retry_required": True,
            "retry_reasons": reasons,
            "picker_order": picker_order,
        }

    ai_out = await analyze_pro_v2_core_keyframe_review(
        images,
        review_round=review_round,
        call_label=f"pro_v2_kf_r{review_round}",
    )
    sc, all_pass, reasons = merge_ai_core_scores(ai_out, review_round=review_round)
    return {
        "review_round": review_round,
        "core_frame_scores": sc,
        "retry_required": not all_pass,
        "retry_reasons": reasons,
        "picker_order": picker_order,
        "ai_raw": ai_out,
    }
