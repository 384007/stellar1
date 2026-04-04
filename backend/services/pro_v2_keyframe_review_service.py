"""Pro v2 — AI review of 6 core keyframes (screen mode trust gate)."""

from __future__ import annotations

import logging
from typing import Any

from services.gemini_service import analyze_pro_v2_core_keyframe_review_for_phases
from services.pro_v2_strategy_profiles import CORE_TO_MISSING_REASON

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

_MIN_B64_LEN = 48


def _empty_scores() -> dict[str, dict[str, Any]]:
    return {
        k: {
            "score": 0,
            "pass_90": False,
            "confidence": 0.0,
            "reason_codes": [],
            "comment_zh": "",
            "comment_en": "",
        }
        for k in CORE_PHASE_ORDER
    }


def _parse_score_entry(raw: Any, *, confidence_ceiling: float | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"score": 0, "pass_90": False, "confidence": 0.0, "reason_codes": [], "comment_zh": "", "comment_en": ""}
    try:
        sc = int(round(float(raw.get("score", 0))))
    except (TypeError, ValueError):
        sc = 0
    sc = max(0, min(100, sc))
    pass90 = sc >= 90
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    if confidence_ceiling is not None:
        conf = min(conf, float(confidence_ceiling))
    reason_codes = raw.get("reason_codes")
    reasons = [str(x).strip().upper() for x in reason_codes] if isinstance(reason_codes, list) else []
    return {
        "score": sc,
        "pass_90": pass90,
        "confidence": round(conf, 4),
        "reason_codes": reasons[:8],
        "comment_zh": str(raw.get("comment_zh") or "").strip()[:200],
        "comment_en": str(raw.get("comment_en") or "").strip()[:200],
    }


def merge_ai_core_scores(
    ai_raw: dict[str, Any],
    *,
    review_round: int,
    confidence_ceiling: float | None = None,
    allowed_keys: set[str] | None = None,
    base_prefill: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], bool, list[str]]:
    """Merge AI JSON into core_frame_scores."""
    base: dict[str, dict[str, Any]]
    if base_prefill:
        base = {k: dict(v) for k, v in base_prefill.items()}
    else:
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
            if kk not in base:
                continue
            if allowed_keys is not None and kk not in allowed_keys:
                continue
            base[kk] = _parse_score_entry(v, confidence_ceiling=confidence_ceiling)

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
        "[PRO_V2][KF_REVIEW] review_round=%s all_core_pass_90=%s retry_required=%s reasons=%s ceiling=%s",
        review_round,
        all_pass,
        retry_required,
        rlist[:10],
        confidence_ceiling,
    )
    return base, all_pass, rlist


def _usable_b64(b64: str) -> bool:
    s = (b64 or "").strip()
    return len(s) >= _MIN_B64_LEN


async def run_core_keyframe_review_round(
    keyframes: list[dict[str, Any]],
    *,
    review_round: int,
    confidence_ceiling: float | None = None,
) -> dict[str, Any]:
    """AI vision on non-empty core frames only; missing phases are structural failures (no empty images to model)."""
    by_phase: dict[str, dict[str, Any]] = {}
    for k in keyframes:
        p = str(k.get("phase") or "").strip()
        if p:
            by_phase[p] = k

    base = _empty_scores()
    missing_reasons: list[str] = []
    valid_pairs: list[tuple[str, str]] = []

    for picker_phase, core_key in _PICKER_TO_CORE.items():
        row = by_phase.get(picker_phase)
        b64 = str((row or {}).get("image_base64") or "").strip()
        if not _usable_b64(b64):
            mr = CORE_TO_MISSING_REASON.get(core_key, f"{core_key.upper()}_IMAGE_MISSING")
            base[core_key] = {
                "score": 0,
                "pass_90": False,
                "confidence": 0.0,
                "reason_codes": [mr],
                "comment_zh": "关键帧缺失或解码失败",
                "comment_en": "Keyframe missing or decode failed",
            }
            missing_reasons.append(mr)
        else:
            valid_pairs.append((core_key, b64))

    sent = [p[0] for p in valid_pairs]
    logger.info(
        "[PRO_V2][KF_REVIEW_INPUT] review_round=%s missing_reasons=%s sent_phases=%s n_images=%s",
        review_round,
        missing_reasons[:8],
        sent,
        len(valid_pairs),
    )

    if not valid_pairs:
        logger.error("[PRO_V2][KF_REVIEW] no_usable_images — NO_CORE_IMAGES")
        reasons = list(dict.fromkeys(missing_reasons + ["NO_CORE_IMAGES"]))
        return {
            "review_round": review_round,
            "core_frame_scores": base,
            "retry_required": True,
            "retry_reasons": reasons,
            "picker_order": list(_PICKER_TO_CORE.keys()),
        }

    ai_out = await analyze_pro_v2_core_keyframe_review_for_phases(
        valid_pairs,
        review_round=review_round,
        call_label=f"pro_v2_kf_r{review_round}",
    )
    allowed = {p[0] for p in valid_pairs}
    sc, all_pass, ai_reasons = merge_ai_core_scores(
        ai_out,
        review_round=review_round,
        confidence_ceiling=confidence_ceiling,
        allowed_keys=allowed,
        base_prefill=base,
    )

    merged_reasons = list(dict.fromkeys(missing_reasons + ai_reasons))
    if str(ai_out.get("ai_provider") or "") == "kf_review_fallback":
        merged_reasons = list(dict.fromkeys(merged_reasons + ["REVIEW_AI_FAILED"]))

    retry_required = (not all_pass) or bool(missing_reasons) or ("REVIEW_AI_FAILED" in merged_reasons)

    logger.info(
        "[PRO_V2][KF_REVIEW] review_round=%s merged_retry=%s reasons=%s",
        review_round,
        retry_required,
        merged_reasons[:12],
    )

    return {
        "review_round": review_round,
        "core_frame_scores": sc,
        "retry_required": retry_required,
        "retry_reasons": merged_reasons,
        "picker_order": list(_PICKER_TO_CORE.keys()),
        "ai_raw": ai_out,
    }
