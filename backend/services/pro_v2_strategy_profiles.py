"""Pro v2 — map AI routing + retry_reasons → real backend execution parameters."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

CORE_TO_MISSING_REASON = {
    "takeaway": "TAKEAWAY_IMAGE_MISSING",
    "backswing_mid": "BACKSWING_MID_IMAGE_MISSING",
    "top": "TOP_IMAGE_MISSING",
    "early_downswing": "EARLY_DOWNSWING_IMAGE_MISSING",
    "impact": "IMPACT_IMAGE_MISSING",
    "release": "RELEASE_IMAGE_MISSING",
}


class RoutingDerivedParams:
    """Concrete knobs derived from `run_pro_v2_ai_routing` JSON (not echo-only)."""

    __slots__ = (
        "quality_level",
        "use_deblur",
        "use_heavy_club_tracking",
        "pose_priority",
        "expected_confidence_ceiling",
        "screen_apply_unsharp",
        "ffmpeg_analysis_vf_prefix",
        "dense_club_emphasis",
        "min_dense_frames",
        "late_strip_club_boost",
        "impact_refine_aggressive",
        "screen_relaxed_base_attempt1",
    )

    def __init__(
        self,
        *,
        quality_level: str,
        use_deblur: bool,
        use_heavy_club_tracking: bool,
        pose_priority: bool,
        expected_confidence_ceiling: float,
        screen_apply_unsharp: bool,
        ffmpeg_analysis_vf_prefix: str,
        dense_club_emphasis: float,
        min_dense_frames: int,
        late_strip_club_boost: int,
        impact_refine_aggressive: bool,
        screen_relaxed_base_attempt1: float,
    ) -> None:
        self.quality_level = quality_level
        self.use_deblur = use_deblur
        self.use_heavy_club_tracking = use_heavy_club_tracking
        self.pose_priority = pose_priority
        self.expected_confidence_ceiling = expected_confidence_ceiling
        self.screen_apply_unsharp = screen_apply_unsharp
        self.ffmpeg_analysis_vf_prefix = ffmpeg_analysis_vf_prefix
        self.dense_club_emphasis = dense_club_emphasis
        self.min_dense_frames = min_dense_frames
        self.late_strip_club_boost = late_strip_club_boost
        self.impact_refine_aggressive = impact_refine_aggressive
        self.screen_relaxed_base_attempt1 = screen_relaxed_base_attempt1


def _boolish(v: object, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no", ""):
        return False
    return default


def derive_routing_execution(route: dict | None) -> RoutingDerivedParams:
    """Turn routing_strategy into parameters that downstream services must honor."""
    r = route or {}
    ql = str(r.get("quality_level") or "medium").strip().lower()
    if ql not in ("high", "medium", "low"):
        ql = "medium"
    use_deblur = _boolish(r.get("use_deblur"), False)
    heavy = _boolish(r.get("use_heavy_club_tracking"), False)
    pose_pri = _boolish(r.get("pose_priority"), False)
    try:
        ceiling = float(r.get("expected_confidence_ceiling", 0.82))
    except (TypeError, ValueError):
        ceiling = 0.82
    ceiling = max(0.15, min(0.98, ceiling))

    screen_apply_unsharp = use_deblur
    ffmpeg_prefix = ""
    if use_deblur:
        ffmpeg_prefix = "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.48"

    club_em = 0.42 if heavy else 0.0
    late_boost = 2 if heavy else 0
    min_dense = 14 if ql == "low" else 16
    impact_agg = heavy or ql == "low"
    relaxed_base = 0.03 if ql == "low" else 0.0

    return RoutingDerivedParams(
        quality_level=ql,
        use_deblur=use_deblur,
        use_heavy_club_tracking=heavy,
        pose_priority=pose_pri,
        expected_confidence_ceiling=round(ceiling, 4),
        screen_apply_unsharp=screen_apply_unsharp,
        ffmpeg_analysis_vf_prefix=ffmpeg_prefix,
        dense_club_emphasis=club_em,
        min_dense_frames=min_dense,
        late_strip_club_boost=late_boost,
        impact_refine_aggressive=impact_agg,
        screen_relaxed_base_attempt1=relaxed_base,
    )


def log_route_apply(rp: RoutingDerivedParams, *, analysis_id: str = "") -> None:
    logger.info(
        "[PRO_V2][ROUTE_APPLY] id=%s quality=%s use_deblur=%s heavy_club=%s pose_priority=%s "
        "ceiling=%s screen_unsharp=%s ffmpeg_vf_prefix=%s dense_club_em=%s min_dense=%s "
        "late_boost=%s impact_agg_default=%s relaxed_base_a1=%s",
        analysis_id or "n/a",
        rp.quality_level,
        rp.use_deblur,
        rp.use_heavy_club_tracking,
        rp.pose_priority,
        rp.expected_confidence_ceiling,
        rp.screen_apply_unsharp,
        repr(rp.ffmpeg_analysis_vf_prefix)[:120],
        rp.dense_club_emphasis,
        rp.min_dense_frames,
        rp.late_strip_club_boost,
        rp.impact_refine_aggressive,
        rp.screen_relaxed_base_attempt1,
    )


def base_picker_tuning_dict(rp: RoutingDerivedParams) -> dict[str, Any]:
    """Routing-driven base: heavy club / low quality widen late strip and soften strike lock."""
    strike = 76 if rp.use_heavy_club_tracking else 80
    late_extra = int(rp.late_strip_club_boost)
    follow_gap_d = 0
    if rp.quality_level == "low":
        strike = max(70, strike - 4)
        late_extra += 2
        follow_gap_d = -1
    return {
        "top_dense_delta": 0,
        "top_use_second_valley": False,
        "impact_variant": 0,
        "strike_percentile": strike,
        "late_min_gap_extra": late_extra,
        "follow_min_gap_delta": follow_gap_d,
        "release_dense_shift": 0,
        "backswing_median_mode": rp.quality_level == "low",
        "downswing_dense_delta": -1 if rp.quality_level == "low" else 0,
        "takeaway_late_shift": 1 if rp.quality_level == "low" else 0,
        "legacy_picker_variant": 0,
    }


def merge_retry_reasons_into_tuning(
    base: dict[str, Any],
    reasons: list[str],
    *,
    routing: RoutingDerivedParams,
) -> dict[str, Any]:
    """Map round-1 failure tokens → second-pass picker behavior."""
    t = dict(base)
    rs = {str(x).strip().upper() for x in reasons if str(x).strip()}

    if "TOP_BELOW_90" in rs or "TOP_IMAGE_MISSING" in rs:
        t["top_dense_delta"] = int(t.get("top_dense_delta", 0)) - 2
        t["top_use_second_valley"] = True
    if "IMPACT_BELOW_90" in rs or "IMPACT_IMAGE_MISSING" in rs:
        t["impact_variant"] = min(3, int(t.get("impact_variant", 0)) + 2)
        t["strike_percentile"] = max(72, int(t.get("strike_percentile", 80)) - 3)
    if "RELEASE_BELOW_90" in rs or "RELEASE_IMAGE_MISSING" in rs:
        t["follow_min_gap_delta"] = int(t.get("follow_min_gap_delta", 0)) - 2
        t["release_dense_shift"] = int(t.get("release_dense_shift", 0)) + 3
        t["late_min_gap_extra"] = max(0, int(t.get("late_min_gap_extra", 0)) - 1)
    if "BACKSWING_MID_BELOW_90" in rs or "BACKSWING_MID_IMAGE_MISSING" in rs:
        t["backswing_median_mode"] = True
    if "EARLY_DOWNSWING_BELOW_90" in rs or "EARLY_DOWNSWING_IMAGE_MISSING" in rs:
        t["downswing_dense_delta"] = int(t.get("downswing_dense_delta", 0)) - 3
    if "TAKEAWAY_BELOW_90" in rs or "TAKEAWAY_IMAGE_MISSING" in rs:
        t["takeaway_late_shift"] = int(t.get("takeaway_late_shift", 0)) + 3
    if "NO_CORE_IMAGES" in rs or "REVIEW_AI_FAILED" in rs:
        t["impact_variant"] = max(int(t.get("impact_variant", 0)), 1)
        t["strike_percentile"] = max(70, int(t.get("strike_percentile", 80)) - 5)

    # Late-strip nudge scales with failure families — not a blind variant=1 rerun.
    pv = 0
    if rs:
        pv = 1
        late_strip_hits = sum(
            1
            for k in (
                "IMPACT_BELOW_90",
                "RELEASE_BELOW_90",
                "IMPACT_IMAGE_MISSING",
                "RELEASE_IMAGE_MISSING",
            )
            if k in rs
        )
        if late_strip_hits:
            pv = 2
        if "NO_CORE_IMAGES" in rs or "REVIEW_AI_FAILED" in rs:
            pv = max(pv, 2)
        if len(rs) >= 4:
            pv = max(pv, 2)
    t["legacy_picker_variant"] = min(2, pv)

    logger.info(
        "[PRO_V2][RETRY_APPLY] reasons=%s -> top_d=%s top_2nd=%s imp_var=%s strike_pct=%s "
        "late_extra=%s follow_gap_d=%s rel_shift=%s bs_med=%s ds_d=%s tw_late=%s legacy_pv=%s routing_q=%s",
        sorted(rs)[:12],
        t.get("top_dense_delta"),
        t.get("top_use_second_valley"),
        t.get("impact_variant"),
        t.get("strike_percentile"),
        t.get("late_min_gap_extra"),
        t.get("follow_min_gap_delta"),
        t.get("release_dense_shift"),
        t.get("backswing_median_mode"),
        t.get("downswing_dense_delta"),
        t.get("takeaway_late_shift"),
        t.get("legacy_picker_variant"),
        routing.quality_level,
    )
    return t


def resolve_screen_unsharp(
    rp: RoutingDerivedParams,
    attempt: int,
    retry_reasons: list[str],
) -> tuple[bool, str, str]:
    """Whether to run crop unsharp, log reason, and strength profile (normal|strong).

    ``routing.use_deblur`` always wins on any attempt. Retry pass can add unsharp when
    routing did not ask for it but round-1 failed (decode / missing / AI).
    """
    if rp.screen_apply_unsharp:
        prof = "strong" if attempt >= 1 and rp.quality_level == "low" else "normal"
        return True, "routing_use_deblur", prof
    if attempt >= 1:
        rs = {str(x).strip().upper() for x in retry_reasons if str(x).strip()}
        if any("IMAGE_MISSING" in x for x in rs):
            return True, "retry_image_missing", "strong"
        if "REVIEW_AI_FAILED" in rs or "NO_CORE_IMAGES" in rs:
            return True, "retry_decode_or_review", "strong"
        if "EARLY_DOWNSWING_BELOW_90" in rs or "TOP_BELOW_90" in rs:
            return True, "retry_motion_pick", "normal"
        if rs:
            return True, "retry_generic_reasons", "normal"
    return False, "off", "normal"


def compute_screen_relaxed_margin(
    attempt: int,
    routing: RoutingDerivedParams,
    retry_reasons: list[str],
) -> float:
    if attempt == 0:
        return 0.0
    rs = {str(x).strip().upper() for x in retry_reasons if str(x).strip()}
    extra = 0.04
    if routing.quality_level == "low":
        extra += 0.02
    if "NO_CORE_IMAGES" in rs or "REVIEW_AI_FAILED" in rs:
        extra += 0.03
    if any(k.endswith("_BELOW_90") or k.endswith("_IMAGE_MISSING") for k in rs):
        extra += 0.01
    return min(0.11, routing.screen_relaxed_base_attempt1 + extra)


def second_pass_impact_aggressive(
    routing: RoutingDerivedParams,
    retry_reasons: list[str],
    first_pass_default: bool,
) -> bool:
    rs = {str(x).strip().upper() for x in retry_reasons if str(x).strip()}
    if "IMPACT_BELOW_90" in rs or "REVIEW_AI_FAILED" in rs or "NO_CORE_IMAGES" in rs:
        return True
    return bool(first_pass_default or routing.use_heavy_club_tracking)


def ffmpeg_vf_for_attempt(
    rp: RoutingDerivedParams,
    attempt: int,
    retry_reasons: list[str],
) -> str | None:
    base = (rp.ffmpeg_analysis_vf_prefix or "").strip().strip(",")
    if attempt == 0:
        return base or None
    rs = {str(x).strip().upper() for x in retry_reasons if str(x).strip()}
    boost = "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.65"
    if "REVIEW_AI_FAILED" in rs or "NO_CORE_IMAGES" in rs:
        boost = "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.82"
    if not base:
        return boost
    if boost.split("=")[0] in base:
        return base
    return f"{boost},{base}"
