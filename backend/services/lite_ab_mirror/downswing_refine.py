"""Lite A-only: after a single 8-event A infer, optionally fix only ``Mid-downswing.frame_index`` using MediaPipe pose.

Does not call A again, does not build a second keyframe set, and does not re-stage the other seven events.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Sequence, Tuple

from services.lite_ab_mirror.constants import EVENT_SEQUENCE

logger = logging.getLogger(__name__)

_MAX_CAND = 256
# Minimum normalized distance along Top→Impact corridor to count as "mid" (not hugging boundaries)
_REL_MIN = 0.12
_REL_MAX = 0.88
_ANGLE_TOO_LIKE_TOP_DEG = 4.0
_ANGLE_TOO_LIKE_IMP_DEG = 5.0


def _rows_by_event(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in rows:
        ev = str(r.get("event_name") or "")
        if ev:
            out[ev] = r
    return out


def _ordered_rows(rows: List[dict]) -> List[dict]:
    """Order as EVENT_SEQUENCE; deep-copy rows so other events stay unchanged except where we assign."""
    by_e = _rows_by_event(rows)
    ordered: List[dict] = []
    for ev in EVENT_SEQUENCE:
        if ev in by_e:
            ordered.append(copy.deepcopy(by_e[ev]))
        else:
            ordered.append({"event_name": ev, "frame_index": 0, "confidence": 0.25})
    return ordered


def _linspace(lo: int, hi: int, n: int) -> List[int]:
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo
    if n <= 1:
        return [lo]
    span = hi - lo
    return [int(round(lo + span * i / (n - 1))) for i in range(n)]


def _candidates_top_impact(
    lo: int,
    hi: int,
    analysis_frames: List[dict],
    *,
    max_frame_index: int,
) -> List[int]:
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo
    mx = max(0, int(max_frame_index))
    lo = max(0, min(lo, mx))
    hi = max(0, min(hi, mx))
    if lo >= hi:
        return [lo]
    pool = sorted(
        {int(f.get("frame_index", -1)) for f in analysis_frames if int(f.get("frame_index", -1)) >= 0}
    )
    cand = [i for i in pool if lo <= i <= hi]
    if len(cand) > _MAX_CAND:
        step = max(1, len(cand) // _MAX_CAND)
        cand = cand[::step][:_MAX_CAND]
    if not cand:
        span = hi - lo + 1
        cand = list(range(lo, hi + 1)) if span <= _MAX_CAND else _linspace(lo, hi, _MAX_CAND)
    return sorted({int(x) for x in cand if 0 <= int(x) <= mx})


def _angles_at_frame(
    frame_index: int,
    vfps: float,
    poses: List[dict],
) -> Dict[str, float]:
    if not poses or vfps <= 1e-6:
        return {}
    t_target = float(frame_index) / vfps
    best: dict | None = None
    best_d = 1e9
    for p in poses:
        if not isinstance(p, dict):
            continue
        t = float(p.get("timestamp", -1e9))
        d = abs(t - t_target)
        if d < best_d:
            best_d = d
            best = p
    if not best:
        return {}
    ang = best.get("angles")
    if not isinstance(ang, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in ang.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _angle_l1(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 1e6
    return sum(abs(a[k] - b[k]) for k in keys) / len(keys)


def _clamp_mid_downswing_only(by: Dict[str, dict], hi: int) -> None:
    """Only adjust Mid-downswing; keep it strictly between Top and Impact (exclusive) and in range."""
    top = int(by["Top"]["frame_index"])
    mds = int(by["Mid-downswing"]["frame_index"])
    imp = int(by["Impact"]["frame_index"])
    mds = max(mds, top + 1)
    mds = min(mds, imp - 1)
    mds = max(0, min(mds, hi))
    by["Mid-downswing"]["frame_index"] = mds


def _norm_along_corridor(idx: int, top: int, imp: int) -> float:
    if imp <= top + 1:
        return 0.5
    return (float(idx) - float(top)) / float(imp - top)


def _poses_have_angle_data(poses: List[dict]) -> bool:
    """True if any pose sample carries a non-empty ``angles`` dict (MediaPipe body lines)."""
    for p in poses:
        if not isinstance(p, dict):
            continue
        ang = p.get("angles")
        if isinstance(ang, dict) and ang:
            return True
    return False


def is_mid_downswing_suspicious(
    mds: int,
    top: int,
    imp: int,
    *,
    ang_mds: Dict[str, float],
    ang_top: Dict[str, float],
    ang_imp: Dict[str, float],
) -> bool:
    if mds <= top or mds >= imp:
        return True
    corridor = imp - top
    if corridor < 3:
        return False
    rel = (mds - top) / float(corridor)
    if rel < _REL_MIN or rel > _REL_MAX:
        return True
    if _angle_l1(ang_mds, ang_top) < _ANGLE_TOO_LIKE_TOP_DEG:
        return True
    if ang_imp and _angle_l1(ang_mds, ang_imp) < _ANGLE_TOO_LIKE_IMP_DEG:
        return True
    return False


# Within this margin of the best pose score, treat frames as tied and prefer corridor mid (true mid-downswing).
_SCORE_NEAR_BEST = 0.035
_CORRIDOR_TARGET_REL = 0.42


def _score_mid_down_candidate_mediapipe(
    idx: int,
    *,
    top: int,
    imp: int,
    ang_top: Dict[str, float],
    ang_imp: Dict[str, float],
    vfps: float,
    poses: List[dict],
) -> float:
    """Prefer frames whose pose reads as between Top and Impact (not still at Top, not yet Impact), in the corridor."""
    ang_c = _angles_at_frame(idx, vfps, poses)
    rel = _norm_along_corridor(idx, top, imp)
    corridor_bonus = 1.0 - abs(rel - 0.45) * 0.9
    sep_top = _angle_l1(ang_c, ang_top) if ang_top else 10.0
    sep_imp = _angle_l1(ang_c, ang_imp) if ang_imp else 10.0
    # Mid-downswing should differ from both anchors; balance separation from Top vs Impact.
    geom_sep = math.sqrt(max(1e-6, sep_top) * max(1e-6, sep_imp))
    pose_sep_norm = min(1.0, geom_sep / 35.0)
    return 0.78 * pose_sep_norm + 0.22 * corridor_bonus


def _pick_best_mid_down_frame(
    cand: List[int],
    *,
    mds: int,
    top: int,
    imp: int,
    vfps: float,
    poses: List[dict],
    ang_top: Dict[str, float],
    ang_imp: Dict[str, float],
    suspicious: bool,
) -> Tuple[int, float, str]:
    """
    Argmax pose score; among near-ties prefer ``rel`` closest to mid-corridor.
    If ``suspicious``, exclude current ``mds`` when there is another candidate so MediaPipe can move the frame.
    """
    scored = [
        (
            idx,
            _score_mid_down_candidate_mediapipe(
                idx,
                top=top,
                imp=imp,
                ang_top=ang_top,
                ang_imp=ang_imp,
                vfps=vfps,
                poses=poses,
            ),
        )
        for idx in cand
    ]
    if not scored:
        return mds, -1e9, "empty"

    pool = scored
    strategy = "argmax"
    if suspicious and len(cand) > 1:
        alt = [(i, s) for i, s in scored if i != mds]
        if alt:
            pool = alt
            strategy = "exclude_mds_when_suspicious"

    best_s = max(s for _, s in pool)
    tier = [(i, s) for i, s in pool if s >= best_s - _SCORE_NEAR_BEST]

    def _corridor_key(t: Tuple[int, float]) -> Tuple[float, float, int]:
        i, s = t
        d = abs(_norm_along_corridor(i, top, imp) - _CORRIDOR_TARGET_REL)
        return (d, -s, i)

    tier.sort(key=_corridor_key)
    best_i, best_sc = tier[0]
    if len(tier) > 1:
        if strategy == "argmax":
            strategy = "near_tie_corridor"
        elif strategy == "exclude_mds_when_suspicious":
            strategy = "exclude_mds_near_tie_corridor"
    return int(best_i), float(best_sc), strategy


def refine_mid_downswing_with_pose_motion(
    rows: List[dict],
    *,
    analysis_frames: List[dict],
    preprocess_meta: dict,
    poses: List[dict] | None,
    max_frame_index: int | None,
    # Back-compat: ignored — A runs once; refine does not use motion/timeline or a second hint pass.
    timeline: List[dict] | None = None,
    motions: Sequence[float] | None = None,
    impact_hint_frame_index: int | None = None,
) -> Tuple[List[dict], List[str], Dict[str, Any]]:
    """
    Post-process A's single 8-row output: only ``Mid-downswing.frame_index`` may change.

    When MediaPipe pose carries ``angles``, always re-pick the best mid-downswing frame inside
    (Top+1 … Impact-1) by pose score; other seven events are never modified. No second A infer.
    """
    _ = (timeline, motions, impact_hint_frame_index)

    meta = dict(preprocess_meta or {})
    hi = max_frame_index if max_frame_index is not None else int(meta.get("max_frame_index", -1))
    if hi is None or hi < 0:
        hi = max((int(x.get("frame_index", 0)) for x in analysis_frames), default=0)
    vfps = float(meta.get("analysis_fps") or 30.0)

    pose_list = list(poses or [])

    working = _ordered_rows(rows)
    by = _rows_by_event(working)

    fail: List[str] = []
    dbg: Dict[str, Any] = {
        "module": "downswing_refine",
        "touched": [],
        "original_mds": None,
        "final_mds": None,
    }

    if "Mid-downswing" not in by or "Top" not in by or "Impact" not in by:
        return working, fail, dbg

    top = int(by["Top"]["frame_index"])
    imp = int(by["Impact"]["frame_index"])
    mds = int(by["Mid-downswing"]["frame_index"])
    dbg["original_mds"] = mds

    ang_top = _angles_at_frame(top, vfps, pose_list)
    ang_imp = _angles_at_frame(imp, vfps, pose_list)
    ang_mds = _angles_at_frame(mds, vfps, pose_list)

    # With MediaPipe angles: always judge downswing in Top→Impact and pick best frame (only MDS row may change).
    # Without angles: legacy path — only refine when structurally/pose suspicious.
    always_pose_pick = _poses_have_angle_data(pose_list)
    suspicious = is_mid_downswing_suspicious(
        mds, top, imp, ang_mds=ang_mds, ang_top=ang_top, ang_imp=ang_imp
    )

    if not always_pose_pick and not suspicious:
        dbg["action"] = "kept_mid_down_not_suspicious"
        dbg["final_mds"] = mds
        return working, fail, dbg

    lo_c = top + 1
    hi_c = imp - 1
    if lo_c >= hi_c:
        fail.append("mid_downswing_semantic_invalid")
        dbg["action"] = "no_corridor"
        dbg["final_mds"] = mds
        return working, fail, dbg

    cand = _candidates_top_impact(lo_c, hi_c, analysis_frames, max_frame_index=hi)
    cand = [c for c in cand if lo_c <= c <= hi_c]

    if not cand:
        fail.append("mid_downswing_semantic_invalid")
        dbg["action"] = "no_candidates"
        dbg["final_mds"] = mds
        return working, fail, dbg

    best, best_s, pick_strategy = _pick_best_mid_down_frame(
        cand,
        mds=mds,
        top=top,
        imp=imp,
        vfps=vfps,
        poses=pose_list,
        ang_top=ang_top,
        ang_imp=ang_imp,
        suspicious=suspicious,
    )

    # A 的帧在 pose 上可能仍是 argmax，但时间轴上明显偏 Top / 偏 Impact：在其余帧里选「接近中段且分不太差」的替代。
    if (
        always_pose_pick
        and int(best) == int(mds)
        and len(cand) > 1
        and imp > top + 2
    ):
        rel_m = _norm_along_corridor(mds, top, imp)
        off_corridor = rel_m < _REL_MIN or rel_m > _REL_MAX or abs(rel_m - _CORRIDOR_TARGET_REL) > 0.12
        if off_corridor:
            others = [
                idx
                for idx in cand
                if int(idx) != int(mds)
            ]
            if others:
                rescored: List[Tuple[int, float]] = [
                    (
                        idx,
                        _score_mid_down_candidate_mediapipe(
                            idx,
                            top=top,
                            imp=imp,
                            ang_top=ang_top,
                            ang_imp=ang_imp,
                            vfps=vfps,
                            poses=pose_list,
                        ),
                    )
                    for idx in others
                ]
                rescored.sort(
                    key=lambda t: (
                        abs(_norm_along_corridor(t[0], top, imp) - _CORRIDOR_TARGET_REL),
                        -t[1],
                    ),
                )
                alt_i, alt_s = rescored[0]
                if alt_s >= best_s - 0.07:
                    best, best_s = int(alt_i), float(alt_s)
                    pick_strategy = f"{pick_strategy}+corridor_rebias"

    dbg["pose_pick_strategy"] = pick_strategy

    by["Mid-downswing"]["frame_index"] = int(best)
    by["Mid-downswing"]["confidence"] = max(
        float(by["Mid-downswing"].get("confidence") or 0.0),
        min(0.92, 0.4 + min(0.45, max(0.0, best_s))),
    )
    _clamp_mid_downswing_only(by, hi)
    dbg["touched"].append("Mid-downswing")
    final_mds_i = int(by["Mid-downswing"]["frame_index"])
    if always_pose_pick:
        dbg["action"] = (
            "refined_mid_down_pose_pick" if final_mds_i != mds else "pose_pick_best_equals_a"
        )
    else:
        dbg["action"] = "refined_mid_down" if final_mds_i != mds else "refined_mid_down_unchanged"
    dbg["from_mds"] = mds
    dbg["to_mds"] = int(by["Mid-downswing"]["frame_index"])
    dbg["final_mds"] = dbg["to_mds"]

    out = [by[e] for e in EVENT_SEQUENCE]

    mds2 = int(_rows_by_event(out)["Mid-downswing"]["frame_index"])
    t2 = int(_rows_by_event(out)["Top"]["frame_index"])
    i2 = int(_rows_by_event(out)["Impact"]["frame_index"])
    rel2 = _norm_along_corridor(mds2, t2, i2)
    if rel2 < _REL_MIN:
        fail.append("mid_downswing_too_close_to_top")
    if rel2 > _REL_MAX:
        fail.append("mid_downswing_too_close_to_impact")
    if not (t2 < mds2 < i2):
        fail.append("mid_downswing_semantic_invalid")

    logger.info(
        "[downswing_refine] action=%s mds %s→%s fails=%s",
        dbg.get("action"),
        dbg.get("original_mds"),
        mds2,
        fail,
    )
    return out, fail, dbg


__all__ = [
    "refine_mid_downswing_with_pose_motion",
    "is_mid_downswing_suspicious",
]
