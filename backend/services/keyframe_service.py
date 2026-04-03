"""
Keyframe extraction service.

Two modes:
  1. extract_keyframes_smart() — pose-aware: uses detected swing phases to pick
     the single most representative frame per phase. Images match labels.
  2. extract_keyframes_and_ai_frames() — temporal: legacy uniform sampling for
     AI vision frames (fed to Gemini). Kept for backward compatibility.
"""

import cv2
import math
import numpy as np
import base64
import logging
from typing import Any, Optional

# Preserved across ensure_keyframes_ordered_for_ai re-validation
_REPAIR_FV_PRESERVE_KEYS = (
    "repair_log",
    "phase_strip_repaired",
    "reselected_top",
    "reselected_impact",
    "dt_fixed_count",
    "enforce_ok",
    "enforce_fail_reasons",
    "dt_min_times_axis",
)

from services.video_utils import get_video_rotation, apply_rotation, read_frame_pose_pipeline

logger = logging.getLogger(__name__)


def _pose_snapshot_for_keyframe(pose: dict) -> dict:
    """Normalized mini-joints for keyframe payload; prefers raw detection (sticky skeleton)."""
    from services.pose_service import get_render_joints

    joints = get_render_joints(pose)
    return {
        "joints": [
            {
                "name": j["name"],
                "nx": float(j.get("normalized", {}).get("x", 0.5)),
                "ny": float(j.get("normalized", {}).get("y", 0.5)),
                "v": float(j.get("visibility", 0)),
            }
            for j in joints
        ],
        "connections": pose.get("connections", []),
    }

SWING_PHASE_META = {
    "address":        {"label_en": "Address",        "label_zh": "准备"},
    "takeaway":       {"label_en": "Takeaway",       "label_zh": "起杆"},
    "backswing":      {"label_en": "Backswing",      "label_zh": "上杆"},
    "top":            {"label_en": "Top of Swing",   "label_zh": "顶点"},
    "downswing":      {"label_en": "Downswing",      "label_zh": "下杆"},
    "impact":         {"label_en": "Impact",         "label_zh": "击球"},
    "follow_through": {"label_en": "Follow-Through", "label_zh": "送杆"},
    "finish":         {"label_en": "Finish",         "label_zh": "收杆"},
}

PHASE_ORDER = list(SWING_PHASE_META.keys())


def _min_pose_index_gap(n_poses: int) -> int:
    return max(1, n_poses // 16)


def _prior_selected_pose_lower(selected: dict[str, dict], phase_id: str) -> int:
    pos = PHASE_ORDER.index(phase_id)
    lo = -1
    for j in range(pos):
        pid = PHASE_ORDER[j]
        if pid in selected:
            lo = max(lo, int(selected[pid]["pose_idx"]))
    return lo


def _later_selected_pose_upper(selected: dict[str, dict], phase_id: str, n_poses: int) -> int:
    pos = PHASE_ORDER.index(phase_id)
    hi = n_poses
    for j in range(pos + 1, len(PHASE_ORDER)):
        pid = PHASE_ORDER[j]
        if pid in selected:
            hi = min(hi, int(selected[pid]["pose_idx"]))
    return hi


def _pose_idx_allowed_for_phase(
    pose_idx: int,
    selected: dict[str, dict],
    phase_id: str,
    n_poses: int,
    used: set[int],
    gap_min: int,
) -> bool:
    if pose_idx in used or not (0 <= pose_idx < n_poses):
        return False
    lo = _prior_selected_pose_lower(selected, phase_id)
    hi = _later_selected_pose_upper(selected, phase_id, n_poses)
    if pose_idx <= lo or pose_idx >= hi:
        return False
    if lo >= 0 and pose_idx - lo < gap_min:
        return False
    if hi < n_poses and hi - pose_idx < gap_min:
        return False
    return True


# Legacy fixed-position phases (kept for fallback only)
SWING_PHASES = [
    {"name": "Address",        "label_en": "Address",        "label_zh": "准备站位", "position": 0.0},
    {"name": "Takeaway",       "label_en": "Takeaway",       "label_zh": "起杆",    "position": 0.12},
    {"name": "Backswing",      "label_en": "Backswing",      "label_zh": "上杆",    "position": 0.28},
    {"name": "Top",            "label_en": "Top of Swing",   "label_zh": "顶点",    "position": 0.42},
    {"name": "Downswing",      "label_en": "Downswing",      "label_zh": "下杆",    "position": 0.58},
    {"name": "Impact",         "label_en": "Impact",         "label_zh": "击球",    "position": 0.72},
    {"name": "Follow Through", "label_en": "Follow-Through", "label_zh": "送杆",    "position": 0.85},
    {"name": "Finish",         "label_en": "Finish",         "label_zh": "收杆",    "position": 0.96},
]

# ── Near-duplicate / visual diff helpers ──

# Histogram-only dup detection false-positives on golf (sky/grass dominate). Only treat high
# histogram correlation as duplicate when the pose is also similar to that prior keyframe.
_VISUAL_DIFF_THRESHOLD = 0.88
# Minimum visual delta vs prior strip (1 − worst pose-gated histogram correlation). Default ≈ 0.12.
_DEFAULT_MIN_VISUAL_DIFF = round(1.0 - _VISUAL_DIFF_THRESHOLD, 4)
# When 8-phase structure is OK but pose-gated histogram / min gap is borderline, one retry with slightly looser thresholds.
_STRIP_QUALITY_RELAX_MIN_VISUAL_DIFF = 0.06
_STRIP_QUALITY_RELAX_TIME_GAP_FACTOR = 0.88
_POSE_GATE_HIST_DUP = 0.20  # _pose_angle_distance below this vs a prior keyframe → hist comparable
_MIN_TIME_INTERVAL_RATIO = 1.0 / 24.0  # minimum fraction of total duration between adjacent keyframes
# When re-selecting, require pose-gated histogram correlation at or below this vs all accepted
_RESELECT_MAX_CORR_WITH_PREV = 0.85
# After repair, timestamp must be strictly after previous keyframe by at least this fraction of min_time_gap
_MIN_TIME_AFTER_PREV_RATIO = 0.85
_EPS_MONO_TIME = 1e-4

# Angle keys used for pose-distance metric
_ANGLE_KEYS_FOR_DISTANCE = [
    "left_elbow", "right_elbow", "left_knee", "right_knee",
    "left_shoulder", "right_shoulder", "x_factor", "spine_tilt",
]

# Phases where we bias toward visible motion change vs the previous strip keyframe (not uniform time).
_EARLY_STRIP_MOTION_PHASES = frozenset({"takeaway", "backswing", "top"})


def _pose_angle_distance(pose_a: dict, pose_b: dict) -> float:
    """Normalized L2 distance between two poses' angle vectors. Range ~0..1."""
    def _vec(p: dict) -> np.ndarray:
        angles = p.get("angles", {})
        return np.array([float(angles.get(k, 0.0)) for k in _ANGLE_KEYS_FOR_DISTANCE])
    va, vb = _vec(pose_a), _vec(pose_b)
    # Normalize by typical max range (~180 degrees per joint)
    diff = np.abs(va - vb) / 180.0
    return float(np.clip(np.sqrt(np.mean(diff ** 2)), 0.0, 1.0))


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalized grayscale histogram for visual similarity comparison."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _visual_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Histogram correlation between two frames. 1.0 = identical, 0.0 = completely different."""
    ha = _frame_histogram(frame_a)
    hb = _frame_histogram(frame_b)
    return float(cv2.compareHist(
        ha.reshape(-1, 1).astype(np.float32),
        hb.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL,
    ))


def _worst_pose_gated_histogram_similarity(
    frame: np.ndarray,
    pose: dict,
    accepted_frames: list[np.ndarray],
    accepted_poses: list[dict],
) -> float:
    """Max histogram correlation vs prior keyframes, only counting pairs with similar pose."""
    worst = 0.0
    for af, ap in zip(accepted_frames, accepted_poses):
        if _pose_angle_distance(pose, ap) > _POSE_GATE_HIST_DUP:
            continue
        worst = max(worst, _visual_similarity(frame, af))
    return worst


def _temporal_ratio_for_phase(phase_id: str) -> float:
    """Legacy SWING_PHASES position 0..1 for phase_id (address → finish)."""
    try:
        i = PHASE_ORDER.index(phase_id)
        return float(SWING_PHASES[i]["position"])
    except (ValueError, IndexError):
        return 0.5


def _min_timestamp_after_prev(prev_timestamp: float, min_time_gap: float) -> float:
    if prev_timestamp <= -100:
        return float("-inf")
    return prev_timestamp + max(min_time_gap * _MIN_TIME_AFTER_PREV_RATIO, _EPS_MONO_TIME)


_STRIP_VIOL_TO_FAIL_CODE = {
    "NEAR_DUPLICATE_CANDIDATE": "NEAR_DUPLICATE_PRESENT",
    "TIME_TOO_CLOSE_CANDIDATE": "TIME_TOO_CLOSE_PRESENT",
    "SOURCE_FRAME_NOT_INCREASING": "SOURCE_FRAME_ORDER_FAIL",
    "NO_LEGAL_CANDIDATE": "NO_LEGAL_CANDIDATE",
}


def _histogram_dup_threshold(*, min_visual_diff: float | None) -> float:
    """Worst allowed pose-gated histogram correlation vs prior strip (higher = more duplicate)."""
    if min_visual_diff is None:
        return float(_VISUAL_DIFF_THRESHOLD)
    return float(max(0.0, min(1.0, 1.0 - float(min_visual_diff))))


def _strip_pick_passes_hard_constraints(
    frame: np.ndarray,
    pose: dict,
    prev_fi: int,
    prev_ts: float,
    min_time_gap: float,
    acc_frames: list[np.ndarray],
    acc_poses: list[dict],
    *,
    min_visual_diff: float | None = None,
) -> tuple[bool, str | None]:
    """Hard strip gates at selection time: increasing raw frame index, min timestamp gap, visual min-diff vs prior picks."""
    fi = int(pose.get("frame_index", 0))
    ts = float(pose.get("timestamp", round(fi / max(30.0, 1e-6), 3)))
    sim_thr = _histogram_dup_threshold(min_visual_diff=min_visual_diff)
    if prev_fi >= 0 and fi <= prev_fi:
        return False, "SOURCE_FRAME_NOT_INCREASING"
    if prev_ts > -100:
        need_ts = _min_timestamp_after_prev(prev_ts, min_time_gap)
        if ts < need_ts:
            return False, "TIME_TOO_CLOSE_CANDIDATE"
    if acc_frames:
        worst = _worst_pose_gated_histogram_similarity(frame, pose, acc_frames, acc_poses)
        if worst > sim_thr:
            return False, "NEAR_DUPLICATE_CANDIDATE"
    return True, None


def recompute_keyframe_details_from_final_strip(
    cap: Any,
    rotation: Any,
    fps: float,
    poses: list[dict],
    keyframes: list[dict],
    min_time_gap: float,
    min_visual_diff: float | None = None,
    *,
    min_time_gap_factor: float = 1.0,
) -> list[dict]:
    """Rebuild per-phase detail rows from the **final** keyframe strip (decoded frames), using ``min_visual_diff`` contract."""
    mvd = float(min_visual_diff) if min_visual_diff is not None else float(_DEFAULT_MIN_VISUAL_DIFF)
    sim_thr = _histogram_dup_threshold(min_visual_diff=mvd)
    eff_min_gap = float(min_time_gap) * float(min_time_gap_factor)
    details: list[dict] = []
    acc_frames: list[np.ndarray] = []
    acc_poses: list[dict] = []
    prev_ts = -999.0
    prev_fi = -1
    for kf in keyframes:
        phase_id = str(kf.get("phase", ""))
        pi = int(kf.get("source_pose_idx", 0))
        fi = int(kf.get("source_frame_index", kf.get("frame_index", 0)))
        if not (0 <= pi < len(poses)):
            pi = max(0, min(pi, len(poses) - 1))
        pose = poses[pi]
        frame = _read_frame_pose_matched(cap, fi, rotation)
        if frame is None:
            frame = _read_frame_with_decode_fallback(cap, fi, rotation)
        if frame is None:
            details.append({
                "phase": phase_id,
                "visual_diff_from_prev": 0.0,
                "is_near_duplicate": False,
                "time_gap": 0.0,
                "time_too_close": False,
                "validation_passed": False,
                "fail_code": "FRAME_DECODE_FAIL",
            })
            prev_fi = max(prev_fi, fi)
            continue
        worst_similarity = 0.0
        visual_diff = 1.0
        is_near_duplicate = False
        if acc_frames:
            worst_similarity = _worst_pose_gated_histogram_similarity(frame, pose, acc_frames, acc_poses)
            visual_diff = round(1.0 - worst_similarity, 4)
            is_near_duplicate = worst_similarity > sim_thr
        ts = float(kf.get("timestamp", pose.get("timestamp", round(fi / max(fps, 1e-6), 3))))
        time_gap = round(ts - prev_ts if prev_ts >= 0 else 999.0, 3)
        time_too_close = time_gap < eff_min_gap and prev_ts >= 0
        fi_mono_ok = prev_fi < 0 or fi > prev_fi
        validation_passed = bool(
            not is_near_duplicate and not time_too_close and fi_mono_ok
        )
        fail_parts: list[str] = []
        if is_near_duplicate:
            fail_parts.append("NEAR_DUPLICATE_PRESENT")
        if time_too_close:
            fail_parts.append("TIME_TOO_CLOSE_PRESENT")
        if not fi_mono_ok:
            fail_parts.append("SOURCE_FRAME_NOT_INCREASING")
        row: dict[str, Any] = {
            "phase": phase_id,
            "visual_diff_from_prev": visual_diff,
            "is_near_duplicate": is_near_duplicate,
            "time_gap": time_gap,
            "time_too_close": time_too_close,
            "validation_passed": validation_passed,
            "recomputed_after_mutation": True,
            "min_visual_diff_applied": round(mvd, 4),
            "min_time_gap_factor_applied": round(float(min_time_gap_factor), 4),
        }
        if not validation_passed and fail_parts:
            row["fail_code"] = "+".join(fail_parts)
        details.append(row)
        acc_frames.append(frame)
        acc_poses.append(pose)
        prev_ts = ts
        prev_fi = fi
    return details


def _phase_candidate_pose_layers(center: int, n_poses: int) -> list[list[int]]:
    """Expanding pose-index rings ±8 → ±12 → ±16 (new indices only in each ring)."""
    layers: list[list[int]] = []
    prev_lo: int | None = None
    prev_hi: int | None = None
    for hw in (8, 12, 16, 20):
        lo, hi = max(0, center - hw), min(n_poses - 1, center + hw)
        if prev_lo is None:
            layer = list(range(lo, hi + 1))
        else:
            layer = [i for i in range(lo, hi + 1) if i < prev_lo or i > prev_hi]
        prev_lo, prev_hi = lo, hi
        layers.append(layer)
    return layers


def _legalize_phase_pick_for_strip(
    cap: Any,
    rotation: Any,
    poses: list[dict],
    phase_id: str,
    best: dict | None,
    candidates: list[int],
    preferred_idx: int | None,
    bucket: list[int],
    selected: dict[str, dict],
    used_pose_indices: set[int],
    min_pg: int,
    strip_prev_fi: int,
    strip_prev_ts: float,
    min_time_gap_early: float,
    strip_acc_frames: list[np.ndarray],
    strip_acc_poses: list[dict],
    kin_ctx: Any,
    exc_ctx: int,
    phase_keyframes: dict[str, int],
    min_visual_diff: float = _DEFAULT_MIN_VISUAL_DIFF,
) -> dict | None:
    """Progressive ±8/±12/±16 pose search; prefer semantic > time margin > visual diff > pose quality among legal picks."""
    from services.swing_flow_utils import (
        validate_impact_semantic_at_index,
        validate_top_semantic_at_index,
    )

    if best is None:
        return None
    pi0 = int(best["pose_idx"])
    fr0 = best["frame"]
    pose0 = poses[pi0]
    ok0, viol0 = _strip_pick_passes_hard_constraints(
        fr0, pose0, strip_prev_fi, strip_prev_ts, min_time_gap_early,
        strip_acc_frames, strip_acc_poses,
        min_visual_diff=min_visual_diff,
    )
    if ok0:
        return best

    def _semantic_ok(pi: int) -> bool:
        if kin_ctx is None:
            return True
        if phase_id == "top":
            ok_t, _ = validate_top_semantic_at_index(pi, kin_ctx)
            return bool(ok_t)
        if phase_id == "impact":
            top_pi = int(phase_keyframes.get("top", 0))
            ok_i, _ = validate_impact_semantic_at_index(pi, top_pi, exc_ctx, kin_ctx)
            return bool(ok_i)
        return True

    need_ts = _min_timestamp_after_prev(strip_prev_ts, min_time_gap_early) if strip_prev_ts > -100 else float("-inf")
    layers = _phase_candidate_pose_layers(pi0, len(poses))
    seed = [i for i in candidates + bucket[:20] if isinstance(i, int) and 0 <= i < len(poses)]
    seed_unique = list(dict.fromkeys(seed))
    seed_sorted = sorted(seed_unique, key=lambda i: (abs(i - pi0), i))
    layers[0] = list(dict.fromkeys(seed_sorted + layers[0]))

    def _rank_tuple(pi: int, frame: np.ndarray, pose: dict) -> tuple[int, float, float, float]:
        sem = 1 if _semantic_ok(pi) else 0
        ts = float(pose.get("timestamp", round(int(pose.get("frame_index", 0)) / 30.0, 3)))
        tg_margin = ts - need_ts
        if strip_acc_frames:
            worst = _worst_pose_gated_histogram_similarity(frame, pose, strip_acc_frames, strip_acc_poses)
            vd = 1.0 - worst
        else:
            vd = 1.0
        if (
            phase_id in _EARLY_STRIP_MOTION_PHASES
            and strip_acc_poses
            and isinstance(strip_acc_poses[-1], dict)
        ):
            vd = min(1.0, vd + 0.24 * _pose_angle_distance(pose, strip_acc_poses[-1]))
        pq = float(_pose_quality_details(pose)["quality"])
        return (sem, tg_margin, vd, pq)

    best_pick: dict | None = None

    for layer in layers:
        legal: list[tuple[int, np.ndarray, dict, tuple[int, float, float, float]]] = []
        layer_sorted = sorted(layer, key=lambda i: (abs(i - pi0), i))
        for e_pi in layer_sorted:
            if not _pose_idx_allowed_for_phase(
                e_pi, selected, phase_id, len(poses), used_pose_indices, min_pg
            ):
                continue
            efi = int(poses[e_pi].get("frame_index", 0))
            efr = _read_frame_pose_matched(cap, efi, rotation)
            if efr is None:
                efr = _read_frame_with_decode_fallback(cap, efi, rotation)
            if efr is None:
                continue
            ep = poses[e_pi]
            ok2, _viol2 = _strip_pick_passes_hard_constraints(
                efr, ep, strip_prev_fi, strip_prev_ts, min_time_gap_early,
                strip_acc_frames, strip_acc_poses,
                min_visual_diff=min_visual_diff,
            )
            if not ok2:
                continue
            rk = _rank_tuple(e_pi, efr, ep)
            legal.append((e_pi, efr, ep, rk))
        if not legal:
            continue
        chosen = max(legal, key=lambda t: t[3])
        e_pi, efr, ep, rk = chosen
        near_pref = 1.0
        if isinstance(preferred_idx, int):
            near_pref = 1.0 - min(abs(e_pi - preferred_idx) / max(len(bucket) + 1, 3), 1.0)
        pose_q = _pose_quality_details(ep)
        img_q = _frame_quality_details(efr)
        score = 0.35 * pose_q["quality"] + 0.25 * img_q["quality"] + 0.40 * near_pref
        if kin_ctx is not None:
            if phase_id == "top" and _semantic_ok(e_pi):
                score += 0.20
            elif phase_id == "impact" and _semantic_ok(e_pi):
                score += 0.22
        if e_pi in used_pose_indices:
            score *= 0.1
        best_pick = {
            "pose_idx": e_pi,
            "frame": efr,
            "confidence": float(np.clip(score, 0.0, 1.0)),
            "fallback_used": isinstance(preferred_idx, int) and e_pi != preferred_idx,
            "selection_reason": _phase_reason(
                phase_id, isinstance(preferred_idx, int) and e_pi != preferred_idx
            ),
        }
        break

    if best_pick is not None:
        return best_pick
    out = dict(best)
    out["strip_constraint_violation"] = viol0 or "NO_LEGAL_CANDIDATE"
    return out


def _commit_strip_accumulator(
    best: dict,
    poses: list[dict],
    fps: float,
    strip_acc_frames: list[np.ndarray],
    strip_acc_poses: list[dict],
) -> tuple[int, float]:
    """Update running strip state after a phase pick is accepted."""
    pi = int(best["pose_idx"])
    strip_acc_frames.append(best["frame"])
    strip_acc_poses.append(poses[pi])
    fi = int(poses[pi].get("frame_index", 0))
    ts = float(poses[pi].get("timestamp", round(fi / max(fps, 1e-6), 3)))
    return fi, ts


def _reselect_temporal_pose_order_ok(
    ts: float,
    pose_idx: int,
    video_frame_idx: int,
    prev_timestamp: float,
    prev_pose_idx: int,
    prev_frame_index: int,
    min_time_gap: float,
) -> bool:
    """Reselect must never move a phase earlier than the previous accepted keyframe."""
    if prev_timestamp <= -100:
        return True
    need_ts = _min_timestamp_after_prev(prev_timestamp, min_time_gap)
    if ts < need_ts:
        return False
    if prev_pose_idx >= 0 and pose_idx <= prev_pose_idx:
        return False
    if prev_frame_index >= 0 and video_frame_idx <= prev_frame_index:
        return False
    return True


def _reselect_distinct_keyframe(
    cap,
    rotation,
    fps: float,
    poses: list[dict],
    phase_id: str,
    start_pose_idx: int,
    all_accepted_frames: list[np.ndarray],
    all_accepted_poses: list[dict],
    prev_timestamp: float,
    min_time_gap: float,
    total_vid_frames: int,
    used_pose_indices: set[int] | None = None,
    prev_pose_idx: int = -1,
    prev_frame_index: int = -1,
) -> tuple[np.ndarray, int, int, float] | None:
    """Find a replacement frame/pose that is visually distinct from ALL accepted keyframes.

    Returns (frame_bgr, pose_idx_for_snapshot, video_frame_index, timestamp) or None.
    """
    n = len(poses)
    if n == 0 or total_vid_frames <= 0:
        return None
    _used = used_pose_indices or set()

    def _is_distinct(frame: np.ndarray, pose: dict) -> tuple[bool, float]:
        """Check frame against ALL accepted frames (pose-gated histogram)."""
        if not all_accepted_frames:
            return True, 1.0
        worst_sim = _worst_pose_gated_histogram_similarity(
            frame, pose, all_accepted_frames, all_accepted_poses
        )
        total_angle_dist = sum(
            _pose_angle_distance(pose, ap) for ap in all_accepted_poses
        )
        avg_angle_dist = total_angle_dist / len(all_accepted_poses) if all_accepted_poses else 0.5
        vis_dist = 1.0 - worst_sim
        combined = 0.4 * vis_dist + 0.6 * avg_angle_dist
        return worst_sim <= _RESELECT_MAX_CORR_WITH_PREV, combined

    def ts_of(pi: int) -> float:
        fi = int(poses[pi].get("frame_index", 0))
        return float(poses[pi].get("timestamp", round(fi / max(fps, 1e-6), 3)))

    # ── Strategy 1: Pose-window search (±20% of pose count) ──
    span = max(int(n * 0.20), 8)
    lo = max(0, start_pose_idx - span)
    hi = min(n, start_pose_idx + span + 1)
    candidates = [i for i in range(lo, hi) if i not in _used]

    # Sort by distance from ideal timestamp
    if prev_timestamp > -100:
        target_ts = prev_timestamp + min_time_gap * 1.2
        candidates.sort(key=lambda i: (abs(ts_of(i) - target_ts), abs(i - start_pose_idx)))
    else:
        candidates.sort(key=lambda i: abs(i - start_pose_idx))

    best: tuple[np.ndarray, int, int, float, float] | None = None
    best_rank = -1.0
    for pi in candidates:
        fi = int(poses[pi].get("frame_index", 0))
        frame = _read_frame_pose_matched(cap, fi, rotation)
        if frame is None:
            continue
        ts = ts_of(pi)
        if not _reselect_temporal_pose_order_ok(
            ts, pi, fi, prev_timestamp, prev_pose_idx, prev_frame_index, min_time_gap,
        ):
            continue
        if prev_timestamp > -100 and (ts - prev_timestamp) < min_time_gap * 0.5:
            continue
        is_ok, combined_dist = _is_distinct(frame, poses[pi])
        if not is_ok:
            continue
        pq = _pose_quality_details(poses[pi])["quality"]
        iq = _frame_quality_details(frame)["quality"]
        rank = 0.52 * combined_dist + 0.28 * pq + 0.20 * iq
        if rank > best_rank:
            best_rank = rank
            best = (frame, pi, fi, ts)

    if best is not None:
        return best[0], best[1], best[2], best[3]

    # Do not jump to arbitrary raw timeline fallback frames. That can produce semantically wrong
    # phases and fake monotonic repairs even if constraints look satisfied.
    return None


def keyframes_to_ai_images(
    keyframes: list[dict], target_width: int = 384, quality: int = 84
) -> list[str]:
    """
    Resize keyframe JPEGs for vision models — same temporal order as keyframes
    (address → … → finish) so AI text matches displayed phase strips.
    """
    out: list[str] = []
    for kf in keyframes:
        try:
            raw = base64.b64decode(kf["image_base64"])
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            resized = _resize_frame(frame, target_width)
            out.append(frame_to_base64(resized, quality=quality))
        except Exception as exc:
            logger.warning("keyframes_to_ai_images skip frame: %s", exc)
    return out


def sync_keyframes_phase_map_and_pose_fields(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    keyframes: list[dict],
) -> None:
    """Align source_pose_idx, frame_index, source_frame_index with poses; refresh phase_keyframes map."""
    for kf in keyframes:
        pid = str(kf.get("phase") or "")
        if not pid:
            continue
        spi = kf.get("source_pose_idx")
        if spi is None and pid in phase_keyframes:
            spi = int(phase_keyframes[pid])
            kf["source_pose_idx"] = spi
        try:
            spi = int(spi) if spi is not None else None
        except (TypeError, ValueError):
            spi = None
        if spi is not None and 0 <= spi < len(poses):
            pose = poses[spi]
            fi = int(pose.get("frame_index", kf.get("frame_index", 0)))
            kf["source_pose_idx"] = spi
            kf["frame_index"] = fi
            kf["source_frame_index"] = fi
            if kf.get("timestamp") is None:
                kf["timestamp"] = round(float(pose.get("timestamp", 0.0)), 3)
            phase_keyframes[pid] = spi
    _sync_phase_keyframes_from_keyframes(keyframes, phase_keyframes)


def hydrate_keyframe_images_from_video(
    video_path: str,
    poses: list[dict],
    keyframes: list[dict],
    keyframe_width: int = 320,
) -> None:
    """Re-read video for keyframes with missing or corrupt JPEG payloads (same pose indices, no uniform strip)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    rotation = get_video_rotation(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for kf in keyframes:
        need = False
        b64 = kf.get("image_base64")
        if not b64:
            need = True
        else:
            try:
                raw = base64.b64decode(b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    need = True
            except Exception:
                need = True
        if not need:
            continue
        spi = kf.get("source_pose_idx")
        if spi is None or not (0 <= int(spi) < len(poses)):
            continue
        pose = poses[int(spi)]
        fi = int(pose.get("frame_index", kf.get("frame_index", 0)))
        frame = _read_frame_pose_matched(cap, fi, rotation)
        if frame is None:
            frame = _read_frame_with_decode_fallback(cap, fi, rotation)
        if frame is None:
            continue
        resized = _resize_frame(frame, keyframe_width)
        kf["image_base64"] = frame_to_base64(resized)
        kf["frame_index"] = fi
        kf["source_frame_index"] = fi
        kf["timestamp"] = round(float(pose.get("timestamp", fi / max(fps, 1e-6))), 3)
    cap.release()


def build_ai_vision_images_from_phase_keyframes(
    video_path: str,
    poses: list[dict],
    keyframes: list[dict],
    phase_keyframes: dict[str, int],
    *,
    target_width: int = 384,
    hydrate_width: int = 320,
) -> list[str]:
    """
    Images for vision models are always derived from the phase keyframe strip
    (after sync + optional hydration). Never use uniform temporal sampling as a fake phase row.
    """
    sync_keyframes_phase_map_and_pose_fields(poses, phase_keyframes, keyframes)
    hydrate_keyframe_images_from_video(video_path, poses, keyframes, hydrate_width)
    return keyframes_to_ai_images(keyframes, target_width=target_width)


def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


def _resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _read_frame_reliable(cap, target_frame_idx, rotation):
    """Sequential decode fallback when a single seek+read fails (rare)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(int(target_frame_idx), max(total - 1, 0))

    margin = 30
    safe_seek = max(0, target - margin)
    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_seek)

    frame = None
    for _ in range(target - safe_seek + 1):
        ret, f = cap.read()
        if ret:
            frame = f
        else:
            break

    if frame is None:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            target / max(cap.get(cv2.CAP_PROP_FPS) or 30.0, 1.0) * 1000.0,
        )
        ret, f = cap.read()
        if ret:
            frame = f

    if frame is not None:
        frame = apply_rotation(frame, rotation)
    return frame


def _read_frame_pose_matched(cap, target_frame_idx: int, rotation) -> np.ndarray | None:
    """Identical to the frame ``extract_poses_from_video`` used for this index."""
    return read_frame_pose_pipeline(cap, int(target_frame_idx), rotation)


def _read_frame_with_decode_fallback(cap, target_frame_idx: int, rotation) -> np.ndarray | None:
    """Search / temporal paths only — pipeline first, then sequential decode."""
    f = read_frame_pose_pipeline(cap, int(target_frame_idx), rotation)
    if f is not None:
        return f
    return _read_frame_reliable(cap, target_frame_idx, rotation)


def _pose_quality_details(pose: dict) -> dict:
    joints = {j.get("name"): j for j in pose.get("joints", [])}
    required = [
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_wrist", "right_wrist", "left_knee", "right_knee",
    ]
    vis_vals = []
    in_frame = 0
    for name in required:
        j = joints.get(name)
        if not j:
            vis_vals.append(0.0)
            continue
        v = float(j.get("visibility", 0.0))
        vis_vals.append(np.clip(v, 0.0, 1.0))
        nx = float(j.get("normalized", {}).get("x", 0.5))
        ny = float(j.get("normalized", {}).get("y", 0.5))
        if 0.02 <= nx <= 0.98 and 0.02 <= ny <= 0.98:
            in_frame += 1

    mean_vis = float(np.mean(vis_vals)) if vis_vals else 0.0
    completeness = in_frame / max(len(required), 1)
    key_joint_vis = []
    for name in ("left_wrist", "right_wrist", "left_hip", "right_hip"):
        j = joints.get(name)
        key_joint_vis.append(float(j.get("visibility", 0.0)) if j else 0.0)
    key_joint_vis_avg = float(np.mean(key_joint_vis)) if key_joint_vis else 0.0

    quality = float(np.clip(
        0.45 * mean_vis + 0.35 * completeness + 0.20 * key_joint_vis_avg,
        0.0,
        1.0,
    ))
    return {
        "quality": quality,
        "visibility": mean_vis,
        "completeness": completeness,
        "key_joint_visibility": key_joint_vis_avg,
    }


def _frame_quality_details(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = float(np.clip(lap_var / 300.0, 0.0, 1.0))

    h, w = gray.shape[:2]
    border = max(6, int(min(h, w) * 0.05))
    center = gray[border:h - border, border:w - border] if h > 2 * border and w > 2 * border else gray
    center_var = float(np.var(center))
    whole_var = float(np.var(gray)) + 1e-6
    border_ratio = float(np.clip(center_var / whole_var, 0.0, 2.0))
    subject_centered = float(np.clip(border_ratio / 1.2, 0.0, 1.0))

    non_blur = sharpness
    quality = float(np.clip(0.70 * sharpness + 0.30 * subject_centered, 0.0, 1.0))
    return {
        "quality": quality,
        "sharpness": sharpness,
        "non_blur": non_blur,
        "subject_centered": subject_centered,
    }


def _phase_reason(phase_id: str, is_fallback: bool) -> str:
    if is_fallback:
        return "neighbor_window_fallback"
    reasons = {
        "address": "stable_setup_min_hand_speed",
        "takeaway": "first_clear_departure_from_address",
        "backswing": "representative_mid_backswing_event",
        "top": "peak_hand_height_with_direction_reversal",
        "downswing": "clear_downswing_acceleration_after_top",
        "impact": "multi_signal_impact_score_peak",
        "follow_through": "post_impact_release_high_speed",
        "finish": "stable_completed_finish_pose",
    }
    return reasons.get(phase_id, "phase_event_selection")


def _sync_phase_keyframes_from_keyframes(
    keyframes: list[dict], phase_keyframes: dict[str, int]
) -> None:
    """Final response.phase_keyframes must match the poses actually encoded in keyframes."""
    for kf in keyframes:
        pid = kf.get("phase")
        spi = kf.get("source_pose_idx")
        if pid and spi is not None:
            phase_keyframes[str(pid)] = int(spi)


def _refresh_keyframe_validation_time_gaps(
    keyframe_validation: list[dict], keyframes: list[dict], min_time_gap: float
) -> None:
    if len(keyframe_validation) != len(keyframes):
        return
    prev_ts = -999.0
    for d, kf in zip(keyframe_validation, keyframes):
        ts = float(kf["timestamp"])
        time_gap = round((ts - prev_ts) if prev_ts >= 0 else 999.0, 3)
        d["time_gap"] = time_gap
        d["time_too_close"] = time_gap < min_time_gap and prev_ts >= 0
        d["validation_passed"] = not d.get("is_near_duplicate", False) and not d["time_too_close"]
        prev_ts = ts


def _strict_increasing_ts_and_fi(keyframes: list[dict]) -> tuple[bool, bool]:
    """Strictly increasing timestamp and video frame_index along output order (AI-safe)."""
    if len(keyframes) == 0:
        return False, False
    if len(keyframes) < 2:
        return True, True
    fi_ok = True
    ts_ok = True
    prev_ts = float(keyframes[0]["timestamp"])
    prev_fi = int(keyframes[0].get("frame_index", 0))
    for i in range(1, len(keyframes)):
        ts = float(keyframes[i]["timestamp"])
        fi = int(keyframes[i].get("frame_index", 0))
        if fi <= prev_fi:
            fi_ok = False
        if ts <= prev_ts:
            ts_ok = False
        prev_ts, prev_fi = ts, fi
    return fi_ok, ts_ok


def _strict_increasing_source_pose_idx(keyframes: list[dict]) -> bool:
    if len(keyframes) < 2:
        return True
    prev = int(keyframes[0].get("source_pose_idx", -1))
    for i in range(1, len(keyframes)):
        cur = int(keyframes[i].get("source_pose_idx", -1))
        if cur <= prev:
            return False
        prev = cur
    return True


def _strict_increasing_source_frame_idx(keyframes: list[dict]) -> bool:
    if len(keyframes) < 2:
        return True
    prev = int(keyframes[0].get("source_frame_index", keyframes[0].get("frame_index", -1)))
    for i in range(1, len(keyframes)):
        cur = int(keyframes[i].get("source_frame_index", keyframes[i].get("frame_index", -1)))
        if cur <= prev:
            return False
        prev = cur
    return True


def _phase_min_gap_contract_ok(keyframes: list[dict], fps: float) -> bool:
    if len(keyframes) != 8:
        return False
    ok, _reasons = _source_frame_index_phase_gaps_ok(keyframes, float(fps))
    return bool(ok)


def _keyframe_atomic_contract_ok(keyframes: list[dict], fps: float, min_time_gap: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if len(keyframes) != 8:
        reasons.append("PHASE_COUNT_NOT_8")
    phases = [str(k.get("phase", "")) for k in keyframes]
    if len(keyframes) == 8 and phases != PHASE_ORDER:
        reasons.append("PHASE_SEQUENCE_MISMATCH")
    fi_ok, ts_ok = _strict_increasing_ts_and_fi(keyframes)
    if not fi_ok:
        reasons.append("FRAME_INDEX_NOT_MONOTONIC")
    if not ts_ok:
        reasons.append("TIMESTAMP_NOT_MONOTONIC")
    if not _strict_increasing_source_pose_idx(keyframes):
        reasons.append("SOURCE_POSE_NOT_MONOTONIC")
    if not _strict_increasing_source_frame_idx(keyframes):
        reasons.append("SOURCE_FRAME_NOT_MONOTONIC")
    for i in range(1, len(keyframes)):
        prev_ts = float(keyframes[i - 1].get("timestamp", 0.0))
        cur_ts = float(keyframes[i].get("timestamp", 0.0))
        if cur_ts - prev_ts < 0:
            reasons.append("NEGATIVE_TIME_GAP_IN_DETAILS")
            break
    if len(keyframes) == 8 and (
        int(keyframes[PHASE_ORDER.index("impact")].get("source_pose_idx", -1))
        <= int(keyframes[PHASE_ORDER.index("downswing")].get("source_pose_idx", -1))
    ):
        reasons.append("IMPACT_NOT_AFTER_DOWNSWING")
    if len(keyframes) == 8 and not _phase_min_gap_contract_ok(keyframes, fps):
        reasons.append("MIN_GAP_VIOLATION")
    if len(keyframes) >= 2:
        for i in range(1, len(keyframes)):
            if float(keyframes[i].get("timestamp", 0.0)) < _min_timestamp_after_prev(
                float(keyframes[i - 1].get("timestamp", 0.0)), float(min_time_gap)
            ):
                reasons.append("MIN_TIME_GAP_VIOLATION")
                break
    return len(reasons) == 0, reasons


# Strict / semantic failures that require joint post-top map repair (not strip-quality alone).
_JOINT_REPAIR_STRICT_PREFIXES = (
    "FRAME_INDEX_NOT_MONOTONIC",
    "TIMESTAMP_NOT_MONOTONIC",
    "NEGATIVE_TIME_GAP_IN_DETAILS",
    "SOURCE_FRAME_NOT_MONOTONIC",
    "SOURCE_POSE_NOT_MONOTONIC",
    "IMPACT_NOT_AFTER_DOWNSWING",
    "MIN_GAP_VIOLATION",
)
_JOINT_REPAIR_SEMANTIC_TRIGGERS = frozenset(
    {
        "PHASE_POSE_INDEX_NOT_INCREASING",
        "PHASE_STRIP_PHASE_ORDER_MISMATCH_AFTER_POSE_SORT",
        "MIN_GAP_VIOLATION:DOWNSWING_IMPACT",
        "MIN_GAP_VIOLATION:NON_MONOTONIC_RAW_FRAMES",
    }
)


def _strict_reasons_need_joint_repair(strict_reasons: list[str] | None) -> bool:
    for r in strict_reasons or []:
        rs = str(r)
        if any(rs.startswith(p) for p in _JOINT_REPAIR_STRICT_PREFIXES):
            return True
    return False


def keyframe_repair_score(gate: dict, rows: list[dict]) -> tuple[int, int, int, int]:
    """Lexicographic: lower is better. For comparing repair candidates when strict gate still fails."""
    strict_reasons = list(gate.get("strict_contract_fail_reasons") or [])
    sem_reasons = list(gate.get("semantic_strip_reasons") or [])
    idx = {
        str(k.get("phase")): int(k.get("source_frame_index", k.get("frame_index", -1)))
        for k in (rows or [])
    }
    imp = idx.get("impact", -1)
    ft = idx.get("follow_through", -1)
    fin = idx.get("finish", -1)
    post_pen = 0
    if imp >= 0 and ft >= 0 and ft - imp < 8:
        post_pen += 1
    if ft >= 0 and fin >= 0 and fin - ft < 8:
        post_pen += 1
    missing = sum(1 for r in (rows or []) if not str(r.get("image_base64") or "").strip())
    return (len(strict_reasons), len(sem_reasons), post_pen, missing)


def _squeeze_phase_pose_indices_monotonic(phase_map: dict[str, int], n_poses: int) -> dict[str, int]:
    """Monotonic phase pose indices with minimum spacing so adjacent labels are not duplicate poses.

    Previously used ``prev + 1`` only, which allowed e.g. top=54 and downswing=55 (near-identical
    thumbnails) and fed strict near-duplicate gates.
    """
    order = PHASE_ORDER
    k = len(order)
    if n_poses < 2:
        return {pid: 0 for pid in order}
    # Target spacing scales with track length; cap so short clips still get a chain.
    ms = max(2, min(8, n_poses // 18))
    max_feasible = max(1, (n_poses - 1) // max(1, k - 1))
    ms = max(1, min(ms, max_feasible))
    prev = -1
    out: dict[str, int] = {}
    for j, pid in enumerate(order):
        remaining = k - j - 1
        lo = 0 if j == 0 else prev + ms
        hi = n_poses - 1 - remaining * ms
        if hi < lo:
            hi = min(n_poses - 1, lo)
        c = int(phase_map.get(pid, lo))
        c = max(lo, min(c, hi))
        out[pid] = c
        prev = c
    return out


def _enforce_increasing_source_frames_on_phase_map(
    poses: list[dict],
    phase_map: dict[str, int],
    *,
    fps: float,
) -> dict[str, int]:
    """Walk phases in order; nudge pose indices forward so video frame_index strictly increases."""
    n = len(poses)
    if n < 2:
        return dict(phase_map)
    min_step = max(2, int(round(float(fps) * 0.04)))
    out = dict(phase_map)
    prev_pi = -1
    prev_fi = -10**9
    for pid in PHASE_ORDER:
        start_pi = max(prev_pi + 1, int(out.get(pid, prev_pi + 1)))
        start_pi = min(max(0, start_pi), n - 1)
        chosen: int | None = None
        for pi2 in range(start_pi, n):
            fi2 = int(poses[pi2].get("frame_index", pi2))
            if pi2 > prev_pi and fi2 > prev_fi and fi2 >= prev_fi + min_step:
                chosen = pi2
                break
        if chosen is None:
            for pi2 in range(start_pi, n):
                fi2 = int(poses[pi2].get("frame_index", pi2))
                if pi2 > prev_pi and fi2 > prev_fi:
                    chosen = pi2
                    break
        if chosen is None:
            chosen = start_pi
        out[pid] = int(chosen)
        prev_pi = int(chosen)
        prev_fi = int(poses[prev_pi].get("frame_index", prev_pi))
    return out


def joint_rebuild_phase_map_for_monotonic_strip(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    *,
    fps: float,
) -> tuple[dict[str, int], bool]:
    """Jointly rebuild post-top phases + enforce monotonic pose and raw frame order.

    Returns (merged_phase_map, material_change_vs_input).
    """
    from services.phase_chain_solver_service import solve_full_phase_chain
    from services.swing_flow_utils import propose_post_impact_chain_indices

    n = len(poses)
    if n < 8:
        return dict(phase_keyframes), False
    base = _squeeze_phase_pose_indices_monotonic(dict(phase_keyframes), n)
    base = _enforce_increasing_source_frames_on_phase_map(poses, base, fps=float(fps))
    before = {p: int(phase_keyframes.get(p, -1)) for p in PHASE_ORDER}
    m = dict(base)
    chain = propose_post_impact_chain_indices(poses, m)
    if chain:
        for k, v in chain.items():
            m[k] = int(v)
    else:
        sol = solve_full_phase_chain(poses, m, phase_windows=None, detections=None, tracks=None, motion_3d=None)
        if sol.get("ok"):
            fm = dict(sol.get("phase_keyframes") or {})
            seq = [int(fm[p]) for p in PHASE_ORDER if p in fm]
            ok_chain = len(seq) == len(PHASE_ORDER) and all(
                seq[i] < seq[i + 1] for i in range(len(seq) - 1)
            )
            if ok_chain:
                for k in ("downswing", "impact", "follow_through", "finish"):
                    if k in fm:
                        m[k] = int(fm[k])
    m = _squeeze_phase_pose_indices_monotonic(m, n)
    m = _enforce_increasing_source_frames_on_phase_map(poses, m, fps=float(fps))
    after = {p: int(m.get(p, -1)) for p in PHASE_ORDER}
    material = any(before[p] != after[p] for p in PHASE_ORDER)
    return m, material


def joint_rebuild_phase_map_for_quality_spacing(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    *,
    fps: float,
) -> tuple[dict[str, int], bool]:
    """Rebuild post-top phase indices to widen pose/raw-frame spacing (quality gate failures).

    Used when monotonic / semantic strip / sync already pass but strict AI gate still reports
    NEAR_DUPLICATE_PRESENT and/or TIME_TOO_CLOSE_PRESENT and local strip-quality repair no-ops.
    """
    from services.swing_flow_utils import propose_quality_spacing_post_top_chain

    n = len(poses)
    if n < 8:
        return dict(phase_keyframes), False
    before = {p: int(phase_keyframes.get(p, -1)) for p in PHASE_ORDER}
    base = dict(phase_keyframes)
    for boost in (1.0, 1.45, 1.9, 2.35, 2.9, 3.4):
        chain = propose_quality_spacing_post_top_chain(poses, base, fps=float(fps), spacing_boost=boost)
        if not chain:
            continue
        m = dict(base)
        for k, v in chain.items():
            m[k] = int(v)
        m = _squeeze_phase_pose_indices_monotonic(m, n)
        m = _enforce_increasing_source_frames_on_phase_map(poses, m, fps=float(fps))
        after = {p: int(m.get(p, -1)) for p in PHASE_ORDER}
        if any(before[p] != after[p] for p in PHASE_ORDER):
            return m, True
    return dict(phase_keyframes), False


def _finalize_smart_keyframes_monotonic(
    keyframes: list[dict],
    keyframe_validation: list[dict],
    poses: list[dict],
    phase_keyframes: dict[str, int],
    selected: dict[str, dict],
    cap,
    rotation,
    fps: float,
    keyframe_width: int,
    min_time_gap: float,
) -> tuple[bool, bool, bool]:
    """Repair any phase that ended up before the previous keyframe in time or frame index."""
    if len(keyframes) < 2:
        return True, True, False
    n = len(poses)
    max_rounds = max(8, len(keyframes) * 3)

    for _ in range(max_rounds):
        fi_ok, ts_ok = _strict_increasing_ts_and_fi(keyframes)
        if fi_ok and ts_ok:
            break
        changed = False
        for i in range(1, len(keyframes)):
            prev_kf = keyframes[i - 1]
            kf = keyframes[i]
            pts = float(prev_kf["timestamp"])
            pfi = int(prev_kf.get("frame_index", 0))
            ppi = int(prev_kf.get("source_pose_idx", -1))
            ts = float(kf["timestamp"])
            fi = int(kf.get("frame_index", 0))
            pi = int(kf.get("source_pose_idx", -1))
            need_ts = _min_timestamp_after_prev(pts, min_time_gap)
            bad = fi <= pfi or ts <= pts or ts < need_ts or (pi >= 0 and ppi >= 0 and pi <= ppi)
            if not bad:
                continue

            phase_id = str(kf["phase"])
            next_pi = None
            next_fi = None
            next_ts = None
            if i + 1 < len(keyframes):
                nkf = keyframes[i + 1]
                next_pi = int(nkf.get("source_pose_idx", -1))
                next_fi = int(nkf.get("frame_index", -1))
                next_ts = float(nkf.get("timestamp", -1.0))
            start_pi = max(ppi + 1, pi + 1, 0)
            end_pi = n
            if next_pi is not None and next_pi >= 0:
                end_pi = min(end_pi, next_pi)
            scan_end = max(start_pi, end_pi)
            if start_pi >= scan_end and len(keyframes) == 8:
                # Local forward scan is empty (e.g. impact pose_idx < downswing): joint-rebuild post-top map.
                new_map, _mat = joint_rebuild_phase_map_for_monotonic_strip(
                    poses, dict(phase_keyframes), fps=float(fps),
                )
                phase_keyframes.clear()
                phase_keyframes.update(new_map)
                tmp_kf, tmp_det = _rebind_keyframes_from_rebuilt_map(
                    cap,
                    rotation,
                    float(fps),
                    poses,
                    dict(phase_keyframes),
                    keyframe_width,
                )
                for j, row in enumerate(tmp_kf):
                    if j < len(keyframes):
                        keyframes[j] = row
                    pid = str(row.get("phase", PHASE_ORDER[j] if j < len(PHASE_ORDER) else ""))
                    if pid in selected:
                        raw = base64.b64decode(row["image_base64"])
                        arr = np.frombuffer(raw, dtype=np.uint8)
                        fr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if fr is not None:
                            selected[pid] = {
                                "pose_idx": int(row["source_pose_idx"]),
                                "frame": fr,
                                "confidence": float(row["confidence"]),
                                "fallback_used": True,
                                "selection_reason": "joint_strip_rebuild_after_monotonic_deadlock",
                            }
                for j in range(min(len(keyframe_validation), len(tmp_det))):
                    keyframe_validation[j] = {
                        **keyframe_validation[j],
                        **tmp_det[j],
                        "monotonic_repair": True,
                        "joint_strip_rebuild": True,
                    }
                for j in range(len(keyframe_validation), len(tmp_det)):
                    keyframe_validation.append(
                        {**tmp_det[j], "monotonic_repair": True, "joint_strip_rebuild": True}
                    )
                logger.warning(
                    "[keyframe] monotonic deadlock at phase=%s — applied joint strip rebuild",
                    phase_id,
                )
                changed = True
                break
            for cand_pi in range(start_pi, scan_end):
                pose = poses[cand_pi]
                cand_fi = int(pose.get("frame_index", 0))
                cand_ts = float(pose.get("timestamp", round(cand_fi / max(fps, 1e-6), 3)))
                if cand_fi <= pfi:
                    continue
                if cand_ts < _min_timestamp_after_prev(pts, min_time_gap):
                    continue
                if next_fi is not None and next_fi >= 0 and cand_fi >= next_fi:
                    continue
                if next_ts is not None and next_ts >= 0 and cand_ts >= next_ts:
                    continue
                frame = _read_frame_pose_matched(cap, cand_fi, rotation)
                if frame is None:
                    frame = _read_frame_with_decode_fallback(cap, cand_fi, rotation)
                if frame is None:
                    continue
                resized = _resize_frame(frame, keyframe_width)
                pose_snapshot = _pose_snapshot_for_keyframe(pose)
                pq = _pose_quality_details(pose)
                iq = _frame_quality_details(frame)
                conf = float(np.clip(0.35 * pq["quality"] + 0.25 * iq["quality"] + 0.40, 0.0, 1.0))
                kf["frame_index"] = cand_fi
                kf["timestamp"] = round(cand_ts, 3)
                kf["source_pose_idx"] = cand_pi
                kf["source_frame_index"] = cand_fi
                kf["image_base64"] = frame_to_base64(resized)
                kf["width"] = resized.shape[1]
                kf["height"] = resized.shape[0]
                kf["pose_snapshot"] = pose_snapshot
                kf["confidence"] = round(max(0.15, float(kf.get("confidence") or conf) * 0.88), 3)
                kf["selection_reason"] = str(kf.get("selection_reason") or "") + "_monotonic_repair"
                kf["fallback_used"] = True
                kf["reselected"] = True
                kf["phase_validation_passed"] = True
                phase_keyframes[phase_id] = cand_pi
                if phase_id in selected:
                    selected[phase_id] = {
                        "pose_idx": cand_pi,
                        "frame": frame,
                        "confidence": float(kf["confidence"]),
                        "fallback_used": True,
                        "selection_reason": "monotonic_final_repair",
                    }
                if i < len(keyframe_validation):
                    keyframe_validation[i]["monotonic_repair"] = True
                logger.warning(
                    "[keyframe] %s: monotonic repair pose_idx %s -> %s (fi %s -> %s)",
                    phase_id, pi, cand_pi, fi, cand_fi,
                )
                changed = True
                break
        if not changed:
            break

    fi_ok, ts_ok = _strict_increasing_ts_and_fi(keyframes)
    failed = not (fi_ok and ts_ok)
    if failed:
        for i in range(1, len(keyframes)):
            prev_kf = keyframes[i - 1]
            cur_kf = keyframes[i]
            prev_fi = int(prev_kf.get("frame_index", 0))
            cur_fi = int(cur_kf.get("frame_index", 0))
            prev_ts = float(prev_kf.get("timestamp", 0.0))
            cur_ts = float(cur_kf.get("timestamp", 0.0))
            if cur_fi <= prev_fi or cur_ts <= prev_ts:
                logger.warning(
                    "[keyframe] final monotonic check still bad: %s after %s (fi %s/%s ts %s/%s)",
                    cur_kf.get("phase"),
                    prev_kf.get("phase"),
                    prev_fi,
                    cur_fi,
                    prev_ts,
                    cur_ts,
                )
    return fi_ok, ts_ok, failed


def repair_phase_strip_by_pose_order(
    keyframes: list[dict],
    poses: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """Sort by ``source_pose_idx``, sync fi/ts from ``poses``, enforce monotonic time — **no phase relabel**."""
    log: list[str] = []
    meta_out: dict[str, Any] = {
        "log": log,
        "relabel_count": 0,
        "pose_sort_phase_mismatch": 0,
        "enforce_ok": True,
        "enforce_fail_reasons": [],
    }
    if len(keyframes) != 8 or not poses:
        return keyframes, meta_out
    sorted_kf = sorted(
        [dict(k) for k in keyframes],
        key=lambda k: int(k.get("source_pose_idx") if k.get("source_pose_idx") is not None else -(10**9)),
    )
    orig_order_spi = [int(k.get("source_pose_idx", -1)) for k in keyframes]
    new_order_spi = [int(k.get("source_pose_idx", -1)) for k in sorted_kf]
    if orig_order_spi != new_order_spi:
        log.append("sorted_keyframes_by_source_pose_idx")
    # Count how many bulk relabels the **deprecated** path would have applied (forbidden).
    mismatch = sum(
        1
        for i, k in enumerate(sorted_kf)
        if str(k.get("phase") or "") != PHASE_ORDER[i]
    )
    meta_out["pose_sort_phase_mismatch"] = int(mismatch)
    for k in sorted_kf:
        spi = int(k["source_pose_idx"])
        ph = str(k.get("phase") or "")
        if ph in SWING_PHASE_META:
            meta = SWING_PHASE_META[ph]
            k["label_en"] = meta["label_en"]
            k["label_zh"] = meta["label_zh"]
        if 0 <= spi < len(poses):
            p = poses[spi]
            fi = int(p.get("frame_index", spi))
            k["frame_index"] = fi
            k["timestamp"] = round(float(p.get("timestamp", 0.0)), 3)
            k["source_frame_index"] = fi
    prev_fi = -1
    prev_ts = -1.0e9
    eps = 1e-4
    for k in sorted_kf:
        fi = int(k.get("frame_index", 0))
        ts = float(k.get("timestamp", 0.0))
        if fi <= prev_fi:
            fi = prev_fi + 1
            k["frame_index"] = fi
            k["source_frame_index"] = fi
            log.append("frame_index_monotonic_enforced")
        if ts <= prev_ts:
            ts = prev_ts + eps
            k["timestamp"] = round(ts, 3)
            log.append("timestamp_monotonic_enforced")
        prev_fi, prev_ts = fi, ts
    return sorted_kf, meta_out


def rebuild_phase_map_from_event_anchors(
    poses: list[dict],
    old_phase_keyframes: dict[str, int],
    *,
    min_gap: int = 3,
    local_window: int = 8,
) -> dict[str, Any]:
    """Rebuild monotonic 8-phase pose map from kinematic top/impact anchors + proportional spacing + local quality."""
    from services.swing_flow_utils import (
        _build_view_agnostic_kinematics,
        detect_phase_events_agnostic,
        validate_impact_semantic_at_index,
        validate_top_semantic_at_index,
    )

    reasons: list[str] = []
    n = len(poses)
    top_reselected = False
    impact_reselected = False
    if n < 8:
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": ["INSUFFICIENT_POSES_FOR_ANCHOR_REBUILD"],
            "top_reselected": False,
            "impact_reselected": False,
        }
    kin = _build_view_agnostic_kinematics(poses)
    if kin is None:
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": ["KINEMATICS_UNAVAILABLE"],
            "top_reselected": False,
            "impact_reselected": False,
        }
    kfc = list(kin.get("kinematic_fail_codes") or [])
    if kfc:
        reasons.extend(kfc)
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": reasons or ["KINEMATIC_FAIL"],
            "top_reselected": False,
            "impact_reselected": False,
        }

    ev = detect_phase_events_agnostic(poses)
    exc_apex = int(ev.get("excursion_apex_idx", max(1, n // 4)))
    top_ev = int(ev["top_pose_idx"])
    imp_ev = int(ev["impact_pose_idx"])
    w = local_window

    def pick_top_spi() -> int | None:
        if bool(ev.get("top_semantic_ok")) and validate_top_semantic_at_index(top_ev, kin)[0]:
            return top_ev
        best_i: int | None = None
        best_key: tuple | None = None
        lo = max(1, top_ev - w)
        hi = min(n - 2, top_ev + w)
        for i in range(lo, hi + 1):
            if not bool(kin["valid"][i]):
                continue
            ok, _ = validate_top_semantic_at_index(i, kin)
            if not ok:
                continue
            key = (float(kin["q"][i]), -abs(i - top_ev))
            if best_key is None or key > best_key:
                best_key = key
                best_i = i
        return best_i

    top_i = pick_top_spi()
    if top_i is None:
        reasons.append("TOP_ANCHOR_SEMANTIC_UNAVAILABLE")
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": reasons,
            "top_reselected": False,
            "impact_reselected": False,
        }

    def pick_impact_spi(for_top: int) -> int | None:
        lo = max(for_top + min_gap, exc_apex + 2, int(n * 0.42), imp_ev - w)
        hi = min(n - 2, int(n * 0.93), imp_ev + w)
        if lo >= hi:
            lo, hi = max(1, for_top + min_gap), min(n - 2, n - 1)
        best_i: int | None = None
        best_key: tuple | None = None
        for i in range(lo, hi + 1):
            if not bool(kin["valid"][i]):
                continue
            ok, _ = validate_impact_semantic_at_index(i, for_top, exc_apex, kin)
            if not ok:
                continue
            key = (float(kin["q"][i]), -abs(i - imp_ev))
            if best_key is None or key > best_key:
                best_key = key
                best_i = i
        return best_i

    imp_i = pick_impact_spi(top_i)
    if imp_i is None:
        reasons.append("IMPACT_ANCHOR_SEMANTIC_UNAVAILABLE")
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": reasons,
            "top_reselected": top_i != int(old_phase_keyframes.get("top", -1)),
            "impact_reselected": False,
        }

    if imp_i < top_i + min_gap:
        reasons.append("TOP_IMPACT_GAP_ANCHOR_FAIL")
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": reasons,
            "top_reselected": top_i != int(old_phase_keyframes.get("top", -1)),
            "impact_reselected": imp_i != int(old_phase_keyframes.get("impact", -1)),
        }

    old_top = int(old_phase_keyframes.get("top", -1)) if old_phase_keyframes.get("top") is not None else -1
    old_imp = int(old_phase_keyframes.get("impact", -1)) if old_phase_keyframes.get("impact") is not None else -1
    top_reselected = old_top >= 0 and top_i != old_top
    impact_reselected = old_imp >= 0 and imp_i != old_imp
    if old_top < 0:
        top_reselected = True
    if old_imp < 0:
        impact_reselected = True

    top_pos = float(SWING_PHASES[PHASE_ORDER.index("top")]["position"])
    imp_pos = float(SWING_PHASES[PHASE_ORDER.index("impact")]["position"])

    def lerp_idx(a: float, b: float, ua: float, ub: float, u: float) -> int:
        if ub <= ua:
            return int(round(b))
        t = (u - ua) / (ub - ua)
        return int(round(a + t * (b - a)))

    nom: dict[str, int] = {}
    nom["top"] = top_i
    nom["impact"] = imp_i

    # Pre-top: map 0..top_pos → [0, top_i]
    for j, pid in enumerate(("address", "takeaway", "backswing")):
        u = float(SWING_PHASES[PHASE_ORDER.index(pid)]["position"])
        raw = lerp_idx(0.0, float(top_i), 0.0, top_pos, u) if top_pos > 1e-9 else j
        nom[pid] = int(raw)
    nom["address"] = max(0, min(nom["address"], top_i - 3))
    nom["takeaway"] = max(nom["address"] + 1, min(nom["takeaway"], top_i - 2))
    nom["backswing"] = max(nom["takeaway"] + 1, min(nom["backswing"], top_i - 1))

    # Downswing between top and impact
    u_ds = float(SWING_PHASES[PHASE_ORDER.index("downswing")]["position"])
    nom["downswing"] = lerp_idx(top_i, imp_i, top_pos, imp_pos, u_ds)
    nom["downswing"] = max(top_i + 1, min(imp_i - 1, nom["downswing"]))
    if nom["downswing"] <= top_i or nom["downswing"] >= imp_i:
        nom["downswing"] = max(top_i + 1, min(imp_i - 1, top_i + max(1, (imp_i - top_i) // 2)))

    end_cap = n - 1
    u_ft = float(SWING_PHASES[PHASE_ORDER.index("follow_through")]["position"])
    u_fin = float(SWING_PHASES[PHASE_ORDER.index("finish")]["position"])
    span = float(end_cap - imp_i)
    if span < 2.0:
        reasons.append("INSUFFICIENT_POST_IMPACT_SPAN")
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": reasons,
            "top_reselected": top_reselected,
            "impact_reselected": impact_reselected,
        }
    nom["follow_through"] = int(round(imp_i + ((u_ft - imp_pos) / max(u_fin - imp_pos, 1e-6)) * span))
    nom["finish"] = int(round(imp_i + ((u_fin - imp_pos) / max(u_fin - imp_pos, 1e-6)) * span))
    nom["follow_through"] = max(imp_i + 1, min(end_cap - 1, nom["follow_through"]))
    nom["finish"] = max(nom["follow_through"] + 1, min(end_cap, nom["finish"]))

    # Monotonic squeeze **without** moving validated semantic anchors ``top_i`` / ``imp_i``.
    nom["top"] = top_i
    nom["impact"] = imp_i
    nom["address"] = max(0, min(nom["address"], top_i - 3))
    nom["takeaway"] = max(nom["address"] + 1, min(nom["takeaway"], top_i - 2))
    nom["backswing"] = max(nom["takeaway"] + 1, min(nom["backswing"], top_i - 1))
    nom["downswing"] = max(top_i + 1, min(nom["downswing"], imp_i - 1))
    nom["follow_through"] = max(imp_i + 1, min(nom["follow_through"], n - 2))
    nom["finish"] = max(nom["follow_through"] + 1, min(nom["finish"], n - 1))
    prev = -1
    top_ord = PHASE_ORDER.index("top")
    imp_ord = PHASE_ORDER.index("impact")
    for ord_i, pid in enumerate(PHASE_ORDER):
        if pid == "top":
            nom[pid] = top_i
            prev = top_i
            continue
        if pid == "impact":
            nom[pid] = imp_i
            prev = imp_i
            continue
        c = max(int(nom[pid]), prev + 1)
        if ord_i < top_ord:
            c = min(c, top_i - 1)
        elif top_ord < ord_i < imp_ord:
            c = min(c, imp_i - 1)
        else:
            c = min(c, n - 1)
        nom[pid] = c
        prev = c

    def local_pick(pid: str, center: int, lo_bound: int, hi_bound: int) -> int:
        lo = max(lo_bound + 1, center - w, 0)
        hi = min(hi_bound - 1, center + w, n - 1)
        if lo > hi:
            return max(lo_bound + 1, min(hi_bound - 1, center, n - 1))
        best_i = center
        best_q = -1.0
        for i in range(lo, hi + 1):
            if not bool(kin["valid"][i]):
                continue
            qv = float(kin["q"][i])
            if qv > best_q:
                best_q = qv
                best_i = i
        return int(best_i)

    rebuilt: dict[str, int] = {}
    prev = -1
    for idx, pid in enumerate(PHASE_ORDER):
        center = int(nom[pid])
        next_center = int(nom[PHASE_ORDER[idx + 1]]) if idx + 1 < len(PHASE_ORDER) else n
        lo_b = prev
        hi_b = max(next_center, min(n, center + w * 2 + 2))
        if pid in ("top", "impact"):
            c = int(center)
        else:
            c = local_pick(pid, center, lo_b, hi_b)
        c = max(c, prev + 1)
        c = min(c, n - 1)
        rebuilt[pid] = c
        prev = c

    prev = -1
    ok_order = True
    for pid in PHASE_ORDER:
        ix = rebuilt[pid]
        if ix <= prev:
            ok_order = False
        prev = ix
    if not ok_order or rebuilt["impact"] < rebuilt["top"] + min_gap:
        reasons.append("REBUILD_MONOTONIC_FAIL")
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": reasons,
            "top_reselected": top_reselected,
            "impact_reselected": impact_reselected,
        }

    t_ok, _ = validate_top_semantic_at_index(rebuilt["top"], kin)
    i_ok, _ = validate_impact_semantic_at_index(
        rebuilt["impact"], rebuilt["top"], exc_apex, kin,
    )
    if not t_ok:
        reasons.append("TOP_SEMANTIC_POST_REBUILD_FAIL")
    if not i_ok:
        reasons.append("IMPACT_SEMANTIC_POST_REBUILD_FAIL")
    if not t_ok or not i_ok:
        return {
            "phase_keyframes_rebuilt": {},
            "rebuild_ok": False,
            "rebuild_reasons": reasons,
            "top_reselected": top_reselected,
            "impact_reselected": impact_reselected,
        }

    return {
        "phase_keyframes_rebuilt": rebuilt,
        "rebuild_ok": True,
        "rebuild_reasons": [],
        "top_reselected": top_reselected,
        "impact_reselected": impact_reselected,
    }


def build_semantic_oriented_phase_map(
    poses: list[dict],
    seed_phase_keyframes: dict[str, int],
    *,
    fps: float,
) -> dict[str, Any]:
    """Build an 8-phase pose-index map from kinematic events (not uniform time or lerp).

    Top / impact are chosen by composite kinematic scores with semantic validation preferred.
    Takeaway / backswing use excursion milestones between address and top.
    Downswing / follow / finish prefer ``propose_post_impact_chain_indices`` (joint semantic chain).
    """
    from services.swing_flow_utils import (
        _build_view_agnostic_kinematics,
        _composite_impact_candidate_score,
        _composite_top_candidate_score,
        detect_phase_events_agnostic,
        propose_post_impact_chain_indices,
        validate_finish_semantic_at_index,
        validate_follow_through_semantic_at_index,
        validate_impact_semantic_at_index,
        validate_top_semantic_at_index,
    )

    n = len(poses)
    empty_debug = {
        "phase_reselect_strategy": "semantic_oriented_v1",
        "phase_candidate_window": {},
        "phase_candidate_scores": {},
        "top_semantic_score": None,
        "impact_semantic_score": None,
        "finish_semantic_score": None,
        "final_phase_semantic_pass": False,
        "final_phase_semantic_fail_reasons": ["INSUFFICIENT_POSES"],
        "source_pose_idx_list": [],
        "min_gap_frames": 0,
        "duplicate_pairs": [],
    }
    if n < 8:
        return {"ok": False, "phase_keyframes": {}, "debug": empty_debug}

    kin = _build_view_agnostic_kinematics(poses)
    if kin is None:
        d = {**empty_debug, "final_phase_semantic_fail_reasons": ["KINEMATICS_UNAVAILABLE"]}
        return {"ok": False, "phase_keyframes": {}, "debug": d}

    kfc = list(kin.get("kinematic_fail_codes") or [])
    if kfc:
        d = {
            **empty_debug,
            "final_phase_semantic_fail_reasons": [str(kfc[0]) if kfc else "KINEMATIC_FAIL"],
        }
        return {"ok": False, "phase_keyframes": {}, "debug": d}

    valid = kin["valid"]
    excursion = kin["excursion"]
    speed = kin["speed_s"]
    setup = int(kin["setup"])
    min_gap = max(2, n // 28, int(round(float(fps) * 0.038)))

    ev = detect_phase_events_agnostic(poses)
    exc_apex = int(ev.get("excursion_apex_idx", max(1, n // 4)))
    top_ev = int(ev["top_pose_idx"])
    imp_ev = int(ev["impact_pose_idx"])
    top_dbg = ev.get("top_candidate_debug") or {}
    imp_dbg = ev.get("impact_candidate_debug") or {}
    w0, w1 = top_dbg.get("window") or [min(max(setup, 1), n - 4), min(n - 2, int(n * 0.78))]
    w0, w1 = int(w0), int(w1)
    w1 = min(max(w0 + 2, w1), n - 2)
    ilo, ihi = imp_dbg.get("window") or [max(exc_apex + 2, int(n * 0.42)), min(n - 2, int(n * 0.93))]
    ilo, ihi = int(ilo), int(ihi)
    if ilo >= ihi:
        ilo, ihi = max(1, n // 2), min(n - 2, n - 1)

    phase_windows: dict[str, list[int]] = {
        "top": [w0 + 1, w1 - 1],
        "impact": [ilo, ihi - 1],
    }

    # ── Top: prefer semantic-passing candidates, ranked by composite score ──
    top_scored: list[tuple[float, int, bool]] = []
    for i in range(w0 + 1, w1):
        if not bool(valid[i]):
            continue
        sc = float(_composite_top_candidate_score(i, kin, w0, w1))
        if sc < 0:
            continue
        ok_t, _ = validate_top_semantic_at_index(i, kin)
        top_scored.append((sc, i, ok_t))
    top_scored.sort(key=lambda t: (t[2], t[0], -abs(t[1] - top_ev)), reverse=True)
    top_i = top_scored[0][1] if top_scored else top_ev
    top_pick_score = float(top_scored[0][0]) if top_scored else 0.0
    top_semantic_ok = bool(top_scored[0][2]) if top_scored else False
    if not top_semantic_ok and top_scored:
        for sc, i, ok_t in top_scored:
            if ok_t:
                top_i = i
                top_pick_score = float(sc)
                top_semantic_ok = True
                break

    # ── Impact: rank by composite impact score, prefer semantic pass ──
    lo_imp = max(exc_apex + 2, top_i + min_gap, ilo)
    hi_imp = min(n - 2, ihi)
    if lo_imp >= hi_imp:
        lo_imp, hi_imp = top_i + min_gap, min(n - 2, n - 1)
    phase_windows["impact"] = [lo_imp, hi_imp]
    imp_scored: list[tuple[float, int, bool]] = []
    for i in range(lo_imp, hi_imp + 1):
        if not bool(valid[i]):
            continue
        sc = float(
            _composite_impact_candidate_score(
                i, top_i, exc_apex, kin, lo_imp, hi_imp + 1, n, top_i,
            )
        )
        if sc < 0:
            continue
        ok_i, _ = validate_impact_semantic_at_index(i, top_i, exc_apex, kin)
        imp_scored.append((sc, i, ok_i))
    imp_scored.sort(key=lambda t: (t[2], t[0], -abs(t[1] - imp_ev)), reverse=True)
    imp_i = imp_scored[0][1] if imp_scored else imp_ev
    imp_pick_score = float(imp_scored[0][0]) if imp_scored else 0.0
    impact_semantic_ok = bool(imp_scored[0][2]) if imp_scored else False
    if not impact_semantic_ok and imp_scored:
        for sc, i, ok_i in imp_scored:
            if ok_i:
                imp_i = i
                imp_pick_score = float(sc)
                impact_semantic_ok = True
                break

    if imp_i <= top_i + 1:
        imp_i = min(max(top_i + min_gap, imp_ev), n - 2)

    # ── Address: first quality pose in early setup band ──
    addr_hi = min(max(setup + 3, 4), top_i - min_gap * 2, n - 6)
    addr_hi = max(addr_hi, 0)
    address_i = 0
    for i in range(0, addr_hi + 1):
        if bool(valid[i]):
            address_i = i
            break
    phase_windows["address"] = [0, addr_hi]

    # ── Takeaway / backswing: excursion milestones (not uniform time) ──
    a, t = address_i, top_i
    if t <= a + 3:
        takeaway_i = min(a + 1, n - 7)
        backswing_i = min(a + 2, n - 6)
    else:
        seg = np.asarray(excursion[a : t + 1], dtype=np.float64)
        seg_v = np.asarray(valid[a : t + 1], dtype=bool)
        if not np.any(seg_v):
            takeaway_i = a + max(1, (t - a) // 3)
            backswing_i = a + max(2, (2 * (t - a)) // 3)
        else:
            e0 = float(np.percentile(seg[seg_v], 15))
            e1 = float(np.percentile(seg[seg_v], 88))
            span = max(e1 - e0, 1e-6)

            def _first_exc_ratio(ratio: float) -> int:
                target = e0 + ratio * span
                best_j = a + 1
                best_d = 1e9
                for j in range(a + 1, t):
                    if not bool(valid[j]):
                        continue
                    d = abs(float(excursion[j]) - target)
                    if d < best_d:
                        best_d = d
                        best_j = j
                return int(min(max(best_j, a + 1), t - 1))

            takeaway_i = _first_exc_ratio(0.28)
            backswing_i = _first_exc_ratio(0.62)
            if backswing_i <= takeaway_i:
                backswing_i = min(takeaway_i + 1, t - 1)
    phase_windows["takeaway"] = [address_i + 1, top_i - 2]
    phase_windows["backswing"] = [takeaway_i + 1, top_i - 1]

    tw = int(min(max(takeaway_i, address_i + 1), top_i - 2))
    bw = int(min(max(backswing_i, tw + 1), top_i - 1))
    partial = {
        "address": int(address_i),
        "takeaway": tw,
        "backswing": bw,
        "top": int(top_i),
    }

    chain = propose_post_impact_chain_indices(poses, dict(partial))
    if not chain:
        # Relaxed joint search (same semantics as propose, slightly wider min_gap)
        min_g2 = max(2, min_gap - 1)
        best_c: dict[str, int] | None = None
        lo = max(top_i + 1, int(n * 0.44))
        hi = min(n - 3, int(n * 0.96))
        _found_chain = False
        for ds in range(lo, min(hi - 3 * min_g2, imp_i + 6)):
            if _found_chain:
                break
            if not valid[ds]:
                continue
            for im in range(max(ds + min_g2, top_i + 2), min(hi, imp_i + 10)):
                if _found_chain:
                    break
                if not valid[im]:
                    continue
                if not validate_impact_semantic_at_index(im, top_i, exc_apex, kin)[0]:
                    continue
                for ft in range(im + min_g2, min(n - 2, im + max(10, n // 4))):
                    if _found_chain:
                        break
                    if not valid[ft]:
                        continue
                    if not validate_follow_through_semantic_at_index(ft, im, kin)[0]:
                        continue
                    for fn in range(ft + min_g2, n):
                        if not valid[fn]:
                            continue
                        if not validate_finish_semantic_at_index(fn, ft, im, kin)[0]:
                            continue
                        best_c = {
                            "downswing": int(ds),
                            "impact": int(im),
                            "follow_through": int(ft),
                            "finish": int(fn),
                        }
                        _found_chain = True
                        break
        chain = best_c or {}

    if not chain:
        d = {
            "phase_reselect_strategy": "semantic_oriented_v1",
            "phase_candidate_window": phase_windows,
            "phase_candidate_scores": {
                "top": [{"idx": x[1], "composite": round(x[0], 4), "semantic_ok": x[2]} for x in top_scored[:5]],
                "impact": [{"idx": x[1], "composite": round(x[0], 4), "semantic_ok": x[2]} for x in imp_scored[:5]],
            },
            "top_semantic_score": round(top_pick_score, 4),
            "impact_semantic_score": round(imp_pick_score, 4),
            "finish_semantic_score": None,
            "final_phase_semantic_pass": False,
            "final_phase_semantic_fail_reasons": ["POST_TOP_CHAIN_UNAVAILABLE"],
            "source_pose_idx_list": [],
            "min_gap_frames": min_gap,
            "duplicate_pairs": [],
        }
        return {"ok": False, "phase_keyframes": {}, "debug": d}

    full: dict[str, int] = {**partial, **{k: int(v) for k, v in chain.items()}}
    full = _squeeze_phase_pose_indices_monotonic(full, n)
    full = _enforce_increasing_source_frames_on_phase_map(poses, full, fps=float(fps))

    spi_list = [int(full[p]) for p in PHASE_ORDER]
    fail_reasons: list[str] = []
    t_ok, _ = validate_top_semantic_at_index(int(full["top"]), kin)
    if not t_ok:
        fail_reasons.append("TOP_SEMANTIC_FAIL")
    i_ok, _ = validate_impact_semantic_at_index(
        int(full["impact"]), int(full["top"]), exc_apex, kin,
    )
    if not i_ok:
        fail_reasons.append("IMPACT_SEMANTIC_FAIL")
    ft_ok, _ = validate_follow_through_semantic_at_index(
        int(full["follow_through"]), int(full["impact"]), kin,
    )
    if not ft_ok:
        fail_reasons.append("FOLLOW_THROUGH_SEMANTIC_FAIL")
    fin_ok, fin_det = validate_finish_semantic_at_index(
        int(full["finish"]),
        int(full["follow_through"]),
        int(full["impact"]),
        kin,
    )
    fin_score = None
    if fin_ok and int(full["finish"]) < len(speed):
        fin_score = round(
            float(1.0 - min(float(speed[int(full["finish"])]) / max(float(speed[int(full["impact"])]), 1e-6), 1.2)),
            4,
        )
    if not fin_ok:
        fail_reasons.append("FINISH_SEMANTIC_FAIL")

    sem_pass = bool(t_ok and i_ok and ft_ok and fin_ok)
    phase_windows["downswing"] = [int(full["top"]) + 1, max(int(full["top"]) + 2, int(full["impact"]) - 1)]
    phase_windows["follow_through"] = [int(full["impact"]) + 1, max(int(full["impact"]) + 2, int(full["finish"]) - 1)]
    phase_windows["finish"] = [int(full["follow_through"]) + 1, n - 1]
    dbg = {
        "phase_reselect_strategy": "semantic_oriented_v1",
        "phase_candidate_window": phase_windows,
        "phase_candidate_scores": {
            "top": [{"idx": x[1], "composite": round(x[0], 4), "semantic_ok": x[2]} for x in top_scored[:5]],
            "impact": [{"idx": x[1], "composite": round(x[0], 4), "semantic_ok": x[2]} for x in imp_scored[:5]],
        },
        "top_semantic_score": round(top_pick_score, 4),
        "impact_semantic_score": round(imp_pick_score, 4),
        "finish_semantic_score": fin_score,
        "final_phase_semantic_pass": sem_pass,
        "final_phase_semantic_fail_reasons": list(fail_reasons),
        "source_pose_idx_list": spi_list,
        "min_gap_frames": min_gap,
        "duplicate_pairs": [],
    }
    prev = -1
    ok_mono = True
    for p in PHASE_ORDER:
        if int(full[p]) <= prev:
            ok_mono = False
        prev = int(full[p])
    if not ok_mono:
        dbg["final_phase_semantic_fail_reasons"] = list(
            set(dbg["final_phase_semantic_fail_reasons"]) | {"MONOTONIC_FAIL_AFTER_SQUEEZE"},
        )
        return {"ok": False, "phase_keyframes": {}, "debug": dbg}

    return {"ok": True, "phase_keyframes": full, "debug": dbg}


def apply_semantic_oriented_keyframe_recovery(
    video_path: str,
    poses: list[dict],
    keyframes: list[dict],
    kf_validation: dict,
    phase_keyframes: dict[str, int],
    keyframe_width: int = 320,
    *,
    prev_fv_keys: dict | None = None,
) -> tuple[list[dict], dict, dict[str, int], bool]:
    """One-shot semantic-oriented strip rebuild + rebind + strict validate (for Plus pre-uniform path)."""
    vcap = cv2.VideoCapture(video_path)
    vid_fps = float(vcap.get(cv2.CAP_PROP_FPS) or 30.0)
    vcap.release()

    bundle = build_semantic_oriented_phase_map(
        poses, dict(phase_keyframes), fps=float(vid_fps),
    )
    dbg = dict(bundle.get("debug") or {})
    base_kv = dict(kf_validation)

    if not bool(bundle.get("ok")):
        merged_fail = {
            **base_kv,
            "phase_oriented_semantic_debug": dbg,
            "semantic_oriented_recovery_attempted": True,
            "final_phase_semantic_pass": bool(dbg.get("final_phase_semantic_pass")),
            "final_phase_semantic_fail_reasons": list(dbg.get("final_phase_semantic_fail_reasons") or []),
        }
        fkv0 = dict(base_kv.get("final_keyframe_validation") or {})
        fkv0["phase_oriented_semantic_debug"] = dbg
        merged_fail["final_keyframe_validation"] = fkv0
        return keyframes, merged_fail, dict(phase_keyframes), False

    sm = dict(bundle["phase_keyframes"])
    cap = cv2.VideoCapture(video_path)
    try:
        sem_kf, sem_det = _rebind_keyframes_from_rebuilt_map(
            cap,
            get_video_rotation(video_path),
            float(vid_fps),
            poses,
            sm,
            keyframe_width,
        )
    finally:
        cap.release()

    sem_gate = validate_final_keyframes_for_ai(
        sem_kf, sm, sem_det, poses=poses, fps=vid_fps,
    )
    prev_fv = prev_fv_keys or {}
    for _pk in _REPAIR_FV_PRESERVE_KEYS:
        if _pk in prev_fv:
            sem_gate[_pk] = prev_fv[_pk]

    dup_pairs = _adjacent_phase_keyframe_hard_dup_reasons(sem_kf, poses) if poses else []
    dbg["duplicate_pairs"] = list(dup_pairs)

    sem_gate = dict(sem_gate)
    sem_gate["phase_oriented_semantic_debug"] = dbg
    sem_gate["duplicate_pairs"] = list(dup_pairs)

    ok = bool(sem_gate.get("pass"))
    merged_ok = {
        **base_kv,
        "details": sem_det,
        "final_phase_keyframes": dict(sm),
        "final_keyframe_validation": sem_gate,
        "final_keyframe_order_ok": bool(sem_gate.get("final_keyframe_order_ok")),
        "final_keyframe_time_order_ok": bool(sem_gate.get("final_keyframe_time_order_ok")),
        "final_phase_keyframes_sync_ok": bool(sem_gate.get("final_phase_keyframes_sync_ok")),
        "final_keyframe_gate_pass": ok,
        "all_passed": ok,
        "final_validation_failed": not ok,
        "final_keyframe_source": "semantic_oriented_repaired" if ok else base_kv.get("final_keyframe_source", "smart_gate_failed"),
        "repair_state": None if ok else "semantic_oriented_attempted",
        "phase_oriented_semantic_debug": dbg,
        "semantic_oriented_recovery_attempted": True,
        "final_phase_semantic_pass": bool(dbg.get("final_phase_semantic_pass")),
        "final_phase_semantic_fail_reasons": list(dbg.get("final_phase_semantic_fail_reasons") or []),
    }

    logger.info(
        "[keyframe] apply_semantic_oriented recovery gate_pass=%s final_phase_semantic_pass=%s "
        "phase_reselect_strategy=%s source_pose_idx_list=%s min_gap_frames=%s "
        "top_semantic_score=%s impact_semantic_score=%s finish_semantic_score=%s "
        "final_phase_semantic_fail_reasons=%s duplicate_pairs=%s",
        ok,
        dbg.get("final_phase_semantic_pass"),
        dbg.get("phase_reselect_strategy"),
        dbg.get("source_pose_idx_list"),
        dbg.get("min_gap_frames"),
        dbg.get("top_semantic_score"),
        dbg.get("impact_semantic_score"),
        dbg.get("finish_semantic_score"),
        dbg.get("final_phase_semantic_fail_reasons"),
        dbg.get("duplicate_pairs"),
    )
    if not ok:
        merged_fail2 = {
            **base_kv,
            "phase_oriented_semantic_debug": dbg,
            "semantic_oriented_recovery_attempted": True,
            "final_phase_semantic_pass": bool(dbg.get("final_phase_semantic_pass")),
            "final_phase_semantic_fail_reasons": list(dbg.get("final_phase_semantic_fail_reasons") or []),
            "final_keyframe_validation": {
                **dict(base_kv.get("final_keyframe_validation") or {}),
                "semantic_oriented_last_gate": sem_gate,
                "phase_oriented_semantic_debug": dbg,
                "duplicate_pairs": list(dup_pairs),
            },
        }
        return keyframes, merged_fail2, dict(phase_keyframes), False

    return sem_kf, merged_ok, sm, True


def _rebind_keyframes_from_rebuilt_map(
    cap: Any,
    rotation: Any,
    fps: float,
    poses: list[dict],
    phase_keyframes: dict[str, int],
    keyframe_width: int,
) -> tuple[list[dict], list[dict]]:
    """Rebuild the 8 keyframe dicts in ``PHASE_ORDER`` from anchor-resolved pose indices (fresh decode)."""
    total_duration = max(
        float(poses[-1].get("timestamp", 0)) - float(poses[0].get("timestamp", 0)),
        0.1,
    ) if poses else 0.1
    min_time_gap = max(total_duration * _MIN_TIME_INTERVAL_RATIO, 0.05)

    keyframes: list[dict] = []
    details: list[dict] = []
    prev_ts = -999.0
    prev_pose_idx = -1
    all_frames: list[np.ndarray] = []
    all_poses: list[dict] = []

    for phase_id in PHASE_ORDER:
        meta = SWING_PHASE_META[phase_id]
        spi = int(phase_keyframes[phase_id])
        pose = poses[spi]
        pfi = int(pose.get("frame_index", spi))
        frame = _read_frame_pose_matched(cap, pfi, rotation)
        if frame is None:
            frame = _read_frame_with_decode_fallback(cap, pfi, rotation)
        if frame is None:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
        pq = _pose_quality_details(pose)
        iq = _frame_quality_details(frame)
        visual_diff = 1.0
        is_near_duplicate = False
        if all_frames:
            worst_similarity = _worst_pose_gated_histogram_similarity(
                frame, pose, all_frames, all_poses,
            )
            visual_diff = round(1.0 - worst_similarity, 4)
            is_near_duplicate = worst_similarity > _VISUAL_DIFF_THRESHOLD
        ts = float(pose.get("timestamp", round(pfi / max(fps, 1e-6), 3)))
        time_gap = ts - prev_ts if prev_ts >= 0 else 999.0
        time_too_close = time_gap < min_time_gap and prev_ts >= 0
        conf = float(np.clip(0.35 * pq["quality"] + 0.25 * iq["quality"] + 0.35, 0.0, 1.0))
        validation_passed = not is_near_duplicate and not time_too_close
        details.append({
            "phase": phase_id,
            "visual_diff_from_prev": visual_diff,
            "is_near_duplicate": is_near_duplicate,
            "time_gap": round(time_gap, 3),
            "time_too_close": time_too_close,
            "validation_passed": validation_passed,
            "anchor_rebind": True,
        })
        resized = _resize_frame(frame, keyframe_width)
        pose_snapshot = _pose_snapshot_for_keyframe(pose)
        keyframes.append({
            "phase": phase_id,
            "label_en": meta["label_en"],
            "label_zh": meta["label_zh"],
            "frame_index": pfi,
            "timestamp": round(ts, 3),
            "confidence": round(conf, 3),
            "selection_reason": "event_anchor_rebuild",
            "fallback_used": True,
            "image_base64": frame_to_base64(resized),
            "width": resized.shape[1],
            "height": resized.shape[0],
            "pose_snapshot": pose_snapshot,
            "source_pose_idx": spi,
            "source_frame_index": pfi,
            "visual_diff_from_prev": visual_diff,
            "phase_validation_passed": validation_passed,
        })
        prev_ts = ts
        prev_pose_idx = spi
        all_frames.append(frame)
        all_poses.append(pose)

    return keyframes, details


def _merge_keyframe_validation_with_repairs(
    fv: dict[str, Any],
    repair_extra: dict[str, Any],
) -> dict[str, Any]:
    if not repair_extra:
        return fv
    out = dict(fv)
    out.update(repair_extra)
    sr = list(out.get("semantic_strip_reasons") or [])
    for r in repair_extra.get("enforce_fail_reasons") or []:
        if r and r not in sr:
            sr.append(r)
    rc = int(out.get("relabel_count") or 0)
    out["relabel_count"] = rc
    if rc > 1:
        out["semantic_strip_ok"] = False
        if "PHASE_STRIP_REPAIR_FAILED_RELABEL_FORBIDDEN" not in sr:
            sr.append("PHASE_STRIP_REPAIR_FAILED_RELABEL_FORBIDDEN")
    if not repair_extra.get("enforce_ok", True):
        out["semantic_strip_ok"] = False
    out["semantic_strip_reasons"] = sr
    out["pass"] = bool(
        out.get("strict_contract_ok")
        and bool(out.get("semantic_strip_ok"))
        and rc <= 1
    )
    if not out["pass"]:
        # Failed rerun cannot keep "reselected success" markers.
        out["reselected_top"] = False
        out["reselected_impact"] = False
        out["top_reselected"] = False
        out["impact_reselected"] = False
    return out


def enforce_top_impact_semantic(
    keyframes: list[dict],
    phase_keyframes: dict[str, int],
    poses: list[dict],
    *,
    min_gap: int = 3,
    window: int = 8,
) -> dict[str, Any]:
    """Repair top/impact pose indices using kinematic events + semantic validators."""
    from services.swing_flow_utils import (
        _build_view_agnostic_kinematics,
        detect_phase_events_agnostic,
        validate_impact_semantic_at_index,
        validate_top_semantic_at_index,
    )

    log: list[str] = []
    reselected_top = False
    reselected_impact = False
    reasons: list[str] = []
    if len(keyframes) != 8 or not poses:
        return {
            "ok": False,
            "log": log,
            "reselected_top": False,
            "reselected_impact": False,
            "reasons": ["BAD_INPUT"],
        }

    kin = _build_view_agnostic_kinematics(poses)
    if kin is None:
        return {
            "ok": False,
            "log": log,
            "reselected_top": False,
            "reselected_impact": False,
            "reasons": ["KINEMATICS_UNAVAILABLE"],
        }

    ev = detect_phase_events_agnostic(poses)
    top_ev = int(ev["top_pose_idx"])
    imp_ev = int(ev["impact_pose_idx"])
    exc_apex = int(ev["excursion_apex_idx"])
    n = len(poses)

    def apply_spi(ph: str, spi: int) -> None:
        phase_keyframes[ph] = spi
        for kk in keyframes:
            if kk.get("phase") == ph:
                kk["source_pose_idx"] = spi
                if 0 <= spi < len(poses):
                    p = poses[spi]
                    kk["frame_index"] = int(p.get("frame_index", spi))
                    kk["timestamp"] = round(float(p.get("timestamp", 0.0)), 3)
                    kk["source_frame_index"] = kk["frame_index"]
                break

    top_pi = int(phase_keyframes.get("top", -1))
    imp_pi = int(phase_keyframes.get("impact", -1))
    if top_pi < 0 or imp_pi < 0:
        return {
            "ok": False,
            "log": log,
            "reselected_top": False,
            "reselected_impact": False,
            "reasons": ["TOP_OR_IMPACT_MISSING"],
        }

    def gap_violation() -> bool:
        return imp_pi <= top_pi + min_gap

    if gap_violation():
        log.append("repair_impact_min_gap")
        lo = max(0, top_pi + min_gap + 1, imp_ev - window)
        hi = min(n - 1, imp_ev + window)
        best_i: int | None = None
        best_key: tuple | None = None
        for i in range(lo, hi + 1):
            if not bool(kin["valid"][i]):
                continue
            ok, _ = validate_impact_semantic_at_index(i, top_pi, exc_apex, kin)
            if not ok:
                continue
            key = (abs(i - imp_ev), -float(kin["q"][i]))
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        if best_i is None:
            reasons.append("IMPACT_REPAIR_FAILED")
            return {
                "ok": False,
                "log": log,
                "reselected_top": reselected_top,
                "reselected_impact": reselected_impact,
                "reasons": reasons,
            }
        apply_spi("impact", best_i)
        imp_pi = best_i
        reselected_impact = True
        log.append(f"impact_moved_to_{best_i}")

    top_ok, _ = validate_top_semantic_at_index(top_pi, kin)
    if not top_ok:
        log.append("repair_top_semantic")
        lo = max(1, top_ev - window)
        hi = min(n - 2, top_ev + window)
        best_i = None
        best_key = None
        for i in range(lo, hi + 1):
            if not bool(kin["valid"][i]):
                continue
            ok, _ = validate_top_semantic_at_index(i, kin)
            if not ok:
                continue
            if i >= imp_pi - min_gap:
                continue
            key = (abs(i - top_ev), -float(kin["q"][i]))
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        if best_i is None:
            reasons.append("TOP_REPAIR_FAILED")
            return {
                "ok": False,
                "log": log,
                "reselected_top": reselected_top,
                "reselected_impact": reselected_impact,
                "reasons": reasons,
            }
        apply_spi("top", best_i)
        top_pi = best_i
        reselected_top = True
        log.append(f"top_moved_to_{best_i}")

    if gap_violation():
        log.append("repair_impact_after_top_move")
        lo = max(0, top_pi + min_gap + 1, imp_ev - window)
        hi = min(n - 1, imp_ev + window)
        best_i = None
        best_key = None
        for i in range(lo, hi + 1):
            if not bool(kin["valid"][i]):
                continue
            ok, _ = validate_impact_semantic_at_index(i, top_pi, exc_apex, kin)
            if not ok:
                continue
            key = (abs(i - imp_ev), -float(kin["q"][i]))
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        if best_i is None:
            reasons.append("IMPACT_REPAIR_FAILED_AFTER_TOP")
            return {
                "ok": False,
                "log": log,
                "reselected_top": reselected_top,
                "reselected_impact": reselected_impact,
                "reasons": reasons,
            }
        apply_spi("impact", best_i)
        imp_pi = best_i
        reselected_impact = True
        log.append(f"impact_moved_to_{best_i}_after_top")

    top_ok, _ = validate_top_semantic_at_index(top_pi, kin)
    imp_ok, _ = validate_impact_semantic_at_index(imp_pi, top_pi, exc_apex, kin)
    if not top_ok:
        reasons.append("TOP_SEMANTIC_STILL_FAIL")
    if not imp_ok:
        reasons.append("IMPACT_SEMANTIC_STILL_FAIL")
    if not top_ok or not imp_ok:
        return {
            "ok": False,
            "log": log,
            "reselected_top": reselected_top,
            "reselected_impact": reselected_impact,
            "reasons": reasons,
        }

    return {
        "ok": True,
        "log": log,
        "reselected_top": reselected_top,
        "reselected_impact": reselected_impact,
        "reasons": [],
    }


def verify_phase_strip_semantics(
    keyframes: list[dict] | None,
    poses: list[dict],
    phase_keyframes: dict[str, int],
) -> dict:
    """Semantic contract for the 8-phase strip (labels, top/impact events, post-impact deceleration)."""
    from services.swing_flow_utils import (
        _build_view_agnostic_kinematics,
        detect_phase_events_agnostic,
        validate_impact_semantic_at_index,
        validate_top_semantic_at_index,
    )

    reasons: list[str] = []
    top_semantic_ok: bool = False
    impact_semantic_ok: bool = False
    expected = tuple(PHASE_ORDER)

    if not poses:
        return {
            "pass": False,
            "reasons": ["NO_POSES_FOR_PHASE_STRIP_SEMANTICS"],
            "top_semantic_ok": False,
            "impact_semantic_ok": False,
        }

    pk: dict[str, int] = dict(phase_keyframes)
    if keyframes and len(keyframes) == 8:
        sorted_by_pose = sorted(
            keyframes,
            key=lambda k: int(k["source_pose_idx"]) if k.get("source_pose_idx") is not None else -(10**9),
        )
        got_sorted = tuple(kf.get("phase") for kf in sorted_by_pose)
        if got_sorted != expected:
            reasons.append("PHASE_STRIP_PHASE_ORDER_MISMATCH_AFTER_POSE_SORT")
        got = tuple(kf.get("phase") for kf in keyframes)
        if got != expected:
            reasons.append("PHASE_STRIP_LABEL_ORDER_MISMATCH")
        for kf in keyframes:
            pid = kf.get("phase")
            spi = kf.get("source_pose_idx")
            if pid and spi is not None:
                pk[str(pid)] = int(spi)
    elif keyframes is not None and len(keyframes) != 8:
        reasons.append("KEYFRAME_COUNT_NOT_8_FOR_SEMANTIC_STRIP")

    prev = -1
    for pid in PHASE_ORDER:
        ix = pk.get(pid)
        if ix is None:
            reasons.append(f"MISSING_PHASE_MAP:{pid}")
            continue
        ix = int(ix)
        if ix <= prev:
            reasons.append("PHASE_POSE_INDEX_NOT_INCREASING")
        prev = ix

    top_raw = pk.get("top")
    imp_raw = pk.get("impact")
    top_i: int | None = int(top_raw) if top_raw is not None else None
    imp_i: int | None = int(imp_raw) if imp_raw is not None else None
    if top_i is None or imp_i is None:
        reasons.append("TOP_OR_IMPACT_MISSING_IN_MAP")
    elif imp_i - top_i < 3:
        reasons.append("TOP_IMPACT_MIN_FRAMES_FAIL")

    kin = _build_view_agnostic_kinematics(poses)
    n = len(poses)
    if kin is None:
        reasons.append("KINEMATICS_UNAVAILABLE")
        return {
            "pass": False,
            "reasons": reasons,
            "top_semantic_ok": False,
            "impact_semantic_ok": False,
        }

    td = kin.get("time_axis_debug") or {}
    if td.get("dt_axis_invalid") and "DT_AXIS_INVALID" not in reasons:
        reasons.append("DT_AXIS_INVALID")
    for c in kin.get("kinematic_fail_codes") or []:
        if c == "NON_FINITE_KINEMATICS" and "NON_FINITE_KINEMATICS" not in reasons:
            reasons.append("NON_FINITE_KINEMATICS")

    ev = detect_phase_events_agnostic(poses)
    exc_ctx = int(ev.get("excursion_apex_idx", max(1, n // 4)))

    if top_i is not None and 0 <= top_i < n:
        top_semantic_ok, _ = validate_top_semantic_at_index(top_i, kin)
        if not top_semantic_ok:
            reasons.append("TOP_SEMANTIC_AT_KEYFRAME_FAIL")
    else:
        reasons.append("TOP_INDEX_INVALID")

    if (
        imp_i is not None
        and top_i is not None
        and 0 <= imp_i < n
    ):
        impact_semantic_ok, _ = validate_impact_semantic_at_index(imp_i, top_i, exc_ctx, kin)
        if not impact_semantic_ok:
            reasons.append("IMPACT_SEMANTIC_AT_KEYFRAME_FAIL")
    else:
        impact_semantic_ok = False
        if imp_i is None or not (0 <= int(imp_i) < n):
            reasons.append("IMPACT_INDEX_INVALID")

    speed = np.asarray(kin["speed_s"], dtype=np.float64)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    valid = kin["valid"]
    med_imp = 0.0
    impact_window_ok = False
    if imp_i is not None and 0 <= imp_i < n:
        lo_w = max(0, imp_i - 1)
        hi_w = min(n, imp_i + 2)
        slice_w = speed[lo_w:hi_w]
        mask_w = valid[lo_w:hi_w]
        vals_imp = [
            float(slice_w[j])
            for j in range(len(slice_w))
            if j < len(mask_w) and bool(mask_w[j])
        ]
        if not vals_imp and len(slice_w) > 0:
            vals_imp = [float(slice_w[j]) for j in range(len(slice_w))]
        if vals_imp:
            med_imp = float(np.median(np.asarray(vals_imp, dtype=np.float64)))
            impact_window_ok = True
        if not math.isfinite(med_imp):
            med_imp = 0.0
    if not impact_window_ok:
        reasons.append("IMPACT_WINDOW_UNAVAILABLE")

    post_vals: list[float] = []
    for pid in ("follow_through", "finish"):
        ix = pk.get(pid)
        if ix is not None:
            ix = int(ix)
        if isinstance(ix, int) and 0 <= ix < n:
            v = float(speed[ix])
            if math.isfinite(v):
                post_vals.append(v)
    post_med = float(np.median(np.asarray(post_vals, dtype=np.float64))) if post_vals else 0.0
    if not math.isfinite(post_med):
        post_med = 0.0

    eps = 1e-5
    # Hands can remain fast into follow-through; require decay vs impact-window median with headroom.
    post_relax = 1.22
    if len(post_vals) == 0:
        reasons.append("POST_IMPACT_PHASES_SPEED_UNAVAILABLE")
    elif impact_window_ok and med_imp > eps and post_med >= med_imp * post_relax - eps:
        reasons.append("POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN")

    pass_ok = len(reasons) == 0
    return {
        "pass": bool(pass_ok),
        "reasons": reasons,
        "top_semantic_ok": bool(top_semantic_ok),
        "impact_semantic_ok": bool(impact_semantic_ok),
    }


def _nearest_pose_idx_for_video_frame(poses: list[dict], target_fi: int) -> int:
    best_i = 0
    best_d = 10**9
    tf = int(target_fi)
    for i, p in enumerate(poses):
        d = abs(int(p.get("frame_index", 0)) - tf)
        if d < best_d:
            best_d, best_i = d, i
    return int(best_i)


def _source_frame_index_phase_gaps_ok(keyframes: list[dict], fps: float) -> tuple[bool, list[str]]:
    """Enforce monotonic raw frame indices and fps-adaptive min gaps between consecutive phases."""
    by_phase = {str(kf.get("phase")): kf for kf in keyframes}
    reasons: list[str] = []
    if len(by_phase) != 8:
        return False, ["MIN_GAP_VIOLATION:INCOMPLETE_STRIP"]

    def _fi(ph: str) -> int:
        kf = by_phase[ph]
        return int(kf.get("source_frame_index", kf.get("frame_index", 0)))

    for i in range(len(PHASE_ORDER) - 1):
        a, b = PHASE_ORDER[i], PHASE_ORDER[i + 1]
        gap_need = max(1, int(round(fps * 0.02)))
        if (a, b) in (
            ("takeaway", "backswing"),
            ("backswing", "top"),
            ("top", "downswing"),
            ("downswing", "impact"),
            ("impact", "follow_through"),
        ):
            gap_need = max(gap_need, max(3, int(round(fps * 0.06))))
        if (a, b) == ("top", "impact"):
            gap_need = max(gap_need, max(4, int(round(fps * 0.12))))
        elif (a, b) == ("impact", "follow_through"):
            gap_need = max(gap_need, max(3, int(round(fps * 0.10))))
        elif (a, b) == ("follow_through", "finish"):
            gap_need = max(gap_need, max(3, int(round(fps * 0.10))))
        if _fi(b) - _fi(a) < gap_need:
            reasons.append(f"MIN_GAP_VIOLATION:{a.upper()}_{b.upper()}")
        if _fi(b) <= _fi(a):
            reasons.append("MIN_GAP_VIOLATION:NON_MONOTONIC_RAW_FRAMES")
            break
    return len(reasons) == 0, reasons


def _adjacent_phase_keyframe_hard_dup_reasons(keyframes: list[dict], poses: list[dict]) -> list[str]:
    """Consecutive strip phases must not share the same raw frame or near-identical pose."""
    out: list[str] = []
    for i in range(len(keyframes) - 1):
        a, b = keyframes[i], keyframes[i + 1]
        fi_a = int(a.get("source_frame_index", a.get("frame_index", -1)))
        fi_b = int(b.get("source_frame_index", b.get("frame_index", -2)))
        if fi_a == fi_b:
            out.append(f"ADJACENT_STRIP_SAME_RAW_FRAME:{a.get('phase')}->{b.get('phase')}")
            continue
        spi_a = int(a.get("source_pose_idx", -1))
        spi_b = int(b.get("source_pose_idx", -2))
        if 0 <= spi_a < len(poses) and 0 <= spi_b < len(poses):
            if abs(fi_b - fi_a) <= 2 and _pose_angle_distance(poses[spi_a], poses[spi_b]) < 0.12:
                out.append(f"ADJACENT_STRIP_NEAR_DUPLICATE:{a.get('phase')}->{b.get('phase')}")
    return out


def refine_phase_keyframe_on_raw_timeline(
    cap: cv2.VideoCapture,
    rotation: int,
    fps: float,
    poses: list[dict],
    phase_keyframes: dict[str, int],
    phase_id: str,
    candidate_pose_idx: int,
    *,
    search_radius_frames: int | None,
    total_video_frames: int,
    impact_anchor_fi: int | None = None,
    follow_through_anchor_fi: int | None = None,
) -> dict[str, Any]:
    """Local search on raw video frame axis; map to nearest pose for kinematic scoring."""
    from services.swing_flow_utils import (
        _build_view_agnostic_kinematics,
        detect_phase_events_agnostic,
        validate_impact_semantic_at_index,
        validate_top_semantic_at_index,
    )

    debug: dict[str, Any] = {"phase": phase_id, "scans": 0}
    cpi = int(candidate_pose_idx)
    if not poses or phase_id not in ("takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"):
        fi0 = int(poses[cpi].get("frame_index", 0)) if 0 <= cpi < len(poses) else 0
        return {
            "best_source_pose_idx": cpi,
            "best_source_frame_index": fi0,
            "refine_debug": {**debug, "skipped": "unsupported_or_empty"},
        }

    kin = _build_view_agnostic_kinematics(poses)
    ev = detect_phase_events_agnostic(poses) if kin is not None else None
    exc_ix = int(ev.get("excursion_apex_idx", 0)) if ev else 0
    top_ev = int(ev.get("top_pose_idx", 0)) if ev else 0
    imp_ev = int(ev.get("impact_pose_idx", len(poses) // 2)) if ev else len(poses) // 2

    base_rad = max(6, int(round(fps * 0.2)))
    if search_radius_frames is not None:
        rad = int(search_radius_frames)
    elif phase_id == "top":
        rad = max(base_rad, max(8, int(round(fps * 0.28))))
    elif phase_id == "impact":
        rad = max(base_rad, max(10, int(round(fps * 0.22))))
    else:
        rad = base_rad
    center_fi = int(poses[cpi].get("frame_index", 0))
    lo = max(0, center_fi - rad)
    hi = min(max(0, total_video_frames - 1), center_fi + rad)

    impact_min_fi_from_top: int | None = None
    if phase_id == "impact":
        top_pi = int(phase_keyframes.get("top", cpi))
        if 0 <= top_pi < len(poses):
            top_fi = int(poses[top_pi].get("frame_index", 0))
            impact_min_fi_from_top = top_fi + max(4, int(round(fps * 0.12)))

    speed_mx = 1.0
    if kin is not None and len(kin["speed_s"]) > 0:
        speed_mx = float(np.max(kin["speed_s"])) + 1e-6

    best_pi = cpi
    best_fi = center_fi
    best_sc = -1e18

    follow_min_fi: int | None = None
    if phase_id == "follow_through" and impact_anchor_fi is not None:
        follow_min_fi = int(impact_anchor_fi) + max(3, int(round(fps * 0.10)))
    finish_min_fi: int | None = None
    if phase_id == "finish" and follow_through_anchor_fi is not None:
        finish_min_fi = int(follow_through_anchor_fi) + max(3, int(round(fps * 0.10)))

    for fi in range(lo, hi + 1):
        if impact_min_fi_from_top is not None and fi < impact_min_fi_from_top:
            continue
        if follow_min_fi is not None and fi < follow_min_fi:
            continue
        if finish_min_fi is not None and fi < finish_min_fi:
            continue
        pi = _nearest_pose_idx_for_video_frame(poses, fi)
        debug["scans"] = int(debug.get("scans", 0)) + 1
        sc = 0.0
        if kin is None:
            sc = -abs(pi - cpi) * 0.01
        elif phase_id == "top":
            ok_t, _ = validate_top_semantic_at_index(pi, kin)
            sp = float(kin["speed_s"][pi]) if pi < len(kin["speed_s"]) else 0.0
            qv = float(kin["q"][pi]) if pi < len(kin["q"]) else 0.0
            sc = (2.0 if ok_t else 0.0) + 0.55 * (sp / speed_mx) + 0.35 * min(qv, 1.0)
            sc -= 0.02 * abs(pi - top_ev)
        elif phase_id == "impact":
            top_pi = int(phase_keyframes.get("top", 0))
            ok_i, _ = validate_impact_semantic_at_index(pi, top_pi, exc_ix, kin)
            sp = float(kin["speed_s"][pi]) if pi < len(kin["speed_s"]) else 0.0
            sc = (2.2 if ok_i else 0.0) + 0.6 * (sp / speed_mx)
            sc -= 0.02 * abs(pi - imp_ev)
        elif phase_id == "takeaway":
            top_pi = int(phase_keyframes.get("top", min(len(poses) - 1, max(cpi + 1, 1))))
            if pi >= top_pi:
                continue
            sp = float(kin["speed_s"][pi]) if pi < len(kin["speed_s"]) else 0.0
            ex = float(kin["excursion"][pi]) if pi < len(kin["excursion"]) else 0.0
            ex_top = float(kin["excursion"][top_pi]) if top_pi < len(kin["excursion"]) else 1.0
            prog = min(max(ex / max(ex_top, 1e-6), 0.0), 1.0)
            sc = 0.9 + 0.45 * prog + 0.25 * min(sp / speed_mx, 1.0)
        elif phase_id == "backswing":
            top_pi = int(phase_keyframes.get("top", min(len(poses) - 1, max(cpi + 1, 1))))
            if pi >= top_pi:
                continue
            ex = float(kin["excursion"][pi]) if pi < len(kin["excursion"]) else 0.0
            ex_top = float(kin["excursion"][top_pi]) if top_pi < len(kin["excursion"]) else 1.0
            prog = min(max(ex / max(ex_top, 1e-6), 0.0), 1.0)
            # Prefer later backswing but before top.
            sc = 1.0 + 0.6 * (1.0 - abs(prog - 0.72))
        elif phase_id == "downswing":
            top_pi = int(phase_keyframes.get("top", 0))
            imp_pi = int(phase_keyframes.get("impact", min(len(poses) - 1, top_pi + 2)))
            if pi <= top_pi or pi >= imp_pi:
                continue
            sp = float(kin["speed_s"][pi]) if pi < len(kin["speed_s"]) else 0.0
            seg = kin["speed_s"][top_pi: max(top_pi + 1, imp_pi + 1)]
            seg_peak = float(np.max(seg)) + 1e-6 if len(seg) else speed_mx
            prog = (pi - top_pi) / max(imp_pi - top_pi, 1)
            sc = 1.05 + 0.5 * min(sp / seg_peak, 1.0) + 0.35 * (1.0 - abs(prog - 0.62))
        elif phase_id == "follow_through":
            imp_pi = int(phase_keyframes.get("impact", 0))
            if pi <= imp_pi:
                continue
            sp = float(kin["speed_s"][pi]) if pi < len(kin["speed_s"]) else 0.0
            seg = kin["speed_s"][imp_pi : min(len(kin["speed_s"]), imp_pi + 12)]
            peak = float(np.max(seg)) + 1e-6 if len(seg) else sp
            sc = 1.0 + 0.5 * max(0.0, 1.0 - sp / peak)
        elif phase_id == "finish":
            sp = float(kin["speed_s"][pi]) if pi < len(kin["speed_s"]) else 0.0
            vok = float(kin["valid"][pi]) if pi < len(kin["valid"]) else 0.0
            sc = 1.0 + 0.4 * max(0.0, 1.0 - sp / speed_mx) + 0.1 * vok
        if sc > best_sc:
            best_sc, best_pi, best_fi = sc, pi, fi

    pose_fi = int(poses[best_pi].get("frame_index", best_fi)) if 0 <= best_pi < len(poses) else int(best_fi)
    return {
        "best_source_pose_idx": int(best_pi),
        # Keep image frame and pose source frame in the same coordinate space.
        "best_source_frame_index": int(pose_fi),
        "refine_debug": {**debug, "best_score": round(float(best_sc), 4)},
    }


def _apply_raw_timeline_refine_to_smart_selection(
    cap: cv2.VideoCapture,
    rotation: int,
    fps: float,
    poses: list[dict],
    phase_keyframes: dict[str, int],
    selected: dict[str, dict],
    total_video_frames: int,
) -> None:
    """Mutates selected + phase_keyframes for top/impact/follow/finish using raw-frame local search."""
    selected_before = {k: dict(v) for k, v in selected.items()}
    phase_before = dict(phase_keyframes)
    impact_fi_anchor: int | None = None
    ft_fi_anchor: int | None = None
    for phase_id in ("takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"):
        if phase_id not in selected:
            continue
        cand_pi = int(selected[phase_id]["pose_idx"])
        res = refine_phase_keyframe_on_raw_timeline(
            cap,
            rotation,
            fps,
            poses,
            phase_keyframes,
            phase_id,
            cand_pi,
            search_radius_frames=None,
            total_video_frames=int(total_video_frames),
            impact_anchor_fi=impact_fi_anchor,
            follow_through_anchor_fi=ft_fi_anchor,
        )
        new_pi = int(res["best_source_pose_idx"])
        new_fi = int(res["best_source_frame_index"])
        frame = _read_frame_pose_matched(cap, new_fi, rotation)
        if frame is None:
            frame = _read_frame_with_decode_fallback(cap, new_fi, rotation)
        if frame is None:
            continue
        selected[phase_id]["pose_idx"] = new_pi
        selected[phase_id]["frame"] = frame
        phase_keyframes[phase_id] = new_pi
        if phase_id == "impact":
            impact_fi_anchor = new_fi
        elif phase_id == "follow_through":
            ft_fi_anchor = new_fi
    tmp_keyframes: list[dict] = []
    for pid in PHASE_ORDER:
        s = selected.get(pid)
        if not s:
            continue
        pi = int(s.get("pose_idx", -1))
        if not (0 <= pi < len(poses)):
            continue
        p = poses[pi]
        fi = int(p.get("frame_index", 0))
        ts = float(p.get("timestamp", round(fi / max(float(fps), 1e-6), 3)))
        tmp_keyframes.append(
            {
                "phase": pid,
                "source_pose_idx": pi,
                "source_frame_index": fi,
                "frame_index": fi,
                "timestamp": ts,
            }
        )
    contract_ok, _reasons = _keyframe_atomic_contract_ok(tmp_keyframes, float(fps), 0.0)
    if not contract_ok:
        selected.clear()
        selected.update(selected_before)
        phase_keyframes.clear()
        phase_keyframes.update(phase_before)
        logger.warning("[keyframe] raw timeline refine rejected; kept pre-refine legal strip")


def validate_final_keyframes_for_ai(
    keyframes: list[dict],
    phase_keyframes: dict[str, int],
    kf_validation_details: list[dict],
    poses: list[dict] | None = None,
    *,
    relabel_count: int = 0,
    fps: float | None = None,
) -> dict:
    """Router-side gate: 8/8 phase contract, strict monotonic fi/ts, phase map sync, semantic strip.

    ``all_passed`` / ``pass`` align with ``final_keyframe_gate_pass`` in callers: any duplicate strip,
    time-too-close row, per-row validation failure, ordering/sync failure, monotonic raw-frame gap
    violation, or semantic strip failure yields ``pass=False``.
    """
    if not keyframes:
        empty_checks = {
            "phase_count_ok": False,
            "phase_sequence_ok": False,
            "unique_source_pose_ok": False,
            "final_keyframe_order_ok": False,
            "final_keyframe_time_order_ok": False,
            "final_phase_keyframes_sync_ok": False,
            # True = at least one detail row has time_gap < 0 (contract problem).
            "negative_time_gap_in_details": False,
            "no_negative_time_gap": True,
            "near_duplicates_ok": False,
            "time_too_close_ok": False,
            "details_all_passed": False,
            "adjacent_strip_ok": False,
            "source_frame_gaps_ok": False,
        }
        return {
            "pass": False,
            "strict_contract_ok": False,
            "semantic_strip_ok": False,
            "semantic_strip_reasons": [],
            "phase_count_ok": False,
            "phase_sequence_ok": False,
            "unique_source_pose_ok": False,
            "final_keyframe_order_ok": False,
            "final_keyframe_time_order_ok": False,
            "final_phase_keyframes_sync_ok": False,
            "negative_time_gap_in_details": False,
            "strict_contract_checks": empty_checks,
            "strict_contract_fail_reasons": ["PHASE_COUNT_NOT_8"],
            "relabel_count": 0,
            "rebuild_used": False,
            "rebuild_reasons": [],
            "top_reselected": False,
            "impact_reselected": False,
            "phase_strip_repaired": False,
            "near_duplicates": 0,
            "time_too_close_count": 0,
            "strip_detail_any_failed": False,
            "source_frame_gaps_ok": True,
            "source_frame_gap_reasons": [],
            "adjacent_strip_hard_dup_reasons": [],
        }
    fi_ok, ts_ok = _strict_increasing_ts_and_fi(keyframes)
    sync_ok = True
    for kf in keyframes:
        pid = kf.get("phase")
        spi = kf.get("source_pose_idx")
        if pid is None or spi is None:
            sync_ok = False
            continue
        if int(phase_keyframes.get(str(pid), -999999)) != int(spi):
            sync_ok = False
            break
    neg_gap = False
    for d in kf_validation_details:
        tg = d.get("time_gap")
        if tg is not None and float(tg) < 0:
            neg_gap = True
            break

    phases = [kf.get("phase") for kf in keyframes]
    phase_count_ok = len(keyframes) == 8
    phase_sequence_ok = phase_count_ok and phases == PHASE_ORDER
    spis: list[int] = []
    for kf in keyframes:
        spi = kf.get("source_pose_idx")
        if spi is None:
            continue
        spis.append(int(spi))
    unique_source_pose_ok = len(spis) == 8 and len(set(spis)) == 8

    n_dup_details = sum(
        1 for d in kf_validation_details if isinstance(d, dict) and d.get("is_near_duplicate")
    )
    n_tc_details = sum(
        1 for d in kf_validation_details if isinstance(d, dict) and d.get("time_too_close")
    )
    any_detail_failed = any(
        isinstance(d, dict) and not d.get("validation_passed", True)
        for d in kf_validation_details
    )
    strip_quality_hard_ok = (
        n_dup_details == 0 and n_tc_details == 0 and not any_detail_failed
    )

    adjacent_reasons: list[str] = []
    if poses and len(keyframes) == 8:
        adjacent_reasons = _adjacent_phase_keyframe_hard_dup_reasons(keyframes, poses)

    gap_ok = True
    gap_reasons: list[str] = []
    if fps is not None and len(keyframes) == 8:
        gap_ok, gap_reasons = _source_frame_index_phase_gaps_ok(keyframes, float(fps))

    strict_contract_checks: dict[str, bool] = {
        "phase_count_ok": phase_count_ok,
        "phase_sequence_ok": phase_sequence_ok,
        "unique_source_pose_ok": unique_source_pose_ok,
        "final_keyframe_order_ok": fi_ok,
        "final_keyframe_time_order_ok": ts_ok,
        "final_phase_keyframes_sync_ok": sync_ok,
        # Align with top-level ``negative_time_gap_in_details``: True when a negative time_gap exists.
        "negative_time_gap_in_details": bool(neg_gap),
        "no_negative_time_gap": not neg_gap,
        "near_duplicates_ok": n_dup_details == 0,
        "time_too_close_ok": n_tc_details == 0,
        "details_all_passed": not any_detail_failed,
        "adjacent_strip_ok": len(adjacent_reasons) == 0,
        "source_frame_gaps_ok": gap_ok,
    }

    strict_contract_fail_reasons: list[str] = []
    if not phase_count_ok:
        strict_contract_fail_reasons.append("PHASE_COUNT_NOT_8")
    if not phase_sequence_ok:
        strict_contract_fail_reasons.append("PHASE_SEQUENCE_MISMATCH")
    if not unique_source_pose_ok:
        strict_contract_fail_reasons.append("SOURCE_POSE_NOT_UNIQUE")
    if not fi_ok:
        strict_contract_fail_reasons.append("FRAME_INDEX_NOT_MONOTONIC")
    if not ts_ok:
        strict_contract_fail_reasons.append("TIMESTAMP_NOT_MONOTONIC")
    if not sync_ok:
        strict_contract_fail_reasons.append("PHASE_KEYFRAME_SYNC_FAIL")
    if neg_gap:
        strict_contract_fail_reasons.append("NEGATIVE_TIME_GAP_IN_DETAILS")
    if n_dup_details > 0:
        strict_contract_fail_reasons.append("NEAR_DUPLICATE_PRESENT")
    if n_tc_details > 0:
        strict_contract_fail_reasons.append("TIME_TOO_CLOSE_PRESENT")
    # Keep ``DETAIL_VALIDATION_FAILED`` for non-overlapping detail failures only.
    # When near-duplicate / time-too-close reasons are already present, this avoids
    # reporting an additional generic reason for the same row-level issue.
    if any_detail_failed and n_dup_details == 0 and n_tc_details == 0:
        strict_contract_fail_reasons.append("DETAIL_VALIDATION_FAILED")
    if adjacent_reasons:
        strict_contract_fail_reasons.append("ADJACENT_STRIP_HARD_DUP")
    for gr in gap_reasons:
        if gr and gr not in strict_contract_fail_reasons:
            strict_contract_fail_reasons.append(str(gr))

    strict_contract_ok = bool(
        phase_sequence_ok
        and unique_source_pose_ok
        and fi_ok
        and ts_ok
        and sync_ok
        and not neg_gap
        and strip_quality_hard_ok
        and not adjacent_reasons
        and gap_ok
    )

    semantic_strip_ok = False
    semantic_strip_reasons: list[str] = []
    if poses is not None:
        sem = verify_phase_strip_semantics(keyframes, poses, phase_keyframes)
        semantic_strip_ok = bool(sem.get("pass"))
        semantic_strip_reasons = list(sem.get("reasons") or [])
    else:
        semantic_strip_reasons = ["POSES_UNAVAILABLE_FOR_SEMANTIC_STRIP"]

    rc_in = int(relabel_count or 0)
    if rc_in > 1:
        semantic_strip_ok = False
        if "PHASE_STRIP_REPAIR_FAILED_RELABEL_FORBIDDEN" not in semantic_strip_reasons:
            semantic_strip_reasons.append("PHASE_STRIP_REPAIR_FAILED_RELABEL_FORBIDDEN")

    for r in adjacent_reasons:
        if r not in semantic_strip_reasons:
            semantic_strip_reasons.append(r)
    if not gap_ok:
        semantic_strip_ok = False
        for r in gap_reasons:
            if r not in semantic_strip_reasons:
                semantic_strip_reasons.append(r)

    pass_all = bool(strict_contract_ok and semantic_strip_ok and rc_in <= 1)

    return {
        "pass": pass_all,
        "strict_contract_ok": strict_contract_ok,
        "semantic_strip_ok": semantic_strip_ok,
        "semantic_strip_reasons": semantic_strip_reasons,
        "phase_count_ok": phase_count_ok,
        "phase_sequence_ok": phase_sequence_ok,
        "unique_source_pose_ok": unique_source_pose_ok,
        "final_keyframe_order_ok": fi_ok,
        "final_keyframe_time_order_ok": ts_ok,
        "final_phase_keyframes_sync_ok": sync_ok,
        "negative_time_gap_in_details": neg_gap,
        "strict_contract_checks": dict(strict_contract_checks),
        "strict_contract_fail_reasons": list(strict_contract_fail_reasons),
        "relabel_count": rc_in,
        "rebuild_used": False,
        "rebuild_reasons": [],
        "top_reselected": False,
        "impact_reselected": False,
        "phase_strip_repaired": False,
        "near_duplicates": n_dup_details,
        "time_too_close_count": n_tc_details,
        "strip_detail_any_failed": bool(any_detail_failed),
        "source_frame_gaps_ok": bool(gap_ok),
        "source_frame_gap_reasons": list(gap_reasons),
        "adjacent_strip_hard_dup_reasons": list(adjacent_reasons),
    }


def extract_keyframes_ordered_fallback(
    video_path: str,
    poses: list[dict],
    _swing_phases: list[dict],
    phase_keyframes_ordered: dict[str, int],
    keyframe_width: int = 320,
    *,
    enforce_time_gap: bool = True,
) -> tuple[list[dict], dict]:
    """
    Deterministic keyframes from a monotonic phase map — no near-duplicate reselect.
    Enforces strictly increasing pose index, frame_index, and timestamp.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], {
            "total_keyframes": 0,
            "near_duplicates": 0,
            "time_too_close": 0,
            "all_passed": False,
            "details": [],
            "source": "ordered_fallback",
        }
    rotation = get_video_rotation(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = len(poses)
    total_duration = max(
        float(poses[-1].get("timestamp", 0)) - float(poses[0].get("timestamp", 0)),
        0.1,
    ) if poses else 0.1
    min_time_gap = max(total_duration * _MIN_TIME_INTERVAL_RATIO, 0.05)

    prev_pi = -1
    prev_fi = -1
    prev_ts = -999.0
    keyframes: list[dict] = []
    details: list[dict] = []

    for phase_id in PHASE_ORDER:
        meta = SWING_PHASE_META[phase_id]
        raw = phase_keyframes_ordered.get(phase_id)
        start_pi = int(raw) if isinstance(raw, int) else (prev_pi + 1 if prev_pi >= 0 else 0)
        pi = max(start_pi, prev_pi + 1, 0)
        pi = min(pi, max(n - 1, 0))
        chosen = None
        used_relaxed = False
        scan_from = pi
        for cand_pi in range(scan_from, n):
            pose = poses[cand_pi]
            cand_fi = int(pose.get("frame_index", 0))
            cand_ts = float(pose.get("timestamp", round(cand_fi / max(fps, 1e-6), 3)))
            if prev_fi >= 0 and cand_fi <= prev_fi:
                continue
            if enforce_time_gap and prev_ts >= 0 and cand_ts < _min_timestamp_after_prev(prev_ts, min_time_gap):
                continue
            frame = _read_frame_pose_matched(cap, cand_fi, rotation)
            if frame is None:
                frame = _read_frame_with_decode_fallback(cap, cand_fi, rotation)
            if frame is None:
                continue
            chosen = (cand_pi, cand_fi, cand_ts, frame, pose)
            break
        if chosen is None:
            # Monotonic frame_index can be impossible for some phase maps (e.g. respaced Gemini map vs pose order).
            # Prefer a decodable frame at a later pose index than dropping the phase (empty UI slot).
            fb_lo = min(max(scan_from, prev_pi + 1), max(n - 1, 0))
            for fb_pi in range(fb_lo, n):
                pose_fb = poses[fb_pi]
                cand_fi_fb = int(pose_fb.get("frame_index", 0))
                cand_ts_fb = float(pose_fb.get("timestamp", round(cand_fi_fb / max(fps, 1e-6), 3)))
                frame_fb = _read_frame_with_decode_fallback(cap, cand_fi_fb, rotation)
                if frame_fb is None:
                    continue
                chosen = (fb_pi, cand_fi_fb, cand_ts_fb, frame_fb, pose_fb)
                used_relaxed = True
                break
        if chosen is None:
            continue
        cand_pi, cand_fi, cand_ts, frame, pose = chosen
        resized = _resize_frame(frame, keyframe_width)
        pose_snapshot = _pose_snapshot_for_keyframe(pose)
        pq = _pose_quality_details(pose)
        iq = _frame_quality_details(frame)
        conf = float(np.clip(0.35 * pq["quality"] + 0.25 * iq["quality"] + 0.35, 0.0, 1.0))
        time_gap = round((cand_ts - prev_ts) if prev_ts >= 0 else 999.0, 3)
        details.append({
            "phase": phase_id,
            "visual_diff_from_prev": 1.0,
            "is_near_duplicate": False,
            "time_gap": time_gap,
            "time_too_close": (time_gap < min_time_gap and prev_ts >= 0) if enforce_time_gap else False,
            "validation_passed": True,
            "ordered_fallback": True,
            "ordered_fallback_relaxed": used_relaxed,
        })
        keyframes.append({
            "phase": phase_id,
            "label_en": meta["label_en"],
            "label_zh": meta["label_zh"],
            "frame_index": cand_fi,
            "timestamp": round(cand_ts, 3),
            "confidence": round(conf, 3),
            "selection_reason": "ordered_fallback_relaxed" if used_relaxed else "ordered_fallback_monotonic",
            "fallback_used": True,
            "image_base64": frame_to_base64(resized),
            "width": resized.shape[1],
            "height": resized.shape[0],
            "pose_snapshot": pose_snapshot,
            "source_pose_idx": cand_pi,
            "source_frame_index": cand_fi,
            "visual_diff_from_prev": 1.0,
            "phase_validation_passed": True,
        })
        prev_pi, prev_fi, prev_ts = cand_pi, cand_fi, cand_ts

    cap.release()
    phase_kf: dict[str, int] = {}
    _sync_phase_keyframes_from_keyframes(keyframes, phase_kf)
    n_dup = sum(1 for d in details if d.get("is_near_duplicate"))
    n_close = sum(1 for d in details if d.get("time_too_close"))
    summary = {
        "total_keyframes": len(keyframes),
        "near_duplicates": n_dup,
        "time_too_close": n_close,
        "all_passed": n_dup == 0 and n_close == 0,
        "details": details,
        "source": "ordered_fallback",
        "final_phase_keyframes": dict(phase_kf),
    }
    return keyframes, summary


def build_uniform_spaced_phase_keyframes(poses: list[dict]) -> dict[str, int]:
    """Eight strictly increasing pose indices spread across the clip (ignores semantic phase map).

    Used when smart strip repairs no-op so Plus can rebind from a **different** temporal ladder.
    """
    n = len(poses)
    if n <= 0:
        return {p: 0 for p in PHASE_ORDER}
    if n < 8:
        out: dict[str, int] = {}
        prev = -1
        for i, p in enumerate(PHASE_ORDER):
            pi = min(max(prev + 1, int(round(i * (n - 1) / 7))), n - 1)
            out[p] = pi
            prev = pi
        return out
    raw = [int(round(i * (n - 1) / 7)) for i in range(8)]
    out_m: dict[str, int] = {}
    prev = -1
    for i, p in enumerate(PHASE_ORDER):
        target = raw[i]
        pi = max(prev + 1, min(target, n - 1))
        out_m[p] = pi
        prev = pi
    return out_m


def try_recover_keyframes_with_authoritative_phase_map(
    video_path: str,
    poses: list[dict],
    swing_phases: list[dict],
    keyframe_width: int,
    vid_fps: float,
    prev_fv: dict,
    authoritative_pre_extract_phase_map: dict[str, int] | None,
    authoritative_chain_backend_ok: bool,
    merged_base: dict,
) -> tuple[list[dict], dict, dict[str, int], str] | None:
    """Rebuild an 8-strip via ordered fallback from the pre-extract authoritative chain (Plus).

    Used when all other repairs would end in ``smart_gate_failed`` but the backend already
    validated ``authoritative_phase_chain`` — semantic indices are preserved via monotonic scan.
    """
    if not authoritative_chain_backend_ok or not authoritative_pre_extract_phase_map:
        return None
    auth = dict(authoritative_pre_extract_phase_map)
    if len(auth) != 8 or len(poses) < 8:
        return None
    if not all(isinstance(auth.get(p), int) for p in PHASE_ORDER):
        return None
    from services.swing_flow_utils import respace_phase_keyframes

    seeds: list[tuple[str, dict[str, int]]] = [
        ("authoritative_pre_extract_raw", dict(auth)),
        ("authoritative_pre_extract_respaced", respace_phase_keyframes(dict(auth), len(poses))),
    ]
    for label, seed in seeds:
        af_kf, af_sum = extract_keyframes_ordered_fallback(
            video_path,
            poses,
            swing_phases,
            seed,
            keyframe_width=keyframe_width,
            enforce_time_gap=True,
        )
        if len(af_kf) != 8:
            logger.info(
                "[keyframe] authoritative_chain_recovery incomplete seed=%s keyframes=%d",
                label,
                len(af_kf),
            )
            continue
        af_details = list(af_sum.get("details") or [])
        af_phase = dict(af_sum.get("final_phase_keyframes") or {})
        af_gate = validate_final_keyframes_for_ai(
            af_kf, af_phase, af_details, poses=poses, fps=vid_fps,
        )
        af_gate = dict(af_gate)
        for _pk in _REPAIR_FV_PRESERVE_KEYS:
            if _pk in prev_fv:
                af_gate[_pk] = prev_fv[_pk]
        if not bool(af_gate.get("pass")):
            continue
        out_phase: dict[str, int] = {}
        out_phase.update(af_phase)
        merged = {
            **merged_base,
            "details": af_details,
            "final_phase_keyframes": dict(af_phase),
            "final_keyframe_validation": dict(af_gate),
            "final_keyframe_order_ok": bool(af_gate.get("final_keyframe_order_ok")),
            "final_keyframe_time_order_ok": bool(af_gate.get("final_keyframe_time_order_ok")),
            "final_phase_keyframes_sync_ok": bool(af_gate.get("final_phase_keyframes_sync_ok")),
            "negative_time_gap_in_details": bool(af_gate.get("negative_time_gap_in_details")),
            "final_keyframe_gate_pass": True,
            "all_passed": True,
            "final_validation_failed": False,
            "final_keyframe_source": "ordered_fallback_authoritative_chain",
            "authoritative_chain_recovered_in_ensure": True,
            "authoritative_chain_recovery_seed": label,
            "repair_state": None,
        }
        logger.info(
            "[keyframe] ensure exit source=ordered_fallback_authoritative_chain gate_pass=True "
            "recovery_seed=%s source_pose_idx=%s",
            label,
            [int(k.get("source_pose_idx", -1)) for k in af_kf],
        )
        return af_kf, merged, out_phase, "ordered_fallback_authoritative_chain"
    logger.warning("[keyframe] authoritative_chain_recovery_failed seeds_tried=raw,respaced")
    return None


def ensure_keyframes_ordered_for_ai(
    video_path: str,
    poses: list[dict],
    swing_phases: list[dict],
    phase_keyframes_ordered_snapshot: dict[str, int],
    keyframes: list[dict],
    kf_validation: dict,
    phase_keyframes: dict[str, int],
    keyframe_width: int = 320,
    *,
    force_uniform_temporal_fallback: bool = False,
    authoritative_pre_extract_phase_map: dict[str, int] | None = None,
    authoritative_chain_backend_ok: bool = False,
    tracks: dict[str, Any] | None = None,
) -> tuple[list[dict], dict, dict[str, int], str]:
    """
    If smart extraction failed monotonic/sync gates, rebuild from respaced phase map.
    Returns (keyframes, merged_kf_validation, phase_keyframes_out, final_keyframe_source).
    """
    logger.info(
        "[keyframe] ensure enter in_keyframes=%d in_phase_keyframes=%d snapshot_phase=%d "
        "authoritative_chain_backend_ok=%s authoritative_pre_extract_keys=%d",
        len(keyframes or []),
        len(phase_keyframes or {}),
        len(phase_keyframes_ordered_snapshot or {}),
        bool(authoritative_chain_backend_ok),
        len(authoritative_pre_extract_phase_map or {}),
    )
    details = list(kf_validation.get("details") or [])
    prev_fv = dict(kf_validation.get("final_keyframe_validation") or {})
    vcap = cv2.VideoCapture(video_path)
    vid_fps = float(vcap.get(cv2.CAP_PROP_FPS) or 30.0)
    vcap.release()

    if force_uniform_temporal_fallback and poses and len(poses) >= 8:
        uni = build_uniform_spaced_phase_keyframes(poses)
        of_kf, of_summary = extract_keyframes_ordered_fallback(
            video_path,
            poses,
            swing_phases,
            uni,
            keyframe_width,
        )
        of_details = list(of_summary.get("details") or [])
        of_phase = dict(of_summary.get("final_phase_keyframes") or {})
        of_gate = validate_final_keyframes_for_ai(
            of_kf,
            of_phase,
            of_details,
            poses=poses,
            fps=vid_fps,
        )
        of_gate = dict(of_gate)
        for _pk in _REPAIR_FV_PRESERVE_KEYS:
            if _pk in prev_fv:
                of_gate[_pk] = prev_fv[_pk]
        old_spi = [int(k.get("source_pose_idx", -1)) for k in keyframes]
        new_spi = [int(k.get("source_pose_idx", -1)) for k in (of_kf or [])]
        mat = bool(of_kf) and old_spi != new_spi
        merged_u: dict[str, Any] = {
            **kf_validation,
            "details": of_details,
            "final_phase_keyframes": dict(of_phase),
            "final_keyframe_validation": dict(of_gate),
            "final_keyframe_order_ok": bool(of_gate.get("final_keyframe_order_ok")),
            "final_keyframe_time_order_ok": bool(of_gate.get("final_keyframe_time_order_ok")),
            "final_phase_keyframes_sync_ok": bool(of_gate.get("final_phase_keyframes_sync_ok")),
            "negative_time_gap_in_details": bool(of_gate.get("negative_time_gap_in_details")),
            "final_keyframe_gate_pass": bool(of_gate.get("pass")),
            "all_passed": bool(of_gate.get("pass")),
            "final_validation_failed": not bool(of_gate.get("pass")),
            "final_keyframe_source": "ordered_fallback_temporal_spread",
            "uniform_temporal_fallback_applied": True,
            "fallback_rebuild_material_change": mat,
            "_skip_strip_quality_round": False,
            "repair_state": None,
        }
        logger.info(
            "[keyframe] ensure uniform_temporal_fallback gate_pass=%s material_change=%s spi_old=%s spi_new=%s",
            bool(of_gate.get("pass")),
            mat,
            old_spi,
            new_spi,
        )
        if bool(of_gate.get("pass")):
            return of_kf, merged_u, of_phase, "ordered_fallback_temporal_spread"

        # Uniform fallback can produce semantically wrong strips in some degraded runs.
        # If it still fails strict gate, keep the current strip/images and only record
        # that fallback was attempted; this avoids replacing user-facing keyframe images
        # with known-bad temporal placeholders.
        kept_phase: dict[str, int] = {}
        _sync_phase_keyframes_from_keyframes(keyframes, kept_phase)
        kept_gate = dict(validate_final_keyframes_for_ai(
            keyframes,
            kept_phase,
            details,
            poses=poses,
            fps=vid_fps,
        ))
        for _pk in _REPAIR_FV_PRESERVE_KEYS:
            if _pk in prev_fv:
                kept_gate[_pk] = prev_fv[_pk]
        merged_keep: dict[str, Any] = {
            **kf_validation,
            "final_phase_keyframes": dict(kept_phase),
            "final_keyframe_validation": dict(kept_gate),
            "final_keyframe_order_ok": bool(kept_gate.get("final_keyframe_order_ok")),
            "final_keyframe_time_order_ok": bool(kept_gate.get("final_keyframe_time_order_ok")),
            "final_phase_keyframes_sync_ok": bool(kept_gate.get("final_phase_keyframes_sync_ok")),
            "negative_time_gap_in_details": bool(kept_gate.get("negative_time_gap_in_details")),
            "final_keyframe_gate_pass": bool(kept_gate.get("pass")),
            "all_passed": bool(kept_gate.get("pass")),
            "final_validation_failed": not bool(kept_gate.get("pass")),
            "final_keyframe_source": str(kf_validation.get("final_keyframe_source") or "smart_gate_failed"),
            "uniform_temporal_fallback_applied": False,
            "uniform_temporal_fallback_attempted": True,
            "uniform_temporal_fallback_rejected": True,
            "fallback_rebuild_material_change": mat,
            "_skip_strip_quality_round": False,
            "repair_state": None,
        }
        logger.warning(
            "[keyframe] ensure uniform_temporal_fallback rejected (gate_pass=false); keeping previous keyframe strip",
        )
        return keyframes, merged_keep, kept_phase, str(merged_keep.get("final_keyframe_source"))

    try:
        gate = validate_final_keyframes_for_ai(
            keyframes, phase_keyframes, details, poses=poses, fps=vid_fps,
        )
    except Exception:
        logger.exception("[keyframe] ensure initial validate failed; continue with fallback path")
        gate = {
            "pass": False,
            "strict_contract_ok": False,
            "semantic_strip_ok": False,
            "strict_contract_fail_reasons": ["ENSURE_INITIAL_VALIDATE_EXCEPTION"],
            "semantic_strip_reasons": ["ENSURE_INITIAL_VALIDATE_EXCEPTION"],
            "final_keyframe_order_ok": False,
            "final_keyframe_time_order_ok": False,
            "final_phase_keyframes_sync_ok": False,
            "negative_time_gap_in_details": False,
        }
    gate = dict(gate)
    for _pk in _REPAIR_FV_PRESERVE_KEYS:
        if _pk in prev_fv:
            gate[_pk] = prev_fv[_pk]
    out_phase: dict[str, int] = {}
    _sync_phase_keyframes_from_keyframes(keyframes, out_phase)

    smart_repaired = any(
        bool(d.get("monotonic_repair")) or bool(d.get("reselected"))
        for d in details
    )
    merged: dict = {
        **kf_validation,
        "final_keyframe_order_ok": gate["final_keyframe_order_ok"],
        "final_keyframe_time_order_ok": gate["final_keyframe_time_order_ok"],
        "final_phase_keyframes_sync_ok": gate["final_phase_keyframes_sync_ok"],
        "negative_time_gap_in_details": gate["negative_time_gap_in_details"],
        "final_phase_keyframes": dict(out_phase),
        "final_keyframe_validation": dict(gate),
        "final_keyframe_gate_pass": gate["pass"],
        # Align with ``final_keyframe_validation.pass`` (strict AI gate — no silent strip mismatch).
        "all_passed": bool(gate["pass"]),
    }
    def _missing_image_count(rows: list[dict]) -> int:
        return sum(1 for r in rows if not str(r.get("image_base64") or "").strip())

    best_candidate: tuple[list[dict], dict[str, int], list[dict], dict, str] | None = None

    if gate["pass"]:
        merged["final_validation_failed"] = False
        merged["final_keyframe_source"] = "smart_repaired" if smart_repaired else "smart"
        logger.info(
            "[keyframe] ensure exit source=%s gate_pass=%s out_keyframes=%d out_phase_keyframes=%d strict_reasons=%s semantic_reasons=%s",
            str(merged.get("final_keyframe_source")),
            bool(merged.get("final_keyframe_gate_pass")),
            len(keyframes or []),
            len(out_phase or {}),
            list((gate.get("strict_contract_fail_reasons") or [])),
            list((gate.get("semantic_strip_reasons") or [])),
        )
        return keyframes, merged, out_phase, str(merged["final_keyframe_source"])

    merged["_skip_strip_quality_round"] = bool(kf_validation.get("_skip_strip_quality_round"))
    merged["_joint_repair_ran"] = False

    # Joint post-top rebuild first when frame/pose order or post-top semantics failed.
    strict_fail_list = list(gate.get("strict_contract_fail_reasons") or [])
    sem_fail_set = set(gate.get("semantic_strip_reasons") or [])
    want_joint = (
        _strict_reasons_need_joint_repair(strict_fail_list)
        or bool(sem_fail_set & _JOINT_REPAIR_SEMANTIC_TRIGGERS)
    )
    if want_joint and poses and len(keyframes) == 8:
        jm, _j_mat = joint_rebuild_phase_map_for_monotonic_strip(
            poses, dict(out_phase or phase_keyframes), fps=float(vid_fps),
        )
        j_cap = cv2.VideoCapture(video_path)
        j_kf: list[dict] = list(keyframes)
        j_det: list[dict] = list(details)
        try:
            if j_cap.isOpened():
                j_kf, j_det = _rebind_keyframes_from_rebuilt_map(
                    j_cap,
                    get_video_rotation(video_path),
                    float(vid_fps),
                    poses,
                    jm,
                    keyframe_width,
                )
        finally:
            j_cap.release()
        j_gate = validate_final_keyframes_for_ai(
            j_kf, jm, j_det, poses=poses, fps=vid_fps,
        )
        merged["_joint_repair_ran"] = True
        if bool(j_gate.get("pass")):
            out_phase.clear()
            out_phase.update(jm)
            _sync_phase_keyframes_from_keyframes(j_kf, out_phase)
            merged.update(
                {
                    "details": j_det,
                    "final_phase_keyframes": dict(out_phase),
                    "final_keyframe_validation": dict(j_gate),
                    "final_keyframe_order_ok": bool(j_gate.get("final_keyframe_order_ok")),
                    "final_keyframe_time_order_ok": bool(j_gate.get("final_keyframe_time_order_ok")),
                    "final_phase_keyframes_sync_ok": bool(j_gate.get("final_phase_keyframes_sync_ok")),
                    "final_keyframe_gate_pass": True,
                    "all_passed": True,
                    "final_validation_failed": False,
                    "final_keyframe_source": "joint_post_top_repaired",
                    "phase_strip_repaired": True,
                    "_skip_strip_quality_round": False,
                    "repair_state": None,
                }
            )
            logger.info(
                "[keyframe] ensure exit source=joint_post_top_repaired gate_pass=True out_keyframes=%d",
                len(j_kf or []),
            )
            return j_kf, merged, out_phase, "joint_post_top_repaired"
        if keyframe_repair_score(j_gate, j_kf) < keyframe_repair_score(gate, keyframes):
            best_candidate = (
                j_kf,
                dict(jm),
                list(j_det),
                dict(j_gate),
                "joint_post_top_best_effort",
            )
        keyframes = j_kf
        out_phase.clear()
        out_phase.update(jm)
        _sync_phase_keyframes_from_keyframes(keyframes, out_phase)
        details = list(j_det)
        gate = validate_final_keyframes_for_ai(
            keyframes, out_phase, details, poses=poses, fps=vid_fps,
        )
        for _pk in _REPAIR_FV_PRESERVE_KEYS:
            if _pk in prev_fv:
                gate[_pk] = prev_fv[_pk]
        merged["final_keyframe_validation"] = dict(gate)
        merged["details"] = details
        merged["final_keyframe_order_ok"] = gate["final_keyframe_order_ok"]
        merged["final_keyframe_time_order_ok"] = gate["final_keyframe_time_order_ok"]
        merged["final_phase_keyframes_sync_ok"] = gate["final_phase_keyframes_sync_ok"]
        merged["final_keyframe_gate_pass"] = gate["pass"]
        merged["all_passed"] = bool(gate["pass"])
        merged["repair_state"] = "joint_repair_attempted"
        merged["_skip_strip_quality_round"] = False
        if gate["pass"]:
            merged["final_validation_failed"] = False
            merged["final_keyframe_source"] = "joint_post_top_repaired"
            return keyframes, merged, out_phase, "joint_post_top_repaired"
        fail_reasons = set(gate.get("strict_contract_fail_reasons") or [])
    else:
        fail_reasons = set(gate.get("strict_contract_fail_reasons") or [])

    # Strip-quality local re-search before anchor rebuild (strict gate strip failures).
    if (
        not bool(merged.get("_skip_strip_quality_round"))
        and {"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT", "IMPACT_SEMANTIC_AT_KEYFRAME_FAIL", "POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"} & fail_reasons
    ):
        rcap = cv2.VideoCapture(video_path)
        try:
            if rcap.isOpened():
                logger.info(
                    "[keyframe] ensure before strip-quality repair keyframes=%d phase_keyframes=%d fail_reasons=%s semantic_reasons=%s",
                    len(keyframes or []),
                    len(out_phase or phase_keyframes or {}),
                    list(fail_reasons),
                    list((gate.get("semantic_strip_reasons") or [])),
                )
                repaired_kf, repaired_details = _reselect_strip_quality_failures(
                    rcap,
                    get_video_rotation(video_path),
                    float(vid_fps),
                    poses,
                    [dict(k) for k in keyframes],
                    dict(out_phase or phase_keyframes),
                    keyframe_width,
                    max(
                        (
                            float(poses[-1].get("timestamp", 0.0))
                            - float(poses[0].get("timestamp", 0.0))
                        ) * _MIN_TIME_INTERVAL_RATIO,
                        0.05,
                    ) if poses else 0.05,
                    semantic_fail_reasons=list((gate.get("semantic_strip_reasons") or [])),
                    strict_contract_fail_reasons=list((gate.get("strict_contract_fail_reasons") or [])),
                )
                logger.info(
                    "[keyframe] ensure after strip-quality repair repaired_keyframes=%d repaired_details=%d",
                    len(repaired_kf or []),
                    len(repaired_details or []),
                )
                repaired_phase: dict[str, int] = {}
                _sync_phase_keyframes_from_keyframes(repaired_kf, repaired_phase)
                old_spi = [int(k.get("source_pose_idx", -1)) for k in keyframes]
                new_spi = [int(k.get("source_pose_idx", -1)) for k in repaired_kf]
                old_sfi = [int(k.get("source_frame_index", k.get("frame_index", -1))) for k in keyframes]
                new_sfi = [int(k.get("source_frame_index", k.get("frame_index", -1))) for k in repaired_kf]
                changed_phase_ids = [
                    PHASE_ORDER[i]
                    for i in range(min(len(old_spi), len(new_spi), len(PHASE_ORDER)))
                    if old_spi[i] != new_spi[i] or old_sfi[i] != new_sfi[i]
                ]
                logger.info(
                    "[keyframe] ensure repair changed_phase_ids=%s old_spi=%s new_spi=%s old_sfi=%s new_sfi=%s",
                    changed_phase_ids,
                    old_spi,
                    new_spi,
                    old_sfi,
                    new_sfi,
                )
                logger.info(
                    "[ROLE=REPAIR_ENGINE] round=strip_quality changed_phase_ids=%s no_op=%s",
                    changed_phase_ids,
                    not bool(changed_phase_ids),
                )
                repair_noop = not bool(changed_phase_ids)
                if repair_noop:
                    logger.warning(
                        "[keyframe] ensure strip-quality repair no-op; forcing deeper rebuild/fallback path"
                    )
                    merged["repair_state"] = "strip_quality_noop"
                    merged["_skip_strip_quality_round"] = True
                else:
                    logger.info(
                        "[keyframe] ensure before validate repaired strip keyframes=%d phase_keyframes=%d",
                        len(repaired_kf or []),
                        len(repaired_phase or {}),
                    )
                    repaired_gate = validate_final_keyframes_for_ai(
                        repaired_kf,
                        repaired_phase,
                        repaired_details,
                        poses=poses,
                        fps=vid_fps,
                    )
                    if bool(repaired_gate.get("pass")):
                        merged.update(
                            {
                                "details": repaired_details,
                                "final_phase_keyframes": dict(repaired_phase),
                                "final_keyframe_validation": dict(repaired_gate),
                                "final_keyframe_order_ok": bool(repaired_gate.get("final_keyframe_order_ok")),
                                "final_keyframe_time_order_ok": bool(repaired_gate.get("final_keyframe_time_order_ok")),
                                "final_phase_keyframes_sync_ok": bool(repaired_gate.get("final_phase_keyframes_sync_ok")),
                                "final_keyframe_gate_pass": True,
                                "all_passed": True,
                                "final_validation_failed": False,
                                "final_keyframe_source": "smart_repaired",
                            }
                        )
                        logger.info(
                            "[keyframe] ensure exit source=%s gate_pass=%s out_keyframes=%d out_phase_keyframes=%d strict_reasons=%s semantic_reasons=%s",
                            "smart_repaired",
                            True,
                            len(repaired_kf or []),
                            len(repaired_phase or {}),
                            list((repaired_gate.get("strict_contract_fail_reasons") or [])),
                            list((repaired_gate.get("semantic_strip_reasons") or [])),
                        )
                        return repaired_kf, merged, repaired_phase, "smart_repaired"
                    if keyframe_repair_score(repaired_gate, repaired_kf) < keyframe_repair_score(gate, keyframes):
                        best_candidate = (
                            repaired_kf,
                            dict(repaired_phase),
                            list(repaired_details),
                            dict(repaired_gate),
                            "smart_repaired_best_effort",
                        )
        except Exception:
            logger.exception("[keyframe] ensure strip-quality repair crashed")
        finally:
            rcap.release()

    # Semantic-oriented phase map (kinematic milestones + joint post-top chain) — before lerp anchor rebuild.
    if poses and len(keyframes) == 8:
        try:
            sem_bundle = build_semantic_oriented_phase_map(
                poses, dict(out_phase or phase_keyframes), fps=float(vid_fps),
            )
            sem_dbg = dict(sem_bundle.get("debug") or {})
            merged["phase_oriented_semantic_debug"] = sem_dbg
            merged["final_phase_semantic_pass"] = bool(sem_dbg.get("final_phase_semantic_pass"))
            merged["final_phase_semantic_fail_reasons"] = list(sem_dbg.get("final_phase_semantic_fail_reasons") or [])
            _fkv_sem = dict(merged.get("final_keyframe_validation") or {})
            _fkv_sem["phase_oriented_semantic_debug"] = sem_dbg
            merged["final_keyframe_validation"] = _fkv_sem
            logger.info(
                "[keyframe] ensure semantic_oriented map_ok=%s final_phase_semantic_pass=%s "
                "phase_reselect_strategy=%s source_pose_idx_list=%s min_gap_frames=%s "
                "top_semantic_score=%s impact_semantic_score=%s finish_semantic_score=%s "
                "final_phase_semantic_fail_reasons=%s",
                bool(sem_bundle.get("ok")),
                sem_dbg.get("final_phase_semantic_pass"),
                sem_dbg.get("phase_reselect_strategy"),
                sem_dbg.get("source_pose_idx_list"),
                sem_dbg.get("min_gap_frames"),
                sem_dbg.get("top_semantic_score"),
                sem_dbg.get("impact_semantic_score"),
                sem_dbg.get("finish_semantic_score"),
                sem_dbg.get("final_phase_semantic_fail_reasons"),
            )
            if bool(sem_bundle.get("ok")):
                sm_map = dict(sem_bundle["phase_keyframes"])
                sem_cap = cv2.VideoCapture(video_path)
                try:
                    sem_kf, sem_det = _rebind_keyframes_from_rebuilt_map(
                        sem_cap,
                        get_video_rotation(video_path),
                        float(vid_fps),
                        poses,
                        sm_map,
                        keyframe_width,
                    )
                finally:
                    sem_cap.release()
                sem_gate = validate_final_keyframes_for_ai(
                    sem_kf, sm_map, sem_det, poses=poses, fps=vid_fps,
                )
                for _pk in _REPAIR_FV_PRESERVE_KEYS:
                    if _pk in prev_fv:
                        sem_gate[_pk] = prev_fv[_pk]
                _dup_p = _adjacent_phase_keyframe_hard_dup_reasons(sem_kf, poses) if poses else []
                sem_dbg["duplicate_pairs"] = list(_dup_p)
                sem_gate = dict(sem_gate)
                sem_gate["phase_oriented_semantic_debug"] = sem_dbg
                sem_gate["duplicate_pairs"] = list(_dup_p)
                if bool(sem_gate.get("pass")):
                    out_phase.clear()
                    out_phase.update(sm_map)
                    _sync_phase_keyframes_from_keyframes(sem_kf, out_phase)
                    merged.update(
                        {
                            "details": sem_det,
                            "final_phase_keyframes": dict(out_phase),
                            "final_keyframe_validation": dict(sem_gate),
                            "final_keyframe_order_ok": bool(sem_gate.get("final_keyframe_order_ok")),
                            "final_keyframe_time_order_ok": bool(sem_gate.get("final_keyframe_time_order_ok")),
                            "final_phase_keyframes_sync_ok": bool(sem_gate.get("final_phase_keyframes_sync_ok")),
                            "final_keyframe_gate_pass": True,
                            "all_passed": True,
                            "final_validation_failed": False,
                            "final_keyframe_source": "semantic_oriented_repaired",
                            "phase_strip_repaired": True,
                            "repair_state": None,
                            "phase_oriented_semantic_debug": sem_dbg,
                            "final_phase_semantic_pass": bool(sem_dbg.get("final_phase_semantic_pass")),
                            "final_phase_semantic_fail_reasons": list(sem_dbg.get("final_phase_semantic_fail_reasons") or []),
                        }
                    )
                    logger.info(
                        "[keyframe] ensure exit source=semantic_oriented_repaired gate_pass=True out_keyframes=%d",
                        len(sem_kf or []),
                    )
                    return sem_kf, merged, out_phase, "semantic_oriented_repaired"
                if keyframe_repair_score(sem_gate, sem_kf) < keyframe_repair_score(gate, keyframes):
                    best_candidate = (
                        sem_kf,
                        dict(sm_map),
                        list(sem_det),
                        dict(sem_gate),
                        "semantic_oriented_best_effort",
                    )
        except Exception as e:
            logger.warning("[keyframe] semantic oriented phase rebuild failed: %s", e)

    # Anchor rebuild + rebind + strict re-validate (legacy lerp spacing between anchors).
    if poses and len(keyframes) == 8:
        try:
            rb = rebuild_phase_map_from_event_anchors(poses, dict(out_phase or phase_keyframes))
            if bool(rb.get("rebuild_ok")):
                rebuilt_map = dict(rb.get("phase_keyframes_rebuilt") or {})
                reb_cap = cv2.VideoCapture(video_path)
                try:
                    reb_keyframes, reb_details = _rebind_keyframes_from_rebuilt_map(
                        reb_cap,
                        get_video_rotation(video_path),
                        float(vid_fps),
                        poses,
                        rebuilt_map,
                        keyframe_width,
                    )
                finally:
                    reb_cap.release()
                reb_gate = validate_final_keyframes_for_ai(
                    reb_keyframes,
                    rebuilt_map,
                    reb_details,
                    poses=poses,
                    fps=vid_fps,
                )
                if bool(reb_gate.get("pass")):
                    merged.update(
                        {
                            "details": reb_details,
                            "final_phase_keyframes": dict(rebuilt_map),
                            "final_keyframe_validation": dict(reb_gate),
                            "final_keyframe_order_ok": bool(reb_gate.get("final_keyframe_order_ok")),
                            "final_keyframe_time_order_ok": bool(reb_gate.get("final_keyframe_time_order_ok")),
                            "final_phase_keyframes_sync_ok": bool(reb_gate.get("final_phase_keyframes_sync_ok")),
                            "final_keyframe_gate_pass": True,
                            "all_passed": True,
                            "final_validation_failed": False,
                            "final_keyframe_source": "smart_repaired",
                            "reselected_top": bool(rb.get("top_reselected")),
                            "reselected_impact": bool(rb.get("impact_reselected")),
                        }
                    )
                    logger.info(
                        "[keyframe] ensure exit source=%s gate_pass=%s out_keyframes=%d out_phase_keyframes=%d strict_reasons=%s semantic_reasons=%s",
                        "smart_repaired",
                        True,
                        len(reb_keyframes or []),
                        len(rebuilt_map or {}),
                        list((reb_gate.get("strict_contract_fail_reasons") or [])),
                        list((reb_gate.get("semantic_strip_reasons") or [])),
                    )
                    return reb_keyframes, merged, rebuilt_map, "smart_repaired"
                if keyframe_repair_score(reb_gate, reb_keyframes) < keyframe_repair_score(gate, keyframes):
                    best_candidate = (
                        reb_keyframes,
                        dict(rebuilt_map),
                        list(reb_details),
                        dict(reb_gate),
                        "smart_rebuilt_best_effort",
                    )
        except Exception as e:
            logger.warning("[keyframe] strict repair attempt failed: %s", e)

    # Quality-spacing joint rebuild when strip-quality repair no-ops (dup/time and/or post-impact semantics).
    strict_fail_q = set(gate.get("strict_contract_fail_reasons") or [])
    quality_only_strict = bool(strict_fail_q) and strict_fail_q.issubset(
        {"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"}
    )
    semantic_ok_gate = bool(gate.get("semantic_strip_ok"))
    semantic_fail_q = set(gate.get("semantic_strip_reasons") or [])
    _post_impact_semantic_q = bool(
        semantic_fail_q
        & (
            {"IMPACT_SEMANTIC_AT_KEYFRAME_FAIL", "POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"}
            | _JOINT_REPAIR_SEMANTIC_TRIGGERS
        )
    )
    order_sync_ok = (
        bool(gate.get("final_keyframe_order_ok"))
        and bool(gate.get("final_keyframe_time_order_ok"))
        and bool(gate.get("final_phase_keyframes_sync_ok"))
    )
    _quality_spacing_eligible = (
        merged.get("repair_state") == "strip_quality_noop"
        and order_sync_ok
        and poses
        and len(keyframes) == 8
        and (
            (quality_only_strict and semantic_ok_gate)
            or quality_only_strict
            or _post_impact_semantic_q
        )
    )
    if _quality_spacing_eligible:
        try:
            qm, q_mat = joint_rebuild_phase_map_for_quality_spacing(
                poses, dict(out_phase or phase_keyframes), fps=float(vid_fps),
            )
            logger.info(
                "[keyframe] quality_spacing_joint_rebuild material_change=%s tail=%s",
                bool(q_mat),
                {k: int(qm[k]) for k in ("downswing", "impact", "follow_through", "finish") if k in qm} if q_mat else {},
            )
            if bool(q_mat):
                qs_cap = cv2.VideoCapture(video_path)
                qs_kf: list[dict] = []
                qs_det: list[dict] = []
                try:
                    if qs_cap.isOpened():
                        qs_kf, qs_det = _rebind_keyframes_from_rebuilt_map(
                            qs_cap,
                            get_video_rotation(video_path),
                            float(vid_fps),
                            poses,
                            qm,
                            keyframe_width,
                        )
                finally:
                    qs_cap.release()
                if qs_kf:
                    qs_gate = validate_final_keyframes_for_ai(
                        qs_kf, qm, qs_det, poses=poses, fps=vid_fps,
                    )
                    for _pk in _REPAIR_FV_PRESERVE_KEYS:
                        if _pk in prev_fv:
                            qs_gate[_pk] = prev_fv[_pk]
                    if bool(qs_gate.get("pass")):
                        out_phase.clear()
                        out_phase.update(qm)
                        _sync_phase_keyframes_from_keyframes(qs_kf, out_phase)
                        merged.update(
                            {
                                "details": qs_det,
                                "final_phase_keyframes": dict(out_phase),
                                "final_keyframe_validation": dict(qs_gate),
                                "final_keyframe_order_ok": bool(qs_gate.get("final_keyframe_order_ok")),
                                "final_keyframe_time_order_ok": bool(qs_gate.get("final_keyframe_time_order_ok")),
                                "final_phase_keyframes_sync_ok": bool(qs_gate.get("final_phase_keyframes_sync_ok")),
                                "final_keyframe_gate_pass": True,
                                "all_passed": True,
                                "final_validation_failed": False,
                                "final_keyframe_source": "quality_spacing_repaired",
                                "phase_strip_repaired": True,
                                "_skip_strip_quality_round": False,
                                "repair_state": None,
                            }
                        )
                        logger.info(
                            "[keyframe] ensure exit source=quality_spacing_repaired gate_pass=True out_keyframes=%d",
                            len(qs_kf or []),
                        )
                        return qs_kf, merged, out_phase, "quality_spacing_repaired"
                    if keyframe_repair_score(qs_gate, qs_kf) < keyframe_repair_score(gate, keyframes):
                        best_candidate = (
                            qs_kf,
                            dict(qm),
                            list(qs_det),
                            dict(qs_gate),
                            "quality_spacing_best_effort",
                        )
                    keyframes = qs_kf
                    out_phase.clear()
                    out_phase.update(qm)
                    _sync_phase_keyframes_from_keyframes(keyframes, out_phase)
                    details = list(qs_det)
                    gate = dict(qs_gate)
                    merged["final_keyframe_validation"] = dict(gate)
                    merged["details"] = details
                    merged["final_keyframe_order_ok"] = gate["final_keyframe_order_ok"]
                    merged["final_keyframe_time_order_ok"] = gate["final_keyframe_time_order_ok"]
                    merged["final_phase_keyframes_sync_ok"] = gate["final_phase_keyframes_sync_ok"]
                    merged["final_keyframe_gate_pass"] = gate["pass"]
                    merged["all_passed"] = bool(gate["pass"])
                    merged["repair_state"] = "quality_spacing_attempted"
        except Exception as e:
            logger.warning("[keyframe] quality spacing joint rebuild failed: %s", e)

    # Forced post-impact joint rebuild when strip repair was a no-op on known failure set.
    strict_fail = set(gate.get("strict_contract_fail_reasons") or [])
    semantic_fail = set(gate.get("semantic_strip_reasons") or [])
    _dup_time_only_relaxed = (
        bool({"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"} & strict_fail)
        and strict_fail.issubset({"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"})
        and bool(gate.get("semantic_strip_ok"))
        and bool(gate.get("final_keyframe_order_ok"))
        and bool(gate.get("final_keyframe_time_order_ok"))
        and bool(gate.get("final_phase_keyframes_sync_ok"))
    )
    _forced_semantic_ok = (
        {"IMPACT_SEMANTIC_AT_KEYFRAME_FAIL", "POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"} & semantic_fail
    ) or bool(semantic_fail & _JOINT_REPAIR_SEMANTIC_TRIGGERS) or (
        merged.get("repair_state") in ("strip_quality_noop", "quality_spacing_attempted")
        and _dup_time_only_relaxed
    )
    _forced_strict_ok = bool(
        {"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"} & strict_fail
    ) or _strict_reasons_need_joint_repair(list(strict_fail))
    _strip_noop_forced_chain = (
        merged.get("repair_state") == "strip_quality_noop"
        and (
            bool(
                semantic_fail
                & (
                    {"IMPACT_SEMANTIC_AT_KEYFRAME_FAIL", "POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"}
                    | _JOINT_REPAIR_SEMANTIC_TRIGGERS
                )
            )
            or bool(strict_fail & {"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"})
        )
    )
    _run_forced_post_impact_chain = (
        not bool(merged.get("_joint_repair_ran"))
        and merged.get("repair_state") in ("strip_quality_noop", "quality_spacing_attempted")
        and poses
        and len(keyframes) == 8
        and (
            (_forced_semantic_ok and _forced_strict_ok)
            or _strip_noop_forced_chain
            or (
                merged.get("repair_state") == "quality_spacing_attempted"
                and not bool(gate.get("pass"))
            )
        )
    )
    if _run_forced_post_impact_chain:
        try:
            from services.phase_chain_solver_service import solve_post_impact_phase_chain

            chain_bundle = solve_post_impact_phase_chain(
                poses, dict(out_phase or phase_keyframes), tracks=tracks,
            )
            logger.info(
                "[keyframe] forced_joint_rebuild material_change=%s reasons=%s chain=%s",
                bool(chain_bundle.get("material_change")),
                list(chain_bundle.get("reasons") or []),
                dict(chain_bundle.get("chain") or {}),
            )
            if bool(chain_bundle.get("material_change")):
                rebuilt_map = dict(chain_bundle.get("phase_keyframes") or {})
                ch_cap = cv2.VideoCapture(video_path)
                try:
                    ch_keyframes, ch_details = _rebind_keyframes_from_rebuilt_map(
                        ch_cap,
                        get_video_rotation(video_path),
                        float(vid_fps),
                        poses,
                        rebuilt_map,
                        keyframe_width,
                    )
                finally:
                    ch_cap.release()
                ch_gate = validate_final_keyframes_for_ai(
                    ch_keyframes,
                    rebuilt_map,
                    ch_details,
                    poses=poses,
                    fps=vid_fps,
                )
                if bool(ch_gate.get("pass")):
                    merged.update(
                        {
                            "details": ch_details,
                            "final_phase_keyframes": dict(rebuilt_map),
                            "final_keyframe_validation": dict(ch_gate),
                            "final_keyframe_gate_pass": True,
                            "all_passed": True,
                            "final_validation_failed": False,
                            "final_keyframe_source": "forced_joint_rebuild",
                            "phase_strip_repaired": True,
                        }
                    )
                    return ch_keyframes, merged, rebuilt_map, "forced_joint_rebuild"
                if keyframe_repair_score(ch_gate, ch_keyframes) < keyframe_repair_score(gate, keyframes):
                    best_candidate = (
                        ch_keyframes,
                        dict(rebuilt_map),
                        list(ch_details),
                        dict(ch_gate),
                        "forced_joint_rebuild_best_effort",
                    )
                    keyframes = ch_keyframes
                    out_phase.clear()
                    out_phase.update(rebuilt_map)
                    _sync_phase_keyframes_from_keyframes(keyframes, out_phase)
                    details = list(ch_details)
                    gate = dict(ch_gate)
                    for _pk in _REPAIR_FV_PRESERVE_KEYS:
                        if _pk in prev_fv:
                            gate[_pk] = prev_fv[_pk]
                    merged["final_keyframe_validation"] = dict(gate)
                    merged["details"] = details
                    merged["final_keyframe_order_ok"] = gate["final_keyframe_order_ok"]
                    merged["final_keyframe_time_order_ok"] = gate["final_keyframe_time_order_ok"]
                    merged["final_phase_keyframes_sync_ok"] = gate["final_phase_keyframes_sync_ok"]
                    merged["final_keyframe_gate_pass"] = gate["pass"]
                    merged["all_passed"] = bool(gate["pass"])
                    merged["rebuild_used"] = True
                    merged["phase_strip_repaired"] = True
                    merged["repair_state"] = "forced_joint_best_effort_adopted"
                    merged["final_keyframe_source"] = "forced_joint_rebuild_best_effort"
        except Exception as e:
            logger.warning("[keyframe] forced joint rebuild failed: %s", e)

    # Final recovery: deterministic ordered fallback strip (monotonic-by-construction).
    fallback_seed = dict(phase_keyframes_ordered_snapshot or out_phase or phase_keyframes)
    of_kf, of_summary = extract_keyframes_ordered_fallback(
        video_path,
        poses,
        swing_phases,
        fallback_seed,
        keyframe_width=keyframe_width,
    )
    of_details = list(of_summary.get("details") or [])
    of_phase = dict(of_summary.get("final_phase_keyframes") or {})
    of_gate = validate_final_keyframes_for_ai(
        of_kf,
        of_phase,
        of_details,
        poses=poses,
        fps=vid_fps,
    )
    if bool(of_gate.get("pass")):
        merged.update(
            {
                "details": of_details,
                "final_phase_keyframes": dict(of_phase),
                "final_keyframe_validation": dict(of_gate),
                "final_keyframe_order_ok": bool(of_gate.get("final_keyframe_order_ok")),
                "final_keyframe_time_order_ok": bool(of_gate.get("final_keyframe_time_order_ok")),
                "final_phase_keyframes_sync_ok": bool(of_gate.get("final_phase_keyframes_sync_ok")),
                "final_keyframe_gate_pass": True,
                "all_passed": True,
                "final_validation_failed": False,
                "final_keyframe_source": "ordered_fallback_repaired",
            }
        )
        logger.info(
            "[keyframe] ensure exit source=%s gate_pass=%s out_keyframes=%d out_phase_keyframes=%d strict_reasons=%s semantic_reasons=%s",
            "ordered_fallback_repaired",
            True,
            len(of_kf or []),
            len(of_phase or {}),
            list((of_gate.get("strict_contract_fail_reasons") or [])),
            list((of_gate.get("semantic_strip_reasons") or [])),
        )
        return of_kf, merged, of_phase, "ordered_fallback_repaired"
    if best_candidate is not None:
        b_kf, b_phase, b_details, b_gate, b_src = best_candidate
        merged.update(
            {
                "details": b_details,
                "final_phase_keyframes": dict(b_phase),
                "final_keyframe_validation": dict(b_gate),
                "final_keyframe_order_ok": bool(b_gate.get("final_keyframe_order_ok")),
                "final_keyframe_time_order_ok": bool(b_gate.get("final_keyframe_time_order_ok")),
                "final_phase_keyframes_sync_ok": bool(b_gate.get("final_phase_keyframes_sync_ok")),
                "final_keyframe_gate_pass": bool(b_gate.get("pass", False)),
                "all_passed": bool(b_gate.get("pass", False)),
                "final_validation_failed": not bool(b_gate.get("pass", False)),
                "final_keyframe_source": b_src,
                "phase_strip_repaired": True,
            }
        )
        return b_kf, merged, b_phase, b_src

    _auth_rec = try_recover_keyframes_with_authoritative_phase_map(
        video_path,
        poses,
        swing_phases,
        keyframe_width,
        vid_fps,
        prev_fv,
        authoritative_pre_extract_phase_map,
        authoritative_chain_backend_ok,
        dict(merged),
    )
    if _auth_rec is not None:
        return _auth_rec

    merged["final_keyframe_source"] = "smart_gate_failed"
    merged["final_validation_failed"] = True
    merged["final_keyframe_gate_pass"] = False
    _pod_fail = merged.get("phase_oriented_semantic_debug")
    if isinstance(_pod_fail, dict) and _pod_fail:
        merged["final_phase_semantic_fail_reasons"] = list(
            merged.get("final_phase_semantic_fail_reasons")
            or _pod_fail.get("final_phase_semantic_fail_reasons")
            or [],
        )
        _fkv_end = dict(merged.get("final_keyframe_validation") or {})
        _fkv_end.setdefault("phase_oriented_semantic_debug", _pod_fail)
        merged["final_keyframe_validation"] = _fkv_end
    logger.warning(
        "[keyframe] ensure exit source=%s gate_pass=%s out_keyframes=%d out_phase_keyframes=%d strict_reasons=%s semantic_reasons=%s final_phase_semantic_fail_reasons=%s",
        "smart_gate_failed",
        False,
        len(keyframes or []),
        len(out_phase or {}),
        list((merged.get("final_keyframe_validation", {}).get("strict_contract_fail_reasons") or [])),
        list((merged.get("final_keyframe_validation", {}).get("semantic_strip_reasons") or [])),
        list(merged.get("final_phase_semantic_fail_reasons") or []),
    )
    return keyframes, merged, out_phase, "smart_gate_failed"


def _reselect_strip_quality_failures(
    cap: cv2.VideoCapture,
    rotation: int,
    fps: float,
    poses: list[dict],
    keyframes: list[dict],
    phase_keyframes: dict[str, int],
    keyframe_width: int,
    min_time_gap: float,
    semantic_fail_reasons: list[str] | None = None,
    strict_contract_fail_reasons: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """True repair for near-duplicate/time-close rows via local candidate re-search.

    This is not a monotonic sort/relabel pass: it actively searches new source poses/frames
    around problematic phases and rewrites keyframe images + indices.
    """
    if len(keyframes) != 8 or not poses:
        return keyframes, []

    orig_keyframes = [dict(k) for k in keyframes]
    orig_phase_keyframes = dict(phase_keyframes)
    details = recompute_keyframe_details_from_final_strip(
        cap, rotation, float(fps), poses, keyframes, min_time_gap, min_visual_diff=_DEFAULT_MIN_VISUAL_DIFF,
    )
    bad_rows = [i for i, d in enumerate(details) if i > 0 and (d.get("is_near_duplicate") or d.get("time_too_close"))]
    sem_reasons = set(semantic_fail_reasons or [])
    strict_c = set(strict_contract_fail_reasons or [])
    if {"IMPACT_SEMANTIC_AT_KEYFRAME_FAIL", "POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"} & sem_reasons:
        for pid in ("downswing", "impact", "follow_through", "finish"):
            try:
                bad_rows.append(PHASE_ORDER.index(pid))
            except ValueError:
                pass
    if {"NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"} & strict_c:
        for pid in ("downswing", "impact", "follow_through", "finish"):
            try:
                bad_rows.append(PHASE_ORDER.index(pid))
            except ValueError:
                pass
    bad_rows = sorted(set(bad_rows))
    if not bad_rows:
        return keyframes, details

    from services.swing_flow_utils import (
        _build_view_agnostic_kinematics,
        detect_phase_events_agnostic,
        validate_impact_semantic_at_index,
        validate_top_semantic_at_index,
    )

    kin = _build_view_agnostic_kinematics(poses)
    ev = detect_phase_events_agnostic(poses) if kin is not None else None
    exc_ix = int(ev.get("excursion_apex_idx", 0)) if ev else 0
    n = len(poses)

    def _gap_need(prev_phase: str, cur_phase: str) -> int:
        base = max(2, int(round(float(fps) * 0.05)))
        if (prev_phase, cur_phase) == ("top", "impact"):
            return max(base, int(round(float(fps) * 0.12)))
        if (prev_phase, cur_phase) in {("impact", "follow_through"), ("follow_through", "finish")}:
            return max(base, int(round(float(fps) * 0.10)))
        return base

    def _apply_pose_idx_to_phase_row(pid: str, new_idx: int, reason_suffix: str) -> bool:
        pidx = PHASE_ORDER.index(pid)
        if not (0 <= int(new_idx) < n):
            return False
        fi = int(poses[int(new_idx)].get("frame_index", new_idx))
        ts = float(poses[int(new_idx)].get("timestamp", round(fi / max(float(fps), 1e-6), 3)))
        frame = _read_frame_pose_matched(cap, fi, rotation)
        if frame is None:
            frame = _read_frame_with_decode_fallback(cap, fi, rotation)
        if frame is None:
            return False
        resized = _resize_frame(frame, keyframe_width)
        keyframes[pidx]["source_pose_idx"] = int(new_idx)
        keyframes[pidx]["source_frame_index"] = fi
        keyframes[pidx]["frame_index"] = fi
        keyframes[pidx]["timestamp"] = round(ts, 3)
        keyframes[pidx]["image_base64"] = frame_to_base64(resized)
        keyframes[pidx]["width"] = resized.shape[1]
        keyframes[pidx]["height"] = resized.shape[0]
        keyframes[pidx]["pose_snapshot"] = _pose_snapshot_for_keyframe(poses[int(new_idx)])
        keyframes[pidx]["selection_reason"] = (
            str(keyframes[pidx].get("selection_reason", "")) + reason_suffix
        )
        keyframes[pidx]["reselected"] = True
        phase_keyframes[pid] = int(new_idx)
        return True

    for row_idx in bad_rows:
        phase_id = str(keyframes[row_idx].get("phase"))
        cur_spi = int(keyframes[row_idx].get("source_pose_idx", -1))
        if cur_spi < 0 or cur_spi >= n:
            continue
        prev_kf = keyframes[row_idx - 1]
        next_kf = keyframes[row_idx + 1] if row_idx + 1 < len(keyframes) else None

        prev_fi = int(prev_kf.get("source_frame_index", prev_kf.get("frame_index", -1)))
        prev_ts = float(prev_kf.get("timestamp", -999.0))
        min_gap_fi = _gap_need(str(prev_kf.get("phase")), phase_id)
        next_fi_limit: int | None = None
        if next_kf is not None:
            next_phase = str(next_kf.get("phase"))
            gap_to_next = _gap_need(phase_id, next_phase)
            next_fi_limit = int(next_kf.get("source_frame_index", next_kf.get("frame_index", 10**9))) - gap_to_next

        win = max(24, min(72, n // 3))
        lo = max(0, cur_spi - win)
        hi = min(n - 1, cur_spi + win)
        best: dict[str, Any] | None = None

        prev_frames: list[np.ndarray] = []
        prev_poses: list[dict] = []
        for pidx in range(0, row_idx):
            pk = keyframes[pidx]
            pfi = int(pk.get("source_frame_index", pk.get("frame_index", -1)))
            pspi = int(pk.get("source_pose_idx", -1))
            if pfi < 0 or not (0 <= pspi < n):
                continue
            pframe = _read_frame_pose_matched(cap, pfi, rotation)
            if pframe is None:
                pframe = _read_frame_with_decode_fallback(cap, pfi, rotation)
            if pframe is None:
                continue
            prev_frames.append(pframe)
            prev_poses.append(poses[pspi])
        next_frame = None
        next_pose = None
        if next_kf is not None:
            next_fi = int(next_kf.get("source_frame_index", next_kf.get("frame_index", -1)))
            next_spi = int(next_kf.get("source_pose_idx", -1))
            if next_fi >= 0 and 0 <= next_spi < n:
                next_frame = _read_frame_pose_matched(cap, next_fi, rotation)
                if next_frame is None:
                    next_frame = _read_frame_with_decode_fallback(cap, next_fi, rotation)
                if next_frame is not None:
                    next_pose = poses[next_spi]

        for cand_pi in range(lo, hi + 1):
            pose = poses[cand_pi]
            cand_fi = int(pose.get("frame_index", cand_pi))
            cand_ts = float(pose.get("timestamp", round(cand_fi / max(float(fps), 1e-6), 3)))
            if cand_fi <= prev_fi + min_gap_fi:
                continue
            if next_fi_limit is not None and cand_fi >= next_fi_limit:
                continue
            if cand_ts < _min_timestamp_after_prev(prev_ts, min_time_gap):
                continue
            if phase_id == "top" and kin is not None:
                ok_top, _ = validate_top_semantic_at_index(cand_pi, kin)
                if not ok_top:
                    continue
            if phase_id == "impact" and kin is not None:
                top_pi = int(phase_keyframes.get("top", cur_spi))
                ok_imp, checks = validate_impact_semantic_at_index(cand_pi, top_pi, exc_ix, kin)
                if not ok_imp or not checks.get("unwinding", False):
                    continue

            frame = _read_frame_pose_matched(cap, cand_fi, rotation)
            if frame is None:
                frame = _read_frame_with_decode_fallback(cap, cand_fi, rotation)
            if frame is None:
                continue
            worst = 0.0
            if prev_frames:
                worst = _worst_pose_gated_histogram_similarity(frame, pose, prev_frames, prev_poses)
            next_sim = 0.0
            if next_frame is not None and next_pose is not None:
                if _pose_angle_distance(pose, next_pose) <= _POSE_GATE_HIST_DUP:
                    next_sim = _visual_similarity(frame, next_frame)
            near_dup = bool(worst > _VISUAL_DIFF_THRESHOLD)
            time_close = bool(cand_ts - prev_ts < min_time_gap)
            if next_kf is not None:
                nts = float(next_kf.get("timestamp", 1e9))
                if nts - cand_ts < min_time_gap:
                    time_close = True
            if near_dup or time_close:
                continue
            if next_sim > _VISUAL_DIFF_THRESHOLD:
                continue
            semantic_bonus = 0.0
            if phase_id == "top" and ev is not None:
                semantic_bonus = -0.02 * abs(cand_pi - int(ev.get("top_pose_idx", cand_pi)))
            elif phase_id == "impact" and ev is not None:
                semantic_bonus = -0.02 * abs(cand_pi - int(ev.get("impact_pose_idx", cand_pi)))
            score = (1.0 - worst) + (1.0 - next_sim) * 0.65 + semantic_bonus
            if best is None or score > float(best["score"]):
                best = {"pi": cand_pi, "fi": cand_fi, "ts": cand_ts, "frame": frame, "score": score}

        if best is None:
            continue

        _apply_pose_idx_to_phase_row(phase_id, int(best["pi"]), "_quality_research")

    _sync_phase_keyframes_from_keyframes(keyframes, phase_keyframes)
    details = recompute_keyframe_details_from_final_strip(
        cap, rotation, float(fps), poses, keyframes, min_time_gap, min_visual_diff=_DEFAULT_MIN_VISUAL_DIFF,
    )
    pre_sem_spi = [int(k.get("source_pose_idx", -1)) for k in keyframes]
    if {"IMPACT_SEMANTIC_AT_KEYFRAME_FAIL", "POST_IMPACT_SPEED_NOT_BELOW_IMPACT_MEDIAN"} & sem_reasons:
        from services.swing_flow_utils import propose_post_impact_chain_indices

        rebuilt_chain = propose_post_impact_chain_indices(poses, phase_keyframes)
        if rebuilt_chain:
            for pid, idx in rebuilt_chain.items():
                _apply_pose_idx_to_phase_row(pid, int(idx), "_semantic_chain_rebuild")
            _sync_phase_keyframes_from_keyframes(keyframes, phase_keyframes)
            details = recompute_keyframe_details_from_final_strip(
                cap, rotation, float(fps), poses, keyframes, min_time_gap, min_visual_diff=_DEFAULT_MIN_VISUAL_DIFF,
            )
            post_sem_spi = [int(k.get("source_pose_idx", -1)) for k in keyframes]
            if post_sem_spi == pre_sem_spi:
                phase_keyframes.clear()
                phase_keyframes.update(orig_phase_keyframes)
                return orig_keyframes, recompute_keyframe_details_from_final_strip(
                    cap,
                    rotation,
                    float(fps),
                    poses,
                    orig_keyframes,
                    min_time_gap,
                    min_visual_diff=_DEFAULT_MIN_VISUAL_DIFF,
                )
    contract_ok, _reasons = _keyframe_atomic_contract_ok(keyframes, float(fps), float(min_time_gap))
    if not contract_ok:
        phase_keyframes.clear()
        phase_keyframes.update(orig_phase_keyframes)
        return orig_keyframes, recompute_keyframe_details_from_final_strip(
            cap,
            rotation,
            float(fps),
            poses,
            orig_keyframes,
            min_time_gap,
            min_visual_diff=_DEFAULT_MIN_VISUAL_DIFF,
        )
    return keyframes, details


def extract_keyframes_smart(
    video_path: str,
    poses: list[dict],
    swing_phases: list[dict],
    phase_keyframes: dict[str, int],
    keyframe_width: int = 320,
) -> tuple[list[dict], dict]:
    """
    Pose-aware keyframe extraction with reliable frame reading.

    For each of the 8 swing phases, reads the exact video frame identified by
    swing phase detection.  Uses sequential-read approach to avoid codec
    seeking inaccuracy (CAP_PROP_POS_FRAMES can miss by several frames on
    H.264/H.265 compressed video).

    IMPORTANT: This function UPDATES phase_keyframes in-place when an
    alternative frame is used due to decode failure, keeping the mapping
    consistent with the actual JPEG images returned.

    Returns (keyframes, keyframe_validation_summary).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video for smart keyframes, falling back to temporal")
        kf = _extract_temporal_keyframes(video_path, 8, keyframe_width)
        fv = validate_final_keyframes_for_ai(kf, {}, [])
        return kf, {
            "total_keyframes": len(kf),
            "near_duplicates": 0,
            "time_too_close": 0,
            "all_passed": True,
            "details": [],
            "final_keyframe_order_ok": fv.get("final_keyframe_order_ok"),
            "final_keyframe_time_order_ok": fv.get("final_keyframe_time_order_ok"),
            "final_validation_failed": not fv.get("pass", False),
            "final_phase_keyframes": {},
            "final_keyframe_validation": fv,
            "final_keyframe_gate_pass": fv.get("pass", False),
        }

    rotation = get_video_rotation(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    from services.swing_flow_utils import (
        detect_phase_events_agnostic,
        refine_phase_keyframes_top_impact,
        validate_impact_semantic_at_index,
        validate_top_semantic_at_index,
        _build_view_agnostic_kinematics,
    )

    refine_phase_keyframes_top_impact(poses, phase_keyframes)
    kin_ctx = _build_view_agnostic_kinematics(poses)
    ev_ctx = detect_phase_events_agnostic(poses) if kin_ctx is not None else None
    exc_ctx = int(ev_ctx["excursion_apex_idx"]) if ev_ctx else 0

    # Build phase buckets for candidate expansion / local fallback.
    phase_buckets: dict[str, list[int]] = {}
    for i, sp in enumerate(swing_phases):
        phase_buckets.setdefault(sp.get("phase_id", ""), []).append(i)
    selected: dict[str, dict] = {}
    used_pose_indices: set[int] = set()  # Track indices already selected by other phases
    min_pg = _min_pose_index_gap(len(poses))
    total_duration_early = max(
        float(poses[-1].get("timestamp", 0)) - float(poses[0].get("timestamp", 0)),
        0.1,
    ) if len(poses) >= 2 else 0.1
    min_time_gap_early = max(total_duration_early * _MIN_TIME_INTERVAL_RATIO, 0.05)
    strip_acc_frames: list[np.ndarray] = []
    strip_acc_poses: list[dict] = []
    strip_prev_fi = -1
    strip_prev_ts = -999.0

    # Candidate scoring in each phase bucket: action-selected pose + image quality.
    for phase_id in PHASE_ORDER:
        preferred_idx = phase_keyframes.get(phase_id)
        bucket = [i for i in phase_buckets.get(phase_id, []) if 0 <= i < len(poses)]

        candidates: list[int] = []
        if isinstance(preferred_idx, int) and 0 <= preferred_idx < len(poses):
            candidates.append(preferred_idx)
        if bucket:
            if candidates:
                bucket = sorted(bucket, key=lambda i: abs(i - candidates[0]))
            candidates.extend(bucket[:10])
        candidates = list(dict.fromkeys(candidates))

        best = None
        best_score = -1.0

        # Prefer the mapped pose (Gemini / kinematic) when it decodes and is still free.
        # Scoring bucket neighbors by sharpness often replaces the correct phase moment.
        if (
            isinstance(preferred_idx, int)
            and 0 <= preferred_idx < len(poses)
            and _pose_idx_allowed_for_phase(
                preferred_idx, selected, phase_id, len(poses), used_pose_indices, min_pg
            )
        ):
            pfi = int(poses[preferred_idx].get("frame_index", 0))
            pframe = _read_frame_pose_matched(cap, pfi, rotation)
            if pframe is not None:
                pq = _pose_quality_details(poses[preferred_idx])
                iq = _frame_quality_details(pframe)
                conf = float(np.clip(0.35 * pq["quality"] + 0.25 * iq["quality"] + 0.40, 0.0, 1.0))
                if kin_ctx is not None:
                    if phase_id == "top":
                        ok_t, _ = validate_top_semantic_at_index(preferred_idx, kin_ctx)
                        if ok_t:
                            conf = min(1.0, conf + 0.22)
                    elif phase_id == "impact":
                        top_pi = int(phase_keyframes.get("top", 0))
                        ok_i, _ = validate_impact_semantic_at_index(
                            preferred_idx, top_pi, exc_ctx, kin_ctx,
                        )
                        if ok_i:
                            conf = min(1.0, conf + 0.24)
                if phase_id in _EARLY_STRIP_MOTION_PHASES:
                    _ix = PHASE_ORDER.index(phase_id)
                    if _ix > 0:
                        _pp = PHASE_ORDER[_ix - 1]
                        if _pp in selected:
                            conf = min(
                                1.0,
                                conf + 0.18 * _pose_angle_distance(poses[preferred_idx], poses[selected[_pp]["pose_idx"]]),
                            )
                best = {
                    "pose_idx": preferred_idx,
                    "frame": pframe,
                    "confidence": conf,
                    "fallback_used": False,
                    "selection_reason": _phase_reason(phase_id, False),
                }

        if best is None:
            for pose_idx in candidates:
                if not _pose_idx_allowed_for_phase(
                    pose_idx, selected, phase_id, len(poses), used_pose_indices, min_pg
                ):
                    continue
                frame_idx = poses[pose_idx].get("frame_index", 0)
                frame = _read_frame_pose_matched(cap, frame_idx, rotation)
                if frame is None:
                    continue
                pose_q = _pose_quality_details(poses[pose_idx])
                img_q = _frame_quality_details(frame)
                near_pref = 1.0
                if isinstance(preferred_idx, int):
                    near_pref = 1.0 - min(abs(pose_idx - preferred_idx) / max(len(bucket) + 1, 3), 1.0)
                score = 0.35 * pose_q["quality"] + 0.25 * img_q["quality"] + 0.40 * near_pref
                if kin_ctx is not None:
                    if phase_id == "top":
                        ok_t, _ = validate_top_semantic_at_index(pose_idx, kin_ctx)
                        if ok_t:
                            score += 0.20
                    elif phase_id == "impact":
                        top_pi = int(phase_keyframes.get("top", 0))
                        ok_i, _ = validate_impact_semantic_at_index(
                            pose_idx, top_pi, exc_ctx, kin_ctx,
                        )
                        if ok_i:
                            score += 0.22
                if pose_idx in used_pose_indices:
                    score *= 0.1
                if phase_id in _EARLY_STRIP_MOTION_PHASES:
                    _ix = PHASE_ORDER.index(phase_id)
                    if _ix > 0:
                        _pp = PHASE_ORDER[_ix - 1]
                        if _pp in selected:
                            score += 0.18 * _pose_angle_distance(poses[pose_idx], poses[selected[_pp]["pose_idx"]])
                if score > best_score:
                    best_score = score
                    best = {
                        "pose_idx": pose_idx,
                        "frame": frame,
                        "confidence": float(np.clip(score, 0.0, 1.0)),
                        "fallback_used": isinstance(preferred_idx, int) and pose_idx != preferred_idx,
                        "selection_reason": _phase_reason(
                            phase_id,
                            isinstance(preferred_idx, int) and pose_idx != preferred_idx,
                        ),
                    }
        if best:
            best = _legalize_phase_pick_for_strip(
                cap,
                rotation,
                poses,
                phase_id,
                best,
                candidates,
                preferred_idx if isinstance(preferred_idx, int) else None,
                bucket,
                selected,
                used_pose_indices,
                min_pg,
                strip_prev_fi,
                strip_prev_ts,
                min_time_gap_early,
                strip_acc_frames,
                strip_acc_poses,
                kin_ctx,
                exc_ctx,
                phase_keyframes,
            )
        if best:
            selected[phase_id] = best
            phase_keyframes[phase_id] = best["pose_idx"]
            used_pose_indices.add(best["pose_idx"])
            nfi, nts = _commit_strip_accumulator(best, poses, float(fps), strip_acc_frames, strip_acc_poses)
            strip_prev_fi, strip_prev_ts = nfi, nts

    # Missing phase handling: neighbor window search first, then explicit low-confidence fallback.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for phase_id in PHASE_ORDER:
        if phase_id in selected:
            continue
        phase_pos = PHASE_ORDER.index(phase_id)
        neighbor_pose_idx = None
        for step in range(1, len(PHASE_ORDER)):
            left = phase_pos - step
            right = phase_pos + step
            if left >= 0 and PHASE_ORDER[left] in selected:
                neighbor_pose_idx = selected[PHASE_ORDER[left]]["pose_idx"]
                break
            if right < len(PHASE_ORDER) and PHASE_ORDER[right] in selected:
                neighbor_pose_idx = selected[PHASE_ORDER[right]]["pose_idx"]
                break

        recovered = None
        if neighbor_pose_idx is not None:
            scan_window = range(max(0, neighbor_pose_idx - 6), min(len(poses), neighbor_pose_idx + 7))
            bucket_set = set(phase_buckets.get(phase_id, []))
            scan_list = sorted(
                scan_window,
                key=lambda i: (0 if i in bucket_set else 1, abs(i - neighbor_pose_idx)),
            )

            def _best_in_pose_list(idxs: list[int]):
                bl = None
                bs = -1.0
                for pose_idx in idxs:
                    frame_idx = poses[pose_idx].get("frame_index", 0)
                    frame = _read_frame_pose_matched(cap, frame_idx, rotation)
                    if frame is None:
                        continue
                    pose_q = _pose_quality_details(poses[pose_idx])
                    img_q = _frame_quality_details(frame)
                    score = 0.55 * pose_q["quality"] + 0.45 * img_q["quality"]
                    if score > bs:
                        bs = score
                        bl = (pose_idx, frame, score)
                return bl

            def _filter_scan_for_bounds(idxs: list[int], gap_min: int) -> list[int]:
                return [
                    i for i in idxs
                    if _pose_idx_allowed_for_phase(
                        i, selected, phase_id, len(poses), used_pose_indices, gap_min
                    )
                ]

            for gap_try in (min_pg, 1):
                filtered_scan = _filter_scan_for_bounds(list(scan_list), gap_try)
                in_window_bucket = [i for i in filtered_scan if i in bucket_set]
                best_local = _best_in_pose_list(in_window_bucket)
                if best_local is None:
                    best_local = _best_in_pose_list(filtered_scan)
                if best_local is not None:
                    recovered = {
                        "pose_idx": best_local[0],
                        "frame": best_local[1],
                        "confidence": float(np.clip(best_local[2] * 0.75, 0.0, 1.0)),
                        "fallback_used": True,
                        "selection_reason": "neighbor_window_recovery_low_confidence",
                    }
                    break

        if recovered is None:
            # Last-resort: search temporal target for unused pose inside (lo, hi), never reuse indices.
            if total_frames <= 1:
                continue
            ratio = phase_pos / max(len(PHASE_ORDER) - 1, 1)
            fallback_frame_idx = int(ratio * (total_frames - 1))
            lo_b = _prior_selected_pose_lower(selected, phase_id)
            hi_b = _later_selected_pose_upper(selected, phase_id, len(poses))
            order_pi = sorted(
                range(len(poses)),
                key=lambda i: abs(int(poses[i].get("frame_index", 0)) - fallback_frame_idx),
            )
            found = None
            for gap_try in (min_pg, 1):
                for pi in order_pi:
                    if not _pose_idx_allowed_for_phase(
                        pi, selected, phase_id, len(poses), used_pose_indices, gap_try
                    ):
                        continue
                    fi = int(poses[pi].get("frame_index", 0))
                    frame = _read_frame_with_decode_fallback(cap, fi, rotation)
                    if frame is None:
                        continue
                    found = (pi, frame, fi, gap_try)
                    break
                if found:
                    break
            if found is None:
                for pi in order_pi:
                    if pi in used_pose_indices:
                        continue
                    if not (lo_b < pi < hi_b):
                        continue
                    fi = int(poses[pi].get("frame_index", 0))
                    frame = _read_frame_with_decode_fallback(cap, fi, rotation)
                    if frame is None:
                        continue
                    found = (pi, frame, fi, 0)
                    break
            if found is None:
                logger.warning("[keyframe] %s: no legal temporal fallback pose", phase_id)
                continue
            pi, frame, fi_used, gtry = found
            ts = float(poses[pi].get("timestamp", round(fi_used / max(fps, 1e-6), 3)))
            reason = "last_resort_temporal_fallback_explicit"
            if gtry == 0:
                reason = "last_resort_temporal_fallback_break_glass"
            recovered = {
                "pose_idx": pi,
                "frame": frame,
                "confidence": 0.22 if gtry == 0 else 0.28,
                "fallback_used": True,
                "selection_reason": reason,
            }

        if recovered is None:
            continue
        rec_cand = sorted(
            set(range(len(poses))) | set(phase_buckets.get(phase_id, [])),
        )
        recovered = _legalize_phase_pick_for_strip(
            cap,
            rotation,
            poses,
            phase_id,
            recovered,
            rec_cand,
            int(recovered["pose_idx"]),
            phase_buckets.get(phase_id, []),
            selected,
            used_pose_indices,
            min_pg,
            strip_prev_fi,
            strip_prev_ts,
            min_time_gap_early,
            strip_acc_frames,
            strip_acc_poses,
            kin_ctx,
            exc_ctx,
            phase_keyframes,
        )
        if recovered is None:
            continue
        selected[phase_id] = recovered
        phase_keyframes[phase_id] = recovered["pose_idx"]
        used_pose_indices.add(int(recovered["pose_idx"]))
        nfi, nts = _commit_strip_accumulator(recovered, poses, float(fps), strip_acc_frames, strip_acc_poses)
        strip_prev_fi, strip_prev_ts = nfi, nts

    total_frames_vf = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    _apply_raw_timeline_refine_to_smart_selection(
        cap, rotation, fps, poses, phase_keyframes, selected, total_frames_vf,
    )

    # Keep cap open through assembly / re-selection (needs fresh reads for repairs)

    # ── Near-duplicate detection and minimum time interval enforcement ──
    total_duration = 0.1
    if poses:
        total_duration = max(
            float(poses[-1].get("timestamp", 0)) - float(poses[0].get("timestamp", 0)),
            0.1,
        )
    min_time_gap = max(total_duration * _MIN_TIME_INTERVAL_RATIO, 0.05)

    prev_frame = None
    prev_timestamp = -999.0
    prev_pose_idx_out = -1
    prev_frame_index_out = -1
    keyframe_validation: list[dict] = []
    all_accepted_frames: list[np.ndarray] = []  # Track ALL accepted keyframe frames
    all_accepted_poses: list[dict] = []  # Track ALL accepted keyframe poses

    keyframes: list[dict] = []
    for phase_id in PHASE_ORDER:
        item = selected.get(phase_id)
        if not item:
            continue
        pose_idx = item["pose_idx"]
        frame = item["frame"]
        pose = poses[pose_idx]
        meta = SWING_PHASE_META[phase_id]
        frame_index = int(pose.get("frame_index", 0))
        timestamp = float(pose.get("timestamp", round(frame_index / fps, 3)))

        # Pose-gated histogram: ignore sky/grass-only matches when body pose already changed
        visual_diff = 0.0
        is_near_duplicate = False
        worst_similarity = 0.0
        if all_accepted_frames:
            worst_similarity = _worst_pose_gated_histogram_similarity(
                frame, pose, all_accepted_frames, all_accepted_poses
            )
            visual_diff = round(1.0 - worst_similarity, 4)
            is_near_duplicate = worst_similarity > _VISUAL_DIFF_THRESHOLD

        # Time interval check
        time_gap = timestamp - prev_timestamp if prev_timestamp >= 0 else 999.0
        time_too_close = time_gap < min_time_gap and prev_timestamp >= 0

        strip_viol = item.get("strip_constraint_violation")
        validation_passed = not is_near_duplicate and not time_too_close
        if strip_viol:
            validation_passed = False
            sv = str(strip_viol)
            if sv == "NEAR_DUPLICATE_CANDIDATE":
                is_near_duplicate = True
            if sv == "TIME_TOO_CLOSE_CANDIDATE":
                time_too_close = True
        vrow: dict[str, Any] = {
            "phase": phase_id,
            "visual_diff_from_prev": visual_diff,
            "is_near_duplicate": is_near_duplicate,
            "time_gap": round(time_gap, 3),
            "time_too_close": time_too_close,
            "validation_passed": validation_passed,
        }
        if strip_viol:
            vrow["fail_code"] = _STRIP_VIOL_TO_FAIL_CODE.get(str(strip_viol), str(strip_viol))
        keyframe_validation.append(vrow)

        reselected = False
        if is_near_duplicate or time_too_close:
            logger.warning(
                "Keyframe %s near-dup=%s time_close=%s (diff=%.3f gap=%.3fs min=%.3fs) — attempting re-select",
                phase_id, is_near_duplicate, time_too_close, visual_diff, time_gap, min_time_gap,
            )
            repaired = _reselect_distinct_keyframe(
                cap,
                rotation,
                fps,
                poses,
                phase_id,
                pose_idx,
                all_accepted_frames,
                all_accepted_poses,
                prev_timestamp,
                min_time_gap,
                total_frames,
                used_pose_indices=used_pose_indices,
                prev_pose_idx=prev_pose_idx_out,
                prev_frame_index=prev_frame_index_out,
            )
            if repaired is not None:
                new_frame, new_pi, new_fi, new_ts = repaired
                if not _reselect_temporal_pose_order_ok(
                    new_ts,
                    new_pi,
                    new_fi,
                    prev_timestamp,
                    prev_pose_idx_out,
                    prev_frame_index_out,
                    min_time_gap,
                ):
                    repaired = None
                elif int(new_fi) == int(frame_index):
                    # A "reselect" that keeps the same source frame is not a real repair.
                    repaired = None
            if repaired is not None:
                new_frame, new_pi, new_fi, new_ts = repaired
                frame = new_frame
                pose_idx = new_pi
                pose = poses[pose_idx]
                frame_index = new_fi
                timestamp = new_ts
                reselected = True
                phase_keyframes[phase_id] = pose_idx
                used_pose_indices.add(pose_idx)
                selected[phase_id] = {
                    "pose_idx": pose_idx,
                    "frame": frame,
                    "confidence": float(np.clip(float(item["confidence"]) * 0.92, 0.15, 0.99)),
                    "fallback_used": True,
                    "selection_reason": (item.get("selection_reason") or "phase_event_selection")
                    + "_reselected_distinct",
                }
                item = selected[phase_id]
                # Recompute validation vs ALL previous output frames (pose-gated)
                worst_similarity = 0.0
                if all_accepted_frames:
                    worst_similarity = _worst_pose_gated_histogram_similarity(
                        frame, pose, all_accepted_frames, all_accepted_poses
                    )
                    visual_diff = round(1.0 - worst_similarity, 4)
                    is_near_duplicate = worst_similarity > _VISUAL_DIFF_THRESHOLD
                else:
                    visual_diff = 1.0
                    is_near_duplicate = False
                time_gap = timestamp - prev_timestamp if prev_timestamp >= 0 else 999.0
                time_too_close = time_gap < min_time_gap and prev_timestamp >= 0
                validation_passed = not is_near_duplicate and not time_too_close
                keyframe_validation[-1].update({
                    "visual_diff_from_prev": visual_diff,
                    "is_near_duplicate": is_near_duplicate,
                    "time_gap": round(time_gap, 3),
                    "time_too_close": time_too_close,
                    "validation_passed": validation_passed,
                    "reselected": True,
                })
            else:
                keyframe_validation[-1]["reselected"] = False

        resized = _resize_frame(frame, keyframe_width)
        pose_snapshot = _pose_snapshot_for_keyframe(pose)

        kf_entry: dict = {
            "phase": phase_id,
            "label_en": meta["label_en"],
            "label_zh": meta["label_zh"],
            "frame_index": frame_index,
            "timestamp": round(timestamp, 3),
            "confidence": round(float(item["confidence"]), 3),
            "selection_reason": item["selection_reason"],
            "fallback_used": bool(item["fallback_used"]),
            "image_base64": frame_to_base64(resized),
            "width": resized.shape[1],
            "height": resized.shape[0],
            "pose_snapshot": pose_snapshot,
            "source_pose_idx": pose_idx,
            "source_frame_index": frame_index,
            "visual_diff_from_prev": visual_diff,
            "phase_validation_passed": validation_passed,
        }
        if reselected:
            kf_entry["reselected"] = True
        keyframes.append(kf_entry)

        prev_frame = frame
        prev_timestamp = timestamp
        prev_pose_idx_out = pose_idx
        prev_frame_index_out = frame_index
        all_accepted_frames.append(frame)
        all_accepted_poses.append(pose)

    _fo, _ft, _fvf = _finalize_smart_keyframes_monotonic(
        keyframes,
        keyframe_validation,
        poses,
        phase_keyframes,
        selected,
        cap,
        rotation,
        fps,
        keyframe_width,
        min_time_gap,
    )
    _sync_phase_keyframes_from_keyframes(keyframes, phase_keyframes)

    repair_extra: dict[str, Any] = {}
    if len(keyframes) == 8 and poses:
        pre_details = recompute_keyframe_details_from_final_strip(
            cap, rotation, float(fps), poses, keyframes, min_time_gap, min_visual_diff=_DEFAULT_MIN_VISUAL_DIFF,
        )
        pre_dup = sum(1 for d in pre_details if d.get("is_near_duplicate"))
        pre_close = sum(1 for d in pre_details if d.get("time_too_close"))
        if pre_dup > 0 or pre_close > 0:
            old_spi = [int(k.get("source_pose_idx", -1)) for k in keyframes]
            trial_keyframes = [dict(k) for k in keyframes]
            trial_phase = dict(phase_keyframes)
            trial_keyframes, trial_details = _reselect_strip_quality_failures(
                cap,
                rotation,
                float(fps),
                poses,
                trial_keyframes,
                trial_phase,
                keyframe_width,
                min_time_gap,
                semantic_fail_reasons=[],
                strict_contract_fail_reasons=["NEAR_DUPLICATE_PRESENT", "TIME_TOO_CLOSE_PRESENT"],
            )
            post_dup = sum(1 for d in trial_details if d.get("is_near_duplicate"))
            post_close = sum(1 for d in trial_details if d.get("time_too_close"))
            trial_contract_ok, _trial_contract_reasons = _keyframe_atomic_contract_ok(
                trial_keyframes, float(fps), float(min_time_gap)
            )
            accept_repair = trial_contract_ok and (post_dup, post_close) < (pre_dup, pre_close)
            if accept_repair:
                keyframes = trial_keyframes
                phase_keyframes.clear()
                phase_keyframes.update(trial_phase)
                keyframe_validation = trial_details
                new_spi = [int(k.get("source_pose_idx", -1)) for k in keyframes]
                repair_extra = {
                    "phase_strip_repaired": old_spi != new_spi,
                    "reselected_top": bool(old_spi[PHASE_ORDER.index("top")] != new_spi[PHASE_ORDER.index("top")]),
                    "reselected_impact": bool(old_spi[PHASE_ORDER.index("impact")] != new_spi[PHASE_ORDER.index("impact")]),
                    "top_reselected": bool(old_spi[PHASE_ORDER.index("top")] != new_spi[PHASE_ORDER.index("top")]),
                    "impact_reselected": bool(old_spi[PHASE_ORDER.index("impact")] != new_spi[PHASE_ORDER.index("impact")]),
                    "enforce_ok": True,
                    "enforce_fail_reasons": [],
                    "relabel_count": 0,
                    "rebuild_used": False,
                    "rebuild_reasons": [],
                }
            else:
                keyframe_validation = pre_details

    if keyframes:
        keyframe_validation = recompute_keyframe_details_from_final_strip(
            cap,
            rotation,
            float(fps),
            poses,
            keyframes,
            min_time_gap,
            min_visual_diff=_DEFAULT_MIN_VISUAL_DIFF,
        )

    relabel_for_validate = int(repair_extra.get("relabel_count") or 0) if repair_extra else 0

    # Aggregate validation after monotonic finalize + optional anchor rebind
    n_duplicates = sum(1 for v in keyframe_validation if v["is_near_duplicate"])
    n_time_close = sum(1 for v in keyframe_validation if v["time_too_close"])
    fi_strict, ts_strict = _strict_increasing_ts_and_fi(keyframes)
    fv_merged = _merge_keyframe_validation_with_repairs(
        validate_final_keyframes_for_ai(
            keyframes,
            phase_keyframes,
            keyframe_validation,
            poses=poses,
            relabel_count=relabel_for_validate,
            fps=float(fps),
        ),
        repair_extra,
    )

    strip_relaxed = False
    _sch0 = dict(fv_merged.get("strict_contract_checks") or {})
    _struct_ok = bool(
        _sch0.get("phase_count_ok")
        and _sch0.get("phase_sequence_ok")
        and _sch0.get("unique_source_pose_ok")
        and _sch0.get("final_keyframe_order_ok")
        and _sch0.get("final_keyframe_time_order_ok")
        and _sch0.get("final_phase_keyframes_sync_ok")
        and _sch0.get("no_negative_time_gap")
        and _sch0.get("adjacent_strip_ok")
        and _sch0.get("source_frame_gaps_ok")
    )
    _strip_ok = bool(
        _sch0.get("near_duplicates_ok")
        and _sch0.get("time_too_close_ok")
        and _sch0.get("details_all_passed")
    )
    if keyframes and _struct_ok and not _strip_ok and not bool(fv_merged.get("pass")):
        kv2 = recompute_keyframe_details_from_final_strip(
            cap,
            rotation,
            float(fps),
            poses,
            keyframes,
            min_time_gap,
            min_visual_diff=_STRIP_QUALITY_RELAX_MIN_VISUAL_DIFF,
            min_time_gap_factor=_STRIP_QUALITY_RELAX_TIME_GAP_FACTOR,
        )
        fv2 = _merge_keyframe_validation_with_repairs(
            validate_final_keyframes_for_ai(
                keyframes,
                phase_keyframes,
                kv2,
                poses=poses,
                relabel_count=relabel_for_validate,
                fps=float(fps),
            ),
            repair_extra,
        )
        if bool(fv2.get("pass")):
            keyframe_validation = kv2
            fv_merged = fv2
            n_duplicates = sum(1 for v in keyframe_validation if v["is_near_duplicate"])
            n_time_close = sum(1 for v in keyframe_validation if v["time_too_close"])
            fi_strict, ts_strict = _strict_increasing_ts_and_fi(keyframes)
            strip_relaxed = True

    kf_validation_summary = {
        "total_keyframes": len(keyframes),
        "near_duplicates": n_duplicates,
        "time_too_close": n_time_close,
        "all_passed": bool(fv_merged.get("pass")),
        "details": keyframe_validation,
        "final_keyframe_order_ok": fi_strict,
        "final_keyframe_time_order_ok": ts_strict,
        "final_validation_failed": not bool(fv_merged.get("pass")),
        "final_phase_keyframes": {p: int(phase_keyframes[p]) for p in PHASE_ORDER if p in phase_keyframes},
        "final_keyframe_validation": fv_merged,
        "strip_quality_relaxed_second_pass": strip_relaxed,
    }
    kf_validation_summary["final_keyframe_gate_pass"] = bool(
        kf_validation_summary.get("final_keyframe_validation", {}).get("pass")
    )
    _ch = dict(fv_merged.get("strict_contract_checks") or {})
    kf_validation_summary["keyframe_quality_repair_attempted"] = True
    kf_validation_summary["keyframe_quality_repair_success"] = bool(
        _ch.get("near_duplicates_ok", False)
        and _ch.get("time_too_close_ok", False)
        and _ch.get("details_all_passed", False)
    )

    logger.info(
        "Smart keyframes: %d phases captured (dups=%d, time_close=%d) — [%s]",
        len(keyframes), n_duplicates, n_time_close,
        ", ".join(kf["phase"] for kf in keyframes),
    )
    cap.release()
    return keyframes, kf_validation_summary


def _extract_temporal_keyframes(
    video_path: str, num_keyframes: int = 8, keyframe_width: int = 320
) -> list[dict]:
    """Fallback: fixed-position temporal keyframes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    rotation = get_video_rotation(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    phases = SWING_PHASES[:num_keyframes]
    keyframes = []
    for phase in phases:
        frame_idx = int(phase["position"] * (total_frames - 1))
        frame = read_frame_pose_pipeline(cap, frame_idx, rotation)
        if frame is None:
            continue
        resized = _resize_frame(frame, keyframe_width)
        keyframes.append({
            "phase": phase["name"],
            "label_en": phase["label_en"],
            "label_zh": phase["label_zh"],
            "frame_index": frame_idx,
            "timestamp": round(frame_idx / fps, 3),
            "image_base64": frame_to_base64(resized),
            "width": resized.shape[1],
            "height": resized.shape[0],
        })

    cap.release()
    return keyframes


def extract_keyframes_and_ai_frames(
    video_path: str,
    num_keyframes: int = 8,
    num_ai_frames: int = 8,
    keyframe_width: int = 320,
    ai_frame_width: int = 384,
) -> tuple[list[dict], list[str], list[int]]:
    """
    Single-pass extraction of temporal keyframe strips AND AI analysis frames.
    Keyframes now use 8-phase layout. AI frames are uniformly sampled for Gemini.

    Third return value: video frame index for each element of ai_frames (same order
    as sent to Gemini) — required when mapping phase picks back to pose timeline.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    rotation = get_video_rotation(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    phases = SWING_PHASES[:num_keyframes]
    kf_indices = {int(p["position"] * (total_frames - 1)) for p in phases}
    ai_indices = set(np.linspace(0, total_frames - 1, num_ai_frames, dtype=int).tolist())
    all_indices = sorted(kf_indices | ai_indices)

    frame_cache: dict[int, np.ndarray] = {}
    for idx in all_indices:
        frame = read_frame_pose_pipeline(cap, int(idx), rotation)
        if frame is not None:
            frame_cache[int(idx)] = frame
    cap.release()

    keyframes = []
    for phase in phases:
        frame_idx = int(phase["position"] * (total_frames - 1))
        frame = frame_cache.get(frame_idx)
        if frame is None:
            continue
        resized = _resize_frame(frame, keyframe_width)
        keyframes.append({
            "phase": phase["name"],
            "label_en": phase["label_en"],
            "label_zh": phase["label_zh"],
            "frame_index": frame_idx,
            "timestamp": round(frame_idx / fps, 3),
            "image_base64": frame_to_base64(resized),
            "width": resized.shape[1],
            "height": resized.shape[0],
        })

    ai_frames: list[str] = []
    ai_frame_video_indices: list[int] = []
    for idx in sorted(ai_indices):
        frame = frame_cache.get(idx)
        if frame is not None:
            ai_frames.append(frame_to_base64(_resize_frame(frame, ai_frame_width), quality=70))
            ai_frame_video_indices.append(int(idx))

    logger.info(f"Extracted {len(keyframes)} keyframes + {len(ai_frames)} AI frames in single pass")
    return keyframes, ai_frames, ai_frame_video_indices


def extract_keyframes(video_path: str, num_frames: int = 8) -> list[dict]:
    kf, _, _ = extract_keyframes_and_ai_frames(video_path, num_keyframes=num_frames, num_ai_frames=0)
    return kf


def extract_all_frames_base64(
    video_path: str, max_frames: int = 8, target_width: int = 384
) -> list[str]:
    _, ai, _ = extract_keyframes_and_ai_frames(
        video_path, num_keyframes=0, num_ai_frames=max_frames, ai_frame_width=target_width
    )
    return ai


def extract_all_frames_base64_with_indices(
    video_path: str, max_frames: int = 8, target_width: int = 384
) -> tuple[list[str], list[int]]:
    """Same as extract_all_frames_base64 plus per-image source video frame indices (Gemini order)."""
    _, ai, ix = extract_keyframes_and_ai_frames(
        video_path, num_keyframes=0, num_ai_frames=max_frames, ai_frame_width=target_width
    )
    return ai, ix


def get_uniform_ai_frame_index_list(total_frames: int, num_ai_frames: int) -> list[int]:
    """Theoretical uniform indices — prefer extract_all_frames_base64_with_indices for exact sync."""
    if total_frames <= 0 or num_ai_frames <= 0:
        return []
    raw = np.linspace(0, max(0, total_frames - 1), num_ai_frames, dtype=int)
    return sorted({int(x) for x in raw.tolist()})


def enforce_phase_windows(
    phase_keyframes: dict[str, int],
    phase_windows: dict[str, list[int]] | None,
) -> dict[str, int]:
    """Clamp phase keyframes to segmentation windows and preserve strict ordering."""
    out = dict(phase_keyframes or {})
    order = ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]
    prev = -1
    for pid in order:
        idx = int(out.get(pid, max(prev + 1, 0)))
        w = (phase_windows or {}).get(pid) or [idx, idx]
        lo = max(prev + 1, int(w[0]))
        hi = max(lo, int(w[1]))
        if idx < lo:
            idx = lo
        elif idx > hi:
            idx = hi
        out[pid] = idx
        prev = idx
    return out
