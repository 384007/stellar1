from __future__ import annotations

import copy
import logging
from typing import Any

import cv2
import numpy as np

from services.keyframe_service import (
    _pose_angle_distance,
    _pose_snapshot_for_keyframe,
    _read_frame_pose_matched,
    _read_frame_with_decode_fallback,
    _resize_frame,
    _visual_similarity,
    frame_to_base64,
    recompute_keyframe_details_from_final_strip,
    validate_final_keyframes_for_ai,
)
from services.swing_flow_utils import (
    _build_view_agnostic_kinematics,
    detect_phase_events_agnostic,
    validate_finish_semantic_at_index,
    validate_follow_through_semantic_at_index,
    validate_impact_semantic_at_index,
)
from services.video_utils import get_video_rotation

logger = logging.getLogger(__name__)

_LATE_STRIP = ("downswing", "impact", "follow_through", "finish")
_LATE_STRIP_SET = frozenset(_LATE_STRIP)


def _phase_row_map(keyframes: list[dict[str, Any]]) -> dict[str, int]:
    return {str(k.get("phase")): i for i, k in enumerate(keyframes) if k.get("phase")}


def _late_strip_lists(keyframes: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    rows = [k for k in keyframes if str(k.get("phase")) in _LATE_STRIP]
    return (
        [int(k.get("source_pose_idx", -1)) for k in rows],
        [int(k.get("source_frame_index", k.get("frame_index", -1))) for k in rows],
    )


def _read_pose_frame(cap: cv2.VideoCapture, rotation: int, poses: list[dict], pose_idx: int):
    fi = int(poses[int(pose_idx)].get("frame_index", pose_idx))
    frame = _read_frame_pose_matched(cap, fi, rotation)
    if frame is None:
        frame = _read_frame_with_decode_fallback(cap, fi, rotation)
    return frame


def _apply_pose_idx(
    cap: cv2.VideoCapture,
    rotation: int,
    fps: float,
    poses: list[dict],
    keyframes: list[dict[str, Any]],
    phase_keyframes: dict[str, int],
    phase_id: str,
    pose_idx: int,
    keyframe_width: int,
) -> bool:
    row_map = _phase_row_map(keyframes)
    row = row_map.get(phase_id)
    if row is None or not (0 <= int(pose_idx) < len(poses)):
        return False
    pose = poses[int(pose_idx)]
    fi = int(pose.get("frame_index", pose_idx))
    ts = float(pose.get("timestamp", round(fi / max(float(fps), 1e-6), 3)))
    frame = _read_pose_frame(cap, rotation, poses, int(pose_idx))
    if frame is None:
        return False
    resized = _resize_frame(frame, keyframe_width)
    keyframes[row]["source_pose_idx"] = int(pose_idx)
    keyframes[row]["source_frame_index"] = fi
    keyframes[row]["frame_index"] = fi
    keyframes[row]["timestamp"] = round(ts, 3)
    keyframes[row]["image_base64"] = frame_to_base64(resized)
    keyframes[row]["width"] = resized.shape[1]
    keyframes[row]["height"] = resized.shape[0]
    keyframes[row]["pose_snapshot"] = _pose_snapshot_for_keyframe(pose)
    keyframes[row]["fallback_used"] = True
    keyframes[row]["selection_reason"] = str(keyframes[row].get("selection_reason") or "") + "_late_strip_cleanup"
    phase_keyframes[phase_id] = int(pose_idx)
    return True


def _min_gap_ds_impact(fps: float, *, wide: bool) -> int:
    if wide:
        return max(2, int(round(fps * 0.06)))
    return max(3, int(round(fps * 0.10)))


def _min_gap_follow(fps: float, *, wide: bool) -> int:
    if wide:
        return max(3, int(round(fps * 0.08)))
    return max(4, int(round(fps * 0.12)))


def _min_gap_finish(fps: float, *, wide: bool) -> int:
    if wide:
        return max(3, int(round(fps * 0.08)))
    return max(4, int(round(fps * 0.12)))


def _candidate_pose_range_for_downswing(
    top_pose_idx: int,
    impact_pose_idx: int,
    fps: float,
    total_poses: int,
    *,
    wide: bool,
) -> tuple[int, int]:
    mg = _min_gap_ds_impact(fps, wide=wide)
    back = 36 if wide else 24
    lo = max(top_pose_idx + 1, impact_pose_idx - back)
    hi = impact_pose_idx - mg
    if lo > hi or hi < 0:
        return 1, 0
    lo = max(0, min(lo, total_poses - 1))
    hi = max(0, min(hi, total_poses - 1))
    return lo, hi


def _candidate_pose_range_for_follow_through(
    impact_pose_idx: int,
    finish_pose_idx: int,
    total_poses: int,
    fps: float,
    *,
    wide: bool,
) -> tuple[int, int]:
    mg = _min_gap_follow(fps, wide=wide)
    lo = impact_pose_idx + mg
    hi = min(total_poses - 2, finish_pose_idx - mg)
    if lo > hi:
        hi = total_poses - 2
    lo = max(0, min(lo, total_poses - 1))
    hi = max(0, min(hi, total_poses - 1))
    if lo > hi:
        return 1, 0
    return lo, hi


def _candidate_pose_range_for_finish(
    follow_pose_idx: int,
    total_poses: int,
    fps: float,
    *,
    wide: bool,
) -> tuple[int, int]:
    mg = _min_gap_finish(fps, wide=wide)
    lo = follow_pose_idx + mg
    hi = total_poses - 1
    lo = max(0, min(lo, total_poses - 1))
    hi = max(0, min(hi, total_poses - 1))
    if lo > hi:
        return 1, 0
    return lo, hi


def _valid_pose_indices(lo: int, hi: int, kin: dict | None, n_poses: int) -> list[int]:
    if lo > hi:
        return []
    if kin is None:
        return list(range(max(0, lo), min(hi, n_poses - 1) + 1))
    valid = kin.get("valid")
    if valid is None:
        return list(range(max(0, lo), min(hi, n_poses - 1) + 1))
    out: list[int] = []
    for i in range(lo, hi + 1):
        if 0 <= i < n_poses and i < len(valid) and bool(valid[i]):
            out.append(i)
    return out


def _score_downswing_candidate(
    i: int,
    top_pi: int,
    impact_pi: int,
    poses: list[dict],
    kin: dict | None,
    min_gap_ds: int,
) -> float:
    if i <= top_pi or i >= impact_pi - min_gap_ds or i < 0 or i >= len(poses):
        return -1e18
    gap = float(impact_pi - i)
    pose_div = float(_pose_angle_distance(poses[i], poses[impact_pi]))
    pre_impact_ok = (i > top_pi + 1) and (i < impact_pi)
    bonus_pre = 4.0 if pre_impact_ok else 0.0
    no_cross = (top_pi < i < impact_pi)
    bonus_cross = 3.0 if no_cross else -5.0
    exc_b = 0.0
    if kin is not None and i < len(kin.get("excursion", [])):
        exc_b = float(kin["excursion"][i]) * 0.15
    sem_bonus = 0.0
    if kin is not None:
        spd = np.asarray(kin.get("speed_s", []), dtype=np.float64)
        if i < len(spd) and top_pi < len(spd) and impact_pi < len(spd):
            if float(spd[i]) >= float(spd[top_pi]) * 0.85 and float(spd[i]) <= float(spd[impact_pi]) * 1.05:
                sem_bonus += 2.5
    return gap * 1.15 + pose_div * 0.55 + bonus_pre + bonus_cross + exc_b + sem_bonus


def _pick_best_downswing_candidate(
    lo: int,
    hi: int,
    top_pi: int,
    impact_pi: int,
    poses: list[dict],
    kin: dict | None,
    fps: float,
    wide: bool,
) -> int | None:
    mg = _min_gap_ds_impact(fps, wide=wide)
    cand = _valid_pose_indices(lo, hi, kin, len(poses))
    if not cand:
        return None
    best_i: int | None = None
    best_sc = -1e18
    for i in cand:
        sc = _score_downswing_candidate(i, top_pi, impact_pi, poses, kin, mg)
        if sc > best_sc:
            best_sc = sc
            best_i = i
    return best_i


def _score_follow_through_candidate(
    i: int,
    impact_pi: int,
    poses: list[dict],
    kin: dict | None,
    impact_frame: np.ndarray | None,
    impact_pose: dict,
    cap: cv2.VideoCapture,
    rotation: int,
) -> tuple[float, bool]:
    if i <= impact_pi or i >= len(poses):
        return -1e18, False
    gap_pi = float(i - impact_pi)
    fi_i = int(poses[i].get("frame_index", i))
    fi_imp = int(poses[impact_pi].get("frame_index", impact_pi))
    gap_fi = float(abs(fi_i - fi_imp))
    pose_div = float(_pose_angle_distance(poses[i], impact_pose))
    vis_diff = 0.0
    if impact_frame is not None:
        cf = _read_pose_frame(cap, rotation, poses, i)
        if cf is not None:
            vis_diff = max(0.0, 1.0 - float(_visual_similarity(cf, impact_frame)))
    sem_ok = False
    if kin is not None:
        sem_ok, _ = validate_follow_through_semantic_at_index(i, impact_pi, kin)
    sem_bonus = 8.0 if sem_ok else 0.0
    score = gap_pi * 1.4 + gap_fi * 0.02 + pose_div * 0.45 + vis_diff * 3.0 + sem_bonus
    return score, sem_ok


def _pick_best_follow_candidate(
    lo: int,
    hi: int,
    impact_pi: int,
    poses: list[dict],
    kin: dict | None,
    cap: cv2.VideoCapture,
    rotation: int,
    impact_frame: np.ndarray | None,
    impact_pose: dict,
) -> int | None:
    cand = _valid_pose_indices(lo, hi, kin, len(poses))
    if not cand:
        return None
    sem_pool = [i for i in cand if validate_follow_through_semantic_at_index(i, impact_pi, kin)[0]]
    pool = sem_pool if sem_pool else cand
    best_i: int | None = None
    best_sc = -1e18
    for i in pool:
        sc, _ = _score_follow_through_candidate(
            i, impact_pi, poses, kin, impact_frame, impact_pose, cap, rotation,
        )
        if sc > best_sc:
            best_sc = sc
            best_i = i
    return best_i


def _score_finish_candidate(
    i: int,
    follow_pi: int,
    impact_pi: int,
    poses: list[dict],
    kin: dict | None,
    follow_frame: np.ndarray | None,
    follow_pose: dict,
    cap: cv2.VideoCapture,
    rotation: int,
) -> tuple[float, bool]:
    if i <= follow_pi or i >= len(poses):
        return -1e18, False
    gap_pi = float(i - follow_pi)
    fi_i = int(poses[i].get("frame_index", i))
    fi_f = int(poses[follow_pi].get("frame_index", follow_pi))
    gap_fi = float(abs(fi_i - fi_f))
    pose_div = float(_pose_angle_distance(poses[i], follow_pose))
    vis_b = 0.0
    if follow_frame is not None:
        cf = _read_pose_frame(cap, rotation, poses, i)
        if cf is not None:
            vis_b = max(0.0, 1.0 - float(_visual_similarity(cf, follow_frame))) * 2.5
    speed_bonus = 0.0
    if kin is not None:
        spd = np.asarray(kin.get("speed_s", []), dtype=np.float64)
        if i < len(spd) and impact_pi < len(spd) and follow_pi < len(spd):
            si, sim, sf = float(spd[i]), float(spd[impact_pi]), float(spd[follow_pi])
            if si < sim * 0.88 and si < sf * 0.92:
                speed_bonus += 4.0
    sem_ok = False
    if kin is not None:
        sem_ok, _ = validate_finish_semantic_at_index(i, follow_pi, impact_pi, kin)
    sem_bonus = 8.0 if sem_ok else 0.0
    score = gap_pi * 1.35 + gap_fi * 0.02 + pose_div * 0.5 + vis_b + speed_bonus + sem_bonus
    return score, sem_ok


def _pick_best_finish_candidate(
    lo: int,
    hi: int,
    follow_pi: int,
    impact_pi: int,
    poses: list[dict],
    kin: dict | None,
    cap: cv2.VideoCapture,
    rotation: int,
    follow_frame: np.ndarray | None,
    follow_pose: dict,
) -> int | None:
    cand = _valid_pose_indices(lo, hi, kin, len(poses))
    if not cand:
        return None
    sem_pool = [
        i
        for i in cand
        if validate_finish_semantic_at_index(i, follow_pi, impact_pi, kin)[0]
    ]
    pool = sem_pool if sem_pool else cand
    best_i: int | None = None
    best_sc = -1e18
    for i in pool:
        sc, _ = _score_finish_candidate(
            i, follow_pi, impact_pi, poses, kin, follow_frame, follow_pose, cap, rotation,
        )
        if sc > best_sc:
            best_sc = sc
            best_i = i
    return best_i


def _metrics_from_state(details: list[dict], gate: dict[str, Any]) -> dict[str, Any]:
    nd = sum(1 for d in details if isinstance(d, dict) and d.get("is_near_duplicate"))
    tc = sum(1 for d in details if isinstance(d, dict) and d.get("time_too_close"))
    rem = [
        str(d.get("phase"))
        for d in details
        if isinstance(d, dict) and (d.get("is_near_duplicate") or d.get("time_too_close"))
    ]
    return {
        "gate_pass": bool(gate.get("pass")),
        "near_duplicates": int(nd),
        "time_too_close_count": int(tc),
        "remaining_near_duplicate_phases": list(rem),
    }


def _is_strict_improved(m_old: dict[str, Any], m_new: dict[str, Any]) -> bool:
    if bool(m_new.get("gate_pass")):
        return True
    if len(m_new.get("remaining_near_duplicate_phases") or []) < len(m_old.get("remaining_near_duplicate_phases") or []):
        return True
    if int(m_new.get("near_duplicates", 0)) < int(m_old.get("near_duplicates", 0)):
        return True
    if int(m_new.get("time_too_close_count", 0)) < int(m_old.get("time_too_close_count", 0)):
        return True
    return False


def _late_strip_metrics_from_details(details: list[dict]) -> dict[str, Any]:
    """Subset metrics for the 4 late-strip phases only (early phases may stay noisy)."""
    nd = 0
    tc = 0
    rem: list[str] = []
    for d in details:
        if not isinstance(d, dict):
            continue
        ph = str(d.get("phase") or "")
        if ph not in _LATE_STRIP_SET:
            continue
        if d.get("is_near_duplicate"):
            nd += 1
        if d.get("time_too_close"):
            tc += 1
        if d.get("is_near_duplicate") or d.get("time_too_close"):
            rem.append(ph)
    return {"near_duplicates": nd, "time_too_close_count": tc, "remaining_near_duplicate_phases": rem}


def _late_strip_better(det_old: list[dict], det_new: list[dict]) -> bool:
    lo = _late_strip_metrics_from_details(det_old)
    ln = _late_strip_metrics_from_details(det_new)
    if int(ln["near_duplicates"]) < int(lo["near_duplicates"]):
        return True
    if int(ln["time_too_close_count"]) < int(lo["time_too_close_count"]):
        return True
    if len(ln["remaining_near_duplicate_phases"]) < len(lo["remaining_near_duplicate_phases"]):
        return True
    return False


def _accept_cleanup_delta(
    m_old: dict[str, Any],
    m_new: dict[str, Any],
    det_old: list[dict],
    det_new: list[dict],
) -> bool:
    """Accept if global strict improvement, or global dup/tc not worse and late-strip subset improved."""
    if _is_strict_improved(m_old, m_new):
        return True
    if int(m_new.get("near_duplicates", 0)) > int(m_old.get("near_duplicates", 0)):
        return False
    if int(m_new.get("time_too_close_count", 0)) > int(m_old.get("time_too_close_count", 0)):
        return False
    return _late_strip_better(det_old, det_new)


def _dup_count(m: dict[str, Any]) -> int:
    return int(m.get("near_duplicates", 0))


def _try_nudge_impact_round2(
    cap: cv2.VideoCapture,
    rotation: int,
    fps: float,
    poses: list[dict],
    keyframes: list[dict[str, Any]],
    phase_keyframes: dict[str, int],
    kin: dict,
    top_pi: int,
    exc_apex: int,
    min_time_gap: float,
    keyframe_width: int,
    m_before: dict[str, Any],
    remaining_phases: list[str],
) -> tuple[bool, list[str]]:
    if "impact" not in set(remaining_phases):
        return False, []
    impact_pi = int(phase_keyframes.get("impact", -1))
    ds_pi = int(phase_keyframes.get("downswing", -1))
    ft_pi = int(phase_keyframes.get("follow_through", -1))
    n = len(poses)
    if impact_pi < 0 or ds_pi < 0 or ft_pi < 0:
        return False, []
    lo = max(top_pi + 2, ds_pi + 1, impact_pi - 4)
    hi = min(n - 2, ft_pi - 1, impact_pi + 4)
    if lo > hi:
        return False, []
    best_m: dict[str, Any] | None = None
    best_kf: list[dict[str, Any]] | None = None
    best_pk: dict[str, int] | None = None
    best_cand: int | None = None
    for cand in range(lo, hi + 1):
        if cand <= ds_pi or cand >= ft_pi:
            continue
        if cand >= len(kin["valid"]) or not bool(kin["valid"][cand]):
            continue
        ok_imp, _ = validate_impact_semantic_at_index(cand, top_pi, exc_apex, kin)
        if not ok_imp:
            continue
        kf_t = copy.deepcopy(keyframes)
        pk_t = dict(phase_keyframes)
        if not _apply_pose_idx(cap, rotation, fps, poses, kf_t, pk_t, "impact", cand, keyframe_width):
            continue
        det = recompute_keyframe_details_from_final_strip(
            cap, rotation, fps, poses, kf_t, min_time_gap,
        )
        gate = validate_final_keyframes_for_ai(kf_t, pk_t, det, poses=poses, fps=float(fps))
        m_try = _metrics_from_state(det, gate)
        if _dup_count(m_try) < _dup_count(m_before) or len(m_try["remaining_near_duplicate_phases"]) < len(
            m_before.get("remaining_near_duplicate_phases") or [],
        ):
            if best_m is None or _dup_count(m_try) < _dup_count(best_m):
                best_m = m_try
                best_kf = kf_t
                best_pk = pk_t
                best_cand = cand
    if best_kf is None or best_pk is None or best_cand is None:
        return False, []
    keyframes.clear()
    keyframes.extend(best_kf)
    phase_keyframes.clear()
    phase_keyframes.update(best_pk)
    return True, ["impact"]


def _execute_round(
    cap: cv2.VideoCapture,
    rotation: int,
    fps: float,
    poses: list[dict],
    keyframes: list[dict[str, Any]],
    phase_keyframes: dict[str, int],
    kin: dict | None,
    top_pi: int,
    min_time_gap: float,
    keyframe_width: int,
    *,
    wide: bool,
    allow_impact_nudge: bool,
) -> tuple[list[str], dict[str, Any], list[dict]]:
    """Mutates keyframes / phase_keyframes. Returns (changed_phase_ids, metrics_after, details_after)."""
    skf = copy.deepcopy(keyframes)
    spk = dict(phase_keyframes)
    det_e = recompute_keyframe_details_from_final_strip(
        cap, rotation, fps, poses, keyframes, min_time_gap,
    )
    gate_e = validate_final_keyframes_for_ai(keyframes, phase_keyframes, det_e, poses=poses, fps=float(fps))
    m_e = _metrics_from_state(det_e, gate_e)

    changed: list[str] = []
    n = len(poses)
    imp_cur = int(phase_keyframes.get("impact", 0))
    impact_pose = poses[imp_cur]
    impact_frame = _read_pose_frame(cap, rotation, poses, imp_cur)

    ft_pi = int(phase_keyframes["follow_through"])
    fn_pi = int(phase_keyframes["finish"])
    ds_pi = int(phase_keyframes["downswing"])
    imp_pi = int(phase_keyframes["impact"])

    lo_f, hi_f = _candidate_pose_range_for_follow_through(imp_pi, fn_pi, n, fps, wide=wide)
    best_ft = _pick_best_follow_candidate(
        lo_f, hi_f, imp_pi, poses, kin, cap, rotation, impact_frame, impact_pose,
    )
    if best_ft is not None and best_ft != ft_pi:
        if _apply_pose_idx(cap, rotation, fps, poses, keyframes, phase_keyframes, "follow_through", best_ft, keyframe_width):
            changed.append("follow_through")
            ft_pi = int(phase_keyframes["follow_through"])

    follow_pose = poses[ft_pi]
    follow_frame = _read_pose_frame(cap, rotation, poses, ft_pi)
    lo_fn, hi_fn = _candidate_pose_range_for_finish(ft_pi, n, fps, wide=wide)
    best_fn = _pick_best_finish_candidate(
        lo_fn, hi_fn, ft_pi, imp_pi, poses, kin, cap, rotation, follow_frame, follow_pose,
    )
    if best_fn is not None and best_fn != fn_pi:
        if _apply_pose_idx(cap, rotation, fps, poses, keyframes, phase_keyframes, "finish", best_fn, keyframe_width):
            changed.append("finish")
            fn_pi = int(phase_keyframes["finish"])

    lo_d, hi_d = _candidate_pose_range_for_downswing(top_pi, imp_pi, fps, n, wide=wide)
    best_ds = _pick_best_downswing_candidate(lo_d, hi_d, top_pi, imp_pi, poses, kin, fps, wide)
    if best_ds is not None and best_ds != ds_pi:
        if _apply_pose_idx(cap, rotation, fps, poses, keyframes, phase_keyframes, "downswing", best_ds, keyframe_width):
            changed.append("downswing")

    det = recompute_keyframe_details_from_final_strip(cap, rotation, fps, poses, keyframes, min_time_gap)
    gate = validate_final_keyframes_for_ai(keyframes, phase_keyframes, det, poses=poses, fps=float(fps))
    m_after = _metrics_from_state(det, gate)

    if allow_impact_nudge and kin is not None and (
        not m_after["gate_pass"] or m_after["remaining_near_duplicate_phases"]
    ):
        ev = detect_phase_events_agnostic(poses)
        exc_apex = int(ev.get("excursion_apex_idx", top_pi))
        rem = list(m_after.get("remaining_near_duplicate_phases") or [])
        nudged, nudge_changed = _try_nudge_impact_round2(
            cap,
            rotation,
            fps,
            poses,
            keyframes,
            phase_keyframes,
            kin,
            top_pi,
            exc_apex,
            min_time_gap,
            keyframe_width,
            m_after,
            rem,
        )
        if nudged:
            changed.extend([c for c in nudge_changed if c not in changed])
            det = recompute_keyframe_details_from_final_strip(cap, rotation, fps, poses, keyframes, min_time_gap)
            gate = validate_final_keyframes_for_ai(keyframes, phase_keyframes, det, poses=poses, fps=float(fps))
            m_after = _metrics_from_state(det, gate)

    det_a = det
    m_a = m_after
    round_kept = _accept_cleanup_delta(m_e, m_a, det_e, det_a)
    if not round_kept:
        keyframes.clear()
        keyframes.extend(skf)
        phase_keyframes.clear()
        phase_keyframes.update(spk)
        det_a = recompute_keyframe_details_from_final_strip(
            cap, rotation, fps, poses, keyframes, min_time_gap,
        )
        gate_a = validate_final_keyframes_for_ai(keyframes, phase_keyframes, det_a, poses=poses, fps=float(fps))
        m_a = _metrics_from_state(det_a, gate_a)
        changed = []

    late_e = _late_strip_metrics_from_details(det_e)
    late_a = _late_strip_metrics_from_details(det_a)
    logger.info(
        "[keyframe] late_strip_round_local late_nd %s→%s late_tc %s→%s round_kept=%s",
        late_e["near_duplicates"],
        late_a["near_duplicates"],
        late_e["time_too_close_count"],
        late_a["time_too_close_count"],
        round_kept,
    )

    return changed, m_a, det_a


def cleanup_pro_late_strip_duplicates(
    video_path: str,
    poses: list[dict],
    keyframes: list[dict[str, Any]],
    phase_keyframes: dict[str, int],
    kf_validation: dict[str, Any],
    *,
    keyframe_width: int = 320,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    if len(keyframes) != 8 or not poses:
        return keyframes, kf_validation, phase_keyframes

    original_keyframes = copy.deepcopy(keyframes)
    original_phase = dict(phase_keyframes)
    original_validation = copy.deepcopy(kf_validation)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return keyframes, kf_validation, phase_keyframes
    try:
        rotation = get_video_rotation(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_duration = max(
            float(poses[-1].get("timestamp", 0.0)) - float(poses[0].get("timestamp", 0.0)),
            0.1,
        )
        min_time_gap = max(total_duration * (1.0 / 24.0), 0.05)

        kin = _build_view_agnostic_kinematics(poses)
        ev = detect_phase_events_agnostic(poses)
        top_pi = int(phase_keyframes.get("top", ev.get("top_pose_idx", 0)))

        det0 = recompute_keyframe_details_from_final_strip(
            cap, rotation, fps, poses, keyframes, min_time_gap,
        )
        gate0 = validate_final_keyframes_for_ai(keyframes, phase_keyframes, det0, poses=poses, fps=float(fps))
        m0 = _metrics_from_state(det0, gate0)

        old_spi, old_sfi = _late_strip_lists(keyframes)
        impact_preserved = True
        impact_shift_frames = 0
        follow_shift_frames = 0
        finish_shift_frames = 0
        cleanup_rounds = 0
        r1_reverted = False
        r2_reverted = False
        all_changed: list[str] = []
        reason_parts: list[str] = []

        work_kf = copy.deepcopy(keyframes)
        work_pk = dict(phase_keyframes)

        logger.info(
            "[keyframe] late_strip_precleanup "
            "late_strip_old_source_pose_idx=%s late_strip_old_source_frame_idx=%s "
            "late_strip_old_remaining_near_duplicate_phases=%s late_strip_old_near_duplicates=%s "
            "late_strip_old_time_too_close_count=%s gate_pass=%s",
            old_spi,
            old_sfi,
            m0["remaining_near_duplicate_phases"],
            m0["near_duplicates"],
            m0["time_too_close_count"],
            m0["gate_pass"],
        )

        # --- Round 1 ---
        snap_pre_r1 = (copy.deepcopy(work_kf), dict(work_pk))
        ch1, m_r1, det_r1 = _execute_round(
            cap,
            rotation,
            fps,
            poses,
            work_kf,
            work_pk,
            kin,
            top_pi,
            min_time_gap,
            keyframe_width,
            wide=False,
            allow_impact_nudge=False,
        )
        cleanup_rounds = 1
        logger.info(
            "[keyframe] late_strip_round=1 changed=%s metrics=%s",
            ch1,
            m_r1,
        )
        if not _accept_cleanup_delta(m0, m_r1, det0, det_r1):
            work_kf, work_pk = snap_pre_r1
            r1_reverted = True
            reason_parts.append("r1_reverted_no_gain_vs_baseline")
            m_after_r1 = dict(m0)
            det_r1 = det0
        else:
            all_changed.extend(ch1)
            reason_parts.append("r1_accepted")
            m_after_r1 = dict(m_r1)
            impact_shift_frames = int(work_pk.get("impact", 0)) - int(original_phase.get("impact", 0))
            follow_shift_frames = int(work_pk.get("follow_through", 0)) - int(original_phase.get("follow_through", 0))
            finish_shift_frames = int(work_pk.get("finish", 0)) - int(original_phase.get("finish", 0))
            impact_preserved = impact_shift_frames == 0

        # --- Round 2 ---
        need_r2 = (not m_after_r1["gate_pass"]) or bool(m_after_r1.get("remaining_near_duplicate_phases"))
        snap_pre_r2 = (copy.deepcopy(work_kf), dict(work_pk))
        det_start_r2 = recompute_keyframe_details_from_final_strip(
            cap, rotation, fps, poses, work_kf, min_time_gap,
        )
        gate_start_r2 = validate_final_keyframes_for_ai(
            work_kf, work_pk, det_start_r2, poses=poses, fps=float(fps),
        )
        m_start_r2 = _metrics_from_state(det_start_r2, gate_start_r2)
        m_r2 = m_after_r1
        ch2: list[str] = []
        det_r2 = det_r1
        if need_r2:
            cleanup_rounds = 2
            ch2, m_r2, det_r2 = _execute_round(
                cap,
                rotation,
                fps,
                poses,
                work_kf,
                work_pk,
                kin,
                top_pi,
                min_time_gap,
                keyframe_width,
                wide=True,
                allow_impact_nudge=True,
            )
            logger.info(
                "[keyframe] late_strip_round=2 changed=%s metrics=%s",
                ch2,
                m_r2,
            )
            if not _accept_cleanup_delta(m_start_r2, m_r2, det_start_r2, det_r2):
                work_kf, work_pk = snap_pre_r2
                r2_reverted = True
                reason_parts.append("r2_reverted_no_delta_vs_r2_start")
                det_r2 = det_start_r2
                m_r2 = dict(m_start_r2)
            else:
                for c in ch2:
                    if c not in all_changed:
                        all_changed.append(c)
                reason_parts.append("r2_accepted")
                impact_shift_frames = int(work_pk.get("impact", 0)) - int(original_phase.get("impact", 0))
                follow_shift_frames = int(work_pk.get("follow_through", 0)) - int(original_phase.get("follow_through", 0))
                finish_shift_frames = int(work_pk.get("finish", 0)) - int(original_phase.get("finish", 0))
                impact_preserved = impact_shift_frames == 0

        det_f = recompute_keyframe_details_from_final_strip(
            cap, rotation, fps, poses, work_kf, min_time_gap,
        )
        gate_f = validate_final_keyframes_for_ai(work_kf, work_pk, det_f, poses=poses, fps=float(fps))
        m_final = _metrics_from_state(det_f, gate_f)

        if not _accept_cleanup_delta(m0, m_final, det0, det_f):
            logger.info(
                "[keyframe] late_strip_cleanup_reverted full_rollback "
                "late_strip_cleanup_pass=false late_strip_cleanup_rounds=%s "
                "late_strip_cleanup_reason=%s",
                cleanup_rounds,
                ";".join(reason_parts) or "no_gain_vs_baseline",
            )
            return original_keyframes, original_validation, original_phase

        new_spi, new_sfi = _late_strip_lists(work_kf)
        resolved = len(m_final.get("remaining_near_duplicate_phases") or []) == 0
        cleanup_pass = True
        cleanup_reason = ";".join(reason_parts) if reason_parts else "late_strip_candidate_rescore"
        cleanup_improved = _accept_cleanup_delta(m0, m_final, det0, det_f)
        cleanup_reverted_any_round = bool(r1_reverted or r2_reverted)

        keyframes.clear()
        keyframes.extend(work_kf)
        phase_keyframes.clear()
        phase_keyframes.update(work_pk)

        fv_out = {
            **dict(gate_f),
            "late_strip_cleanup_accepted_by_service": True,
            "late_strip_cleanup_applied": True,
            "late_strip_cleanup_resolved": resolved,
            "remaining_near_duplicate_phases": list(m_final.get("remaining_near_duplicate_phases") or []),
            "impact_preserved": bool(impact_preserved),
            "impact_shift_frames": int(impact_shift_frames),
            "follow_shift_frames": int(follow_shift_frames),
            "finish_shift_frames": int(finish_shift_frames),
            "late_strip_cleanup_rounds": int(cleanup_rounds),
            "late_strip_cleanup_changed_phase_ids": list(all_changed),
            "late_strip_cleanup_improved": bool(cleanup_improved),
            "late_strip_cleanup_reverted": cleanup_reverted_any_round,
            "late_strip_cleanup_pass": cleanup_pass,
            "late_strip_cleanup_reason": cleanup_reason,
            "near_duplicates": int(m_final["near_duplicates"]),
            "time_too_close_count": int(m_final["time_too_close_count"]),
        }

        new_validation = {
            **kf_validation,
            "details": det_f,
            "near_duplicates": int(m_final["near_duplicates"]),
            "time_too_close": int(m_final["time_too_close_count"]),
            "all_passed": bool(gate_f.get("pass")),
            "final_phase_keyframes": dict(phase_keyframes),
            "final_keyframe_validation": {
                **dict(kf_validation.get("final_keyframe_validation") or {}),
                **fv_out,
            },
            "final_keyframe_order_ok": bool(gate_f.get("final_keyframe_order_ok")),
            "final_keyframe_time_order_ok": bool(gate_f.get("final_keyframe_time_order_ok")),
            "final_keyframe_gate_pass": bool(gate_f.get("pass")),
            "final_keyframe_source": "late_strip_cleanup_repaired" if bool(gate_f.get("pass")) else str(
                kf_validation.get("final_keyframe_source") or "late_strip_cleanup_attempted",
            ),
            "late_strip_cleanup_pass": cleanup_pass,
            "late_strip_cleanup_reason": cleanup_reason,
            "late_strip_cleanup_changed_phase_ids": list(all_changed),
        }

        logger.info(
            "[keyframe] late_strip_postcleanup "
            "late_strip_new_source_pose_idx=%s late_strip_new_source_frame_idx=%s "
            "late_strip_new_remaining_near_duplicate_phases=%s late_strip_new_near_duplicates=%s "
            "late_strip_new_time_too_close_count=%s impact_preserved=%s impact_shift_frames=%s "
            "follow_shift_frames=%s finish_shift_frames=%s late_strip_cleanup_pass=%s "
            "late_strip_cleanup_rounds=%s late_strip_cleanup_reason=%s "
            "late_strip_cleanup_changed_phase_ids=%s late_strip_cleanup_improved=%s "
            "late_strip_cleanup_reverted=%s gate_pass=%s",
            new_spi,
            new_sfi,
            m_final["remaining_near_duplicate_phases"],
            m_final["near_duplicates"],
            m_final["time_too_close_count"],
            impact_preserved,
            impact_shift_frames,
            follow_shift_frames,
            finish_shift_frames,
            cleanup_pass,
            cleanup_rounds,
            cleanup_reason,
            all_changed,
            cleanup_improved,
            cleanup_reverted_any_round,
            m_final["gate_pass"],
        )

        return keyframes, new_validation, phase_keyframes
    finally:
        cap.release()
