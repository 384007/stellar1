"""Joint chain solver for post-impact phases."""

from __future__ import annotations

from typing import Any

from services.provider_registry import role_log

PHASE_IDS = ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]


def _track_frames(tracks: dict[str, Any] | None) -> set[int]:
    rows = []
    if isinstance(tracks, dict):
        rows.extend(tracks.get("person_tracks") or [])
        rows.extend(tracks.get("club_tracks") or [])
        rows.extend(tracks.get("ball_tracks") or [])
    return {int(r.get("frame_index", -1)) for r in rows if int(r.get("frame_index", -1)) >= 0}


def _det_by_frame(detections: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for d in detections or []:
        fi = int(d.get("frame_index", -1))
        if fi < 0:
            continue
        out.setdefault(fi, []).append(d)
    return out


def _pose_quality(p: dict) -> float:
    q = p.get("pose_quality")
    if isinstance(q, (int, float)):
        return float(q)
    joints = p.get("joints") or []
    if not joints:
        return 0.0
    vis = [float(j.get("visibility", 0.0)) for j in joints]
    return sum(vis) / max(len(vis), 1)


def solve_full_phase_chain(
    poses: list[dict],
    coarse_phase_keyframes: dict[str, int],
    phase_windows: dict[str, list[int]] | None = None,
    detections: list[dict] | None = None,
    tracks: dict[str, Any] | None = None,
    motion_3d: list | None = None,
) -> dict[str, Any]:
    from services.swing_flow_utils import _build_view_agnostic_kinematics, compute_chain_kinematic_markers

    n = len(poses)
    if n < 8:
        return {"ok": False, "error": "insufficient_pose_frames", "phase_keyframes": dict(coarse_phase_keyframes or {})}
    det_map = _det_by_frame(detections or [])
    track_frame_set = _track_frames(tracks)
    kin = _build_view_agnostic_kinematics(poses) if n >= 8 else None
    sp = kin["speed_s"] if kin is not None else [0.0] * n
    rev = kin["dot_rev"] if kin is not None else [0.0] * n
    xf_d = kin["xf_d"] if kin is not None else [0.0] * n
    km = compute_chain_kinematic_markers(poses) if n >= 8 else {"ok": False}
    motion_speed: list[float] = [0.0] * n
    motion_settle_idx = n - 1
    if motion_3d:
        try:
            for i in range(1, min(n, len(motion_3d))):
                prev = motion_3d[i - 1] or []
                cur = motion_3d[i] or []
                if prev and cur and len(prev) == len(cur):
                    s = 0.0
                    c = 0
                    for a, b in zip(prev, cur):
                        if len(a) >= 3 and len(b) >= 3:
                            s += ((float(b[0]) - float(a[0])) ** 2 + (float(b[1]) - float(a[1])) ** 2 + (float(b[2]) - float(a[2])) ** 2) ** 0.5
                            c += 1
                    motion_speed[i] = (s / max(c, 1))
            nz = [x for x in motion_speed if x > 0.0]
            ref = (sum(nz) / max(len(nz), 1)) if nz else 0.0
            for i in range(int(n * 0.65), n):
                if motion_speed[i] <= ref * 0.45:
                    motion_settle_idx = i
                    break
        except Exception:
            motion_speed = [0.0] * n
    impact_seed = int((coarse_phase_keyframes or {}).get("impact", min(n - 1, int(n * 0.7))))
    if km.get("ok"):
        impact_seed = int(round(0.6 * impact_seed + 0.4 * int(km.get("impact_seed", impact_seed))))
    if kin is not None:
        lo = max(1, int(n * 0.45))
        hi = min(n - 2, int(n * 0.92))
        impact_seed = max(lo, min(hi, max(range(lo, hi + 1), key=lambda i: float(sp[i]) + max(0.0, -float(xf_d[i])))))
    final_map: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    prev = -1
    reasons: list[str] = []
    for i, pid in enumerate(PHASE_IDS):
        coarse = int((coarse_phase_keyframes or {}).get(pid, max(0, min(n - 1, int(i * (n - 1) / 7)))))
        w = (phase_windows or {}).get(pid) or [max(0, coarse - max(2, n // 12)), min(n - 1, coarse + max(2, n // 12))]
        lo = max(prev + 1, int(w[0]))
        hi = min(n - 1, int(w[1]))
        if lo > hi:
            lo = min(n - 1, max(prev + 1, int(w[0]), int(w[1])))
            hi = lo
        best_idx = lo
        best_score = -1e9
        for idx in range(lo, hi + 1):
            p = poses[idx]
            frame_i = int(p.get("frame_index", idx))
            q = _pose_quality(p)
            track_bonus = 0.25 if frame_i in track_frame_set else 0.0
            cls = {str(d.get("class_name", "")).lower() for d in det_map.get(frame_i, [])}
            det_bonus = 0.0
            if pid in {"address", "takeaway", "backswing", "top", "downswing"} and "person" in cls:
                det_bonus += 0.1
            if pid in {"impact", "follow_through"} and ("club" in cls or "ball" in cls):
                det_bonus += 0.35
            if pid == "finish" and "person" in cls:
                det_bonus += 0.1
            dist_penalty = abs(idx - coarse) / max(n, 1)
            dyn = 0.0
            if pid == "top":
                dyn += 0.35 * float(rev[idx])
                dyn += 0.2 * (1.0 - min(float(sp[idx]) / max(float(max(sp)), 1e-6), 1.0))
                if i > 0 and idx - prev < max(1, n // 18):
                    dyn -= 0.6
            elif pid == "impact":
                dyn += 0.4 * min(float(sp[idx]) / max(float(max(sp)), 1e-6), 1.0)
                dyn += 0.25 * max(0.0, -float(xf_d[idx]))
                dist_penalty += 0.6 * (abs(idx - impact_seed) / max(n, 1))
                if abs(idx - impact_seed) > max(2, n // 12):
                    dyn -= 1.2
            elif pid == "follow_through":
                dyn += 0.25 * min(float(sp[idx]) / max(float(max(sp)), 1e-6), 1.0)
                if "impact" in final_map and idx <= final_map["impact"] + max(1, n // 30):
                    dyn -= 0.5
                if idx > n - 3:
                    dyn -= 0.8
                if motion_3d:
                    dyn += 0.2 * min(float(motion_speed[idx]) / max(float(max(motion_speed) or 1.0), 1e-6), 1.0)
            elif pid == "finish":
                dyn += 0.3 * (1.0 - min(float(sp[idx]) / max(float(max(sp)), 1e-6), 1.0))
                if "follow_through" in final_map and idx - final_map["follow_through"] < max(2, n // 25):
                    dyn -= 0.7
                if motion_3d:
                    dyn += 0.35 * (1.0 - min(abs(idx - motion_settle_idx) / max(n, 1), 1.0))
            score = q + track_bonus + det_bonus + dyn - 0.2 * dist_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        final_map[pid] = int(best_idx)
        evidence[pid] = {
            "selected_pose_idx": int(best_idx),
            "coarse_pose_idx": int(coarse),
            "window": [int(lo), int(hi)],
            "pose_quality": float(_pose_quality(poses[best_idx])),
            "track_supported": bool(int(poses[best_idx].get("frame_index", best_idx)) in track_frame_set),
            "det_classes": sorted({str(d.get("class_name", "")).lower() for d in det_map.get(int(poses[best_idx].get("frame_index", best_idx)), [])}),
            "speed": float(sp[best_idx]) if best_idx < len(sp) else 0.0,
            "direction_reversal": float(rev[best_idx]) if best_idx < len(rev) else 0.0,
            "xf_derivative": float(xf_d[best_idx]) if best_idx < len(xf_d) else 0.0,
            "score": float(best_score),
            "motion3d_speed": float(motion_speed[best_idx]) if best_idx < len(motion_speed) else 0.0,
            "motion3d_settle_idx": int(motion_settle_idx),
        }
        prev = int(best_idx)
        reasons.append(f"{pid}:window={lo}-{hi},coarse={coarse},selected={best_idx},score={best_score:.3f}")
    if final_map["impact"] <= final_map["top"] + 1:
        return {"ok": False, "error": "impact_not_after_top", "phase_keyframes": final_map, "reasons": reasons, "evidence": evidence}
    role_log(f"[ROLE=CHAIN_SOLVER] impact_seed={impact_seed} coarse={coarse_phase_keyframes} selected={final_map}")
    return {"ok": True, "phase_keyframes": final_map, "reasons": reasons, "evidence": evidence, "impact_seed": int(impact_seed)}


def solve_post_impact_phase_chain(
    poses: list[dict],
    phase_keyframes: dict[str, int],
    tracks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from services.swing_flow_utils import propose_post_impact_chain_indices, propose_quality_spacing_post_top_chain

    chain = propose_post_impact_chain_indices(
        poses,
        dict(phase_keyframes),
        tracks=tracks,
        anchor_penalty_scale=0.22,
    )
    out = dict(phase_keyframes)
    material_change = False
    reasons: list[str] = []
    if not chain:
        reasons.append("NO_CHAIN_CANDIDATE")
    for k, v in chain.items():
        if out.get(k) != v:
            material_change = True
        out[k] = int(v)
    if chain and not material_change:
        reasons.append("CHAIN_NO_MATERIAL_CHANGE")
    if not material_change and len(poses) >= 16:
        qs = propose_quality_spacing_post_top_chain(poses, dict(phase_keyframes), fps=30.0)
        if qs:
            for k, v in qs.items():
                if k not in ("downswing", "impact", "follow_through", "finish"):
                    continue
                vi = int(v)
                if out.get(k) != vi:
                    material_change = True
                out[k] = vi
            if material_change:
                chain = {k: out[k] for k in ("downswing", "impact", "follow_through", "finish")}
                reasons.append("CHAIN_FROM_QUALITY_SPACING_FALLBACK")
    return {
        "phase_keyframes": out,
        "chain": chain,
        "chain_rebuilt": bool(chain),
        "material_change": material_change,
        "reasons": reasons,
    }
