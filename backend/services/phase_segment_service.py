"""Temporal phase segmentation facade with stable output schema."""

from __future__ import annotations

import json
import os
from typing import Any

from services.provider_registry import role_log


def _mmaction_phase_label_map() -> dict[str, str]:
    """Optional JSON object: model class name → golf ``phase_id`` (or ``unknown``)."""
    raw = (os.getenv("STELLAR_MMACTION2_PHASE_LABEL_MAP") or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        role_log("[ROLE=MMACTION2] phase_label_map_json_invalid")
        return {}


def _mmaction2_action_enabled() -> bool:
    """Whether to invoke the MMAction2 provider.

    - ``STELLAR_ACTION_BACKEND=disabled`` → off.
    - Any other non-empty value (e.g. ``mmaction2``) → on.
    - If unset, on when both ``STELLAR_MMACTION2_CONFIG`` and ``STELLAR_MMACTION2_CHECKPOINT`` are set
      so deploys can enable by paths alone without a third flag.
    """
    raw = (os.getenv("STELLAR_ACTION_BACKEND") or "").strip().lower()
    if raw == "disabled":
        return False
    if raw not in ("", "disabled"):
        return True
    cfg = (os.getenv("STELLAR_MMACTION2_CONFIG") or "").strip()
    ckpt = (os.getenv("STELLAR_MMACTION2_CHECKPOINT") or "").strip()
    return bool(cfg and ckpt)


def segment_swing_phases(
    poses: list[dict],
    tracks: dict[str, Any] | None = None,
    detections: list[dict] | None = None,
    motion_3d: list | None = None,
    video_path: str | None = None,
    *,
    precomputed_action: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from services.phase_action_fusion import (
        build_boundary_confidence,
        build_per_frame_phase_logits,
        fuse_kinematic_phases_with_action_priors,
        median_smooth_phase_labels,
        phase_confidence_summary,
        temporal_prior_strength,
    )
    from services.swing_flow_utils import compute_chain_kinematic_markers, detect_phase_events_agnostic, detect_swing_phases, get_phase_keyframes, _build_view_agnostic_kinematics

    swing_phases = detect_swing_phases(poses)
    n = len(poses)
    per_frame_phase_kin: list[str] = []
    for i in range(n):
        ph = swing_phases[i].get("phase_id") if i < len(swing_phases) else "address"
        per_frame_phase_kin.append(str(ph))

    action_prior: dict[str, Any] = {}
    action_meta: dict[str, Any] = {"provider_name": "disabled", "status": "disabled", "error_reason": "disabled"}
    if precomputed_action is not None:
        action_prior, action_meta = precomputed_action
        action_prior = dict(action_prior or {})
        action_meta = dict(action_meta or {})
    elif _mmaction2_action_enabled():
        from services.providers.action_mmaction2_provider import run as run_action

        action_meta = run_action(video_path or "", n)
        if action_meta.get("status") == "ok":
            action_prior = dict(action_meta.get("payload") or {})
    else:
        role_log(f"[ROLE=MMACTION2] status=disabled clip_len={n}")

    win_list = list(action_prior.get("window_predictions") or [])
    label_map = _mmaction_phase_label_map()
    if action_meta.get("status") == "ok":
        per_frame_phase_raw = fuse_kinematic_phases_with_action_priors(
            per_frame_phase_kin,
            window_predictions=win_list,
            action_label=str(action_prior.get("action_label") or ""),
            action_confidence=float(action_prior.get("action_confidence") or 0.0),
            label_map=label_map if label_map else None,
        )
    else:
        per_frame_phase_raw = list(per_frame_phase_kin)

    per_frame_phase = median_smooth_phase_labels(per_frame_phase_raw, k=3)
    clip_conf = float(action_prior.get("action_confidence") or 0.0) if action_meta.get("status") == "ok" else 0.0
    win_peak = max((float(w.get("confidence") or 0.0) for w in win_list), default=0.0)
    tps = temporal_prior_strength(clip_conf, win_list)
    per_frame_phase_logits = build_per_frame_phase_logits(
        per_frame_phase,
        action_clip_confidence=clip_conf,
        window_peak_confidence=win_peak,
    )
    phase_boundaries: list[dict[str, Any]] = []
    bi = 0
    while bi < len(per_frame_phase):
        pid0 = per_frame_phase[bi]
        bj = bi
        while bj < len(per_frame_phase) and per_frame_phase[bj] == pid0:
            bj += 1
        phase_boundaries.append({"phase_id": pid0, "start_idx": bi, "end_idx": bj - 1})
        bi = bj
    boundary_confidence = build_boundary_confidence(phase_boundaries, temporal_boost=tps)
    phase_confidence = phase_confidence_summary(
        action_clip_confidence=clip_conf,
        window_predictions=win_list,
        boundary_confidences=boundary_confidence,
    )
    segment_bundle: dict[str, Any] = {
        "per_frame_phase": per_frame_phase,
        "phase_boundaries": phase_boundaries,
        "temporal_prior_strength": round(tps, 4),
        "per_frame_phase_logits": per_frame_phase_logits,
        "boundary_confidence": boundary_confidence,
        "phase_confidence": phase_confidence,
    }
    phase_keyframes = get_phase_keyframes(swing_phases, poses, segment_bundle=segment_bundle)
    kin = _build_view_agnostic_kinematics(poses) if n >= 8 else None
    km = compute_chain_kinematic_markers(poses) if n >= 8 else {"ok": False}
    ev = detect_phase_events_agnostic(poses) if n >= 8 else {}
    top_ev = int(ev.get("top_pose_idx", phase_keyframes.get("top", max(0, n // 2))))
    impact_ev = int(ev.get("impact_pose_idx", phase_keyframes.get("impact", max(0, int(n * 0.7)))))
    track_frames = {int(r.get("frame_index", -1)) for k in ("person_tracks", "club_tracks", "ball_tracks") for r in (tracks or {}).get(k, []) if int(r.get("frame_index", -1)) >= 0}
    det_impact = sorted({
        int(d.get("frame_index", -1))
        for d in (detections or [])
        if str(d.get("class_name", "")).lower() in {"club", "ball"} and int(d.get("frame_index", -1)) >= 0
    })
    if det_impact:
        impact_ev = min(det_impact, key=lambda x: abs(x - impact_ev))
    top_center = int(round(0.65 * top_ev + 0.35 * int(phase_keyframes.get("top", top_ev))))
    impact_center = int(round(0.75 * impact_ev + 0.10 * int(phase_keyframes.get("impact", impact_ev)) + 0.15 * int(km.get("impact_seed", impact_ev))))
    ds_span = max(impact_center - top_center, 2)
    downswing_event_center = min(
        max(top_center + max(2, ds_span // 3), top_center + 1),
        max(top_center + 1, impact_center - max(2, n // 28)),
    )
    ft_center = min(n - 3, max(impact_center + 1, impact_center + max(2, n // 14)))
    finish_center = min(n - 1, max(ft_center + 2, int(n * 0.9)))
    if kin is not None:
        speed = kin["speed_s"]
        valid = kin["valid"]
        post = [i for i in range(min(len(speed) - 1, impact_center + 1), len(speed)) if valid[i]]
        if post:
            ft_center = min((i for i in post if i <= n - 3), default=ft_center, key=lambda i: abs(speed[i] - float(speed[impact_center]) * 0.7))
            finish_center = min((i for i in post if i >= ft_center + 2), default=finish_center, key=lambda i: abs(speed[i] - float(speed[impact_center]) * 0.35))
    if motion_3d:
        try:
            speed3 = [0.0] * n
            for i in range(1, min(n, len(motion_3d))):
                a = motion_3d[i - 1] or []
                b = motion_3d[i] or []
                if a and b and len(a) == len(b):
                    s = 0.0
                    c = 0
                    for p0, p1 in zip(a, b):
                        if len(p0) >= 3 and len(p1) >= 3:
                            s += ((float(p1[0]) - float(p0[0])) ** 2 + (float(p1[1]) - float(p0[1])) ** 2 + (float(p1[2]) - float(p0[2])) ** 2) ** 0.5
                            c += 1
                    speed3[i] = s / max(c, 1)
            nz = [v for v in speed3 if v > 0.0]
            if nz:
                ref = sum(nz) / len(nz)
                settle = next((i for i in range(max(ft_center + 2, int(n * 0.65)), n) if speed3[i] <= ref * 0.45), finish_center)
                finish_center = int(max(ft_center + 2, min(n - 1, settle)))
        except Exception:
            pass
    def _window(center: int, radius: int, frame_set: set[int] | None = None) -> list[int]:
        lo = max(0, center - radius)
        hi = min(max(n - 1, 0), center + radius)
        if frame_set:
            inside = [f for f in frame_set if lo <= f <= hi]
            if inside:
                lo = max(lo, min(inside))
                hi = min(hi, max(inside))
        return [int(lo), int(max(lo, hi))]
    windows = {
        "address": _window(int(phase_keyframes.get("address", 0)), max(2, n // 18), track_frames),
        "takeaway": _window(int(phase_keyframes.get("takeaway", max(1, n // 8))), max(2, n // 16), track_frames),
        "backswing": _window(int(phase_keyframes.get("backswing", max(2, n // 3))), max(2, n // 14), track_frames),
        "top": _window(top_center, max(2, n // 16), track_frames),
        "downswing": _window(int(downswing_event_center), max(2, n // 14), track_frames),
        "impact": _window(impact_center, max(2, n // 20), set(det_impact) if det_impact else track_frames),
        "follow_through": _window(ft_center, max(2, n // 16), track_frames),
        "finish": _window(finish_center, max(2, n // 14), track_frames),
    }
    ordered = ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]
    prev_hi = -1
    for pid in ordered:
        lo, hi = windows[pid]
        lo = max(lo, prev_hi + 1)
        hi = max(hi, lo)
        windows[pid] = [lo, hi]
        prev_hi = hi
    window_evidence = {
        "top_center": top_center,
        "impact_center": impact_center,
        "follow_center": ft_center,
        "finish_center": finish_center,
        "top_event_idx": top_ev,
        "impact_event_idx": impact_ev,
        "impact_seed": int(km.get("impact_seed", impact_ev)),
        "club_ball_frames": det_impact,
        "track_frame_count": len(track_frames),
        "downswing_event_center": int(downswing_event_center),
    }
    phase_logits: list[dict[str, Any]] = [
        {
            "boundary_index": int(b.get("boundary_index", ix)),
            "phase_id": b.get("phase_id"),
            "start_idx": b.get("start_idx"),
            "end_idx": b.get("end_idx"),
            "confidence": float(b.get("confidence") or 0.74),
            "kind": "kinematic_segment",
        }
        for ix, b in enumerate(boundary_confidence)
    ]
    if action_meta.get("status") == "ok" and action_prior:
        phase_logits.append(
            {
                "kind": "temporal_action_prior",
                "confidence": float(action_prior.get("action_confidence") or 0.0),
                "window_count": len(win_list),
                "temporal_prior_strength": round(tps, 4),
            },
        )
    role_log(f"[ROLE=PHASE_WINDOW] windows={windows} evidence={window_evidence}")
    return {
        "swing_phases": swing_phases,
        "phase_keyframes": phase_keyframes,
        "phase_windows": windows,
        "phase_window_evidence": window_evidence,
        "phase_logits": phase_logits,
        "phase_boundaries": phase_boundaries,
        "per_frame_phase": per_frame_phase,
        "per_frame_phase_logits": per_frame_phase_logits,
        "boundary_confidence": boundary_confidence,
        "phase_confidence": phase_confidence,
        "temporal_prior_strength": round(tps, 4),
        "action_prior": action_prior,
        "action_provider_meta": action_meta,
    }


def compact_phase_c_context_for_plus_prompt(seg: dict[str, Any]) -> dict[str, Any]:
    """Phase C: compact temporal-segmentation summary for Plus Gemini (no per-frame arrays).

    Feeds fused kinematic strip + optional action-prior confidence into the vision model so
    narrative certainty tracks backend segmentation quality.
    """
    pc = dict(seg.get("phase_confidence") or {}) if isinstance(seg.get("phase_confidence"), dict) else {}
    am = dict(seg.get("action_provider_meta") or {}) if isinstance(seg.get("action_provider_meta"), dict) else {}
    keys = (
        "clip_action_confidence",
        "window_count",
        "window_confidence_mean",
        "boundary_confidence_mean",
        "global_segmentation_confidence",
    )
    slim_pc = {k: pc[k] for k in keys if k in pc}
    return {
        "phase_c_version": "1",
        "temporal_prior_strength": round(float(seg.get("temporal_prior_strength") or 0.0), 4),
        "phase_confidence": slim_pc,
        "phase_boundary_segment_count": len(list(seg.get("phase_boundaries") or [])),
        "action_backend": {
            "status": str(am.get("status") or "unknown"),
            "name": str(am.get("provider_name") or "none"),
        },
    }
