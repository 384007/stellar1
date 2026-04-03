"""Fuse kinematic per-frame phases with MMAction2 temporal signals (clip + optional windows)."""

from __future__ import annotations

from collections import Counter
from typing import Any

PHASE_ORDER_8 = (
    "address",
    "takeaway",
    "backswing",
    "top",
    "downswing",
    "impact",
    "follow_through",
    "finish",
)
_PHASE_TO_I = {p: i for i, p in enumerate(PHASE_ORDER_8)}
_PHASE_SET = frozenset(PHASE_ORDER_8)


def _resolve_action_label_to_phase(label: str, label_map: dict[str, str]) -> str | None:
    """Map a model class string to a golf phase id, ``unknown``, or None (no mapping)."""
    lab = (label or "").strip().lower()
    if not lab:
        return None
    for key, val in label_map.items():
        k = str(key).strip().lower()
        v = str(val).strip().lower()
        if not k:
            continue
        if lab == k:
            return None if v == "unknown" else (v if v in _PHASE_SET else None)
    for key, val in sorted(label_map.items(), key=lambda x: -len(str(x[0]))):
        k = str(key).strip().lower()
        v = str(val).strip().lower()
        if k and k in lab:
            return None if v == "unknown" else (v if v in _PHASE_SET else None)
    return None


def fuse_kinematic_phases_with_action_priors(
    kin_phases: list[str],
    *,
    window_predictions: list[dict[str, Any]] | None,
    action_label: str = "",
    action_confidence: float = 0.0,
    label_map: dict[str, str] | None = None,
    min_window_confidence: float = 0.12,
    min_clip_confidence: float = 0.42,
    clip_span_blend: float = 0.35,
) -> list[str]:
    """Blend kinematic per-frame phases with MMAction2 clip + sliding-window outputs.

    Kinetics (and other) checkpoints rarely use golf phase ids; callers pass a ``label_map``
    (e.g. from ``STELLAR_MMACTION2_PHASE_LABEL_MAP``) to connect class names to phases.
    When nothing maps, kinematic phases are returned unchanged (clip signal still feeds logits).
    """
    if not kin_phases:
        return []
    out = [str(p) if str(p) in _PHASE_SET else "address" for p in kin_phases]
    n = len(out)
    lm = {str(k): str(v) for k, v in (label_map or {}).items()}
    touched = False

    wins = sorted(window_predictions or [], key=lambda x: int(x.get("pose_start_idx") or 0))
    for w in wins:
        ph = _resolve_action_label_to_phase(str(w.get("label") or ""), lm)
        if ph is None:
            continue
        conf = float(w.get("confidence") or 0.0)
        if conf < min_window_confidence:
            continue
        s = int(w.get("pose_start_idx") or 0)
        e = int(w.get("pose_end_idx") or s)
        s = max(0, min(s, n - 1))
        e = max(s, min(e, n - 1))
        for i in range(s, e + 1):
            out[i] = ph
        touched = True

    if not touched and lm:
        ph_clip = _resolve_action_label_to_phase(str(action_label or ""), lm)
        if ph_clip is not None and float(action_confidence) >= min_clip_confidence:
            half = max(1, int(n * clip_span_blend * 0.5))
            c = n // 2
            lo = max(0, c - half)
            hi = min(n - 1, c + half)
            for i in range(lo, hi + 1):
                out[i] = ph_clip

    return out


def median_smooth_phase_labels(phases: list[str], k: int = 3) -> list[str]:
    if not phases or k < 2:
        return list(phases)
    half = k // 2
    n = len(phases)
    out: list[str] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        c = Counter(str(x) for x in phases[lo:hi] if str(x) in _PHASE_SET)
        if not c:
            out.append(str(phases[i]))
        else:
            out.append(c.most_common(1)[0][0])
    return out


def temporal_prior_strength(
    action_confidence: float,
    window_predictions: list[dict[str, Any]] | None,
) -> float:
    """Scalar in ~[0,1] for downstream gating (segment map competition)."""
    w = window_predictions or []
    if w:
        confs = [float(x.get("confidence") or 0.0) for x in w]
        m = sum(confs) / max(len(confs), 1)
        return min(1.0, 0.45 * m + 0.25 * min(1.0, len(w) / 6.0))
    return min(1.0, max(0.0, float(action_confidence)) * 0.85)


def build_per_frame_phase_logits(
    per_frame_phase: list[str],
    *,
    action_clip_confidence: float = 0.0,
    window_peak_confidence: float = 0.0,
) -> list[list[float]]:
    """Eight-way soft scores per pose step (product-safe: no model vendor names)."""
    n = len(per_frame_phase)
    clip_b = min(0.22, max(0.0, float(action_clip_confidence)) * 0.2)
    win_b = min(0.18, max(0.0, float(window_peak_confidence)) * 0.15)
    boost = clip_b + win_b
    rows: list[list[float]] = []
    for i in range(n):
        row = [0.04] * 8
        pid = str(per_frame_phase[i])
        if pid in _PHASE_TO_I:
            j = _PHASE_TO_I[pid]
            row[j] = 0.68 + boost
        rows.append(row)
    return rows


def build_boundary_confidence(
    phase_boundaries: list[dict[str, Any]],
    base: float = 0.74,
    temporal_boost: float = 0.0,
) -> list[dict[str, Any]]:
    b = min(0.2, max(0.0, temporal_boost))
    out: list[dict[str, Any]] = []
    for i, bd in enumerate(phase_boundaries):
        c = min(0.98, base + b * (1.0 - i / max(len(phase_boundaries), 1)))
        out.append(
            {
                "boundary_index": i,
                "phase_id": bd.get("phase_id"),
                "start_idx": bd.get("start_idx"),
                "end_idx": bd.get("end_idx"),
                "confidence": round(c, 4),
            },
        )
    return out


def phase_confidence_summary(
    *,
    action_clip_confidence: float,
    window_predictions: list[dict[str, Any]] | None,
    boundary_confidences: list[dict[str, Any]],
) -> dict[str, Any]:
    w = window_predictions or []
    w_mean = sum(float(x.get("confidence") or 0) for x in w) / max(len(w), 1) if w else 0.0
    b_vals = [float(x.get("confidence") or 0) for x in boundary_confidences]
    b_mean = sum(b_vals) / max(len(b_vals), 1) if b_vals else 0.0
    return {
        "clip_action_confidence": round(float(action_clip_confidence), 4),
        "window_count": len(w),
        "window_confidence_mean": round(w_mean, 4),
        "boundary_confidence_mean": round(b_mean, 4),
        "global_segmentation_confidence": round(min(0.99, 0.5 * b_mean + 0.35 * w_mean + 0.15 * float(action_clip_confidence)), 4),
    }
