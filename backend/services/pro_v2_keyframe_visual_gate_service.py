"""Pro v2 — adjacent keyframe visual diversity gate (Screen Mode)."""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

from services.pro_v2_keyframe_picker_service import PHASE_ORDER

logger = logging.getLogger(__name__)

# Normalized mean L1 diff between downscaled grayscale frames; below → visually duplicate.
_ADJACENT_MIN_DIFF = 0.022
_GLOBAL_MIN_DIFF = 0.018


def _b64_to_gray_small(b64: str, size: tuple[int, int] = (80, 45)) -> np.ndarray | None:
    s = (b64 or "").strip()
    if len(s) < 64:
        return None
    try:
        raw = base64.b64decode(s, validate=False)
    except Exception:
        return None
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        return None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA)


def _mean_norm_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def run_keyframe_visual_diversity_gate(keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare consecutive phases; fail if too many adjacent pairs look identical."""
    by_phase: dict[str, dict[str, Any]] = {}
    for k in keyframes:
        p = str(k.get("phase") or "").strip()
        if p:
            by_phase[p] = k

    grays: list[tuple[str, np.ndarray]] = []
    missing: list[str] = []
    for ph in PHASE_ORDER:
        row = by_phase.get(ph)
        b64 = str((row or {}).get("image_base64") or "").strip()
        g = _b64_to_gray_small(b64)
        if g is None:
            missing.append(ph)
        else:
            grays.append((ph, g))

    duplicate_pairs: list[list[str]] = []
    adjacent_diffs: list[float] = []

    for i in range(len(grays) - 1):
        pa, ga = grays[i]
        pb, gb = grays[i + 1]
        d = _mean_norm_l1(ga, gb)
        adjacent_diffs.append(d)
        if d < _ADJACENT_MIN_DIFF:
            duplicate_pairs.append([pa, pb])

    # Global: if max spread is tiny, whole strip collapsed
    global_min = 1.0
    if len(grays) >= 2:
        for i in range(len(grays)):
            for j in range(i + 1, len(grays)):
                global_min = min(global_min, _mean_norm_l1(grays[i][1], grays[j][1]))

    # Same frame_index reused across phases
    index_reason = False
    idx_by_phase = [int(by_phase[p].get("frame_index") or -1) for p in PHASE_ORDER if p in by_phase]
    if len(idx_by_phase) >= 2:
        from collections import Counter

        c = Counter(idx_by_phase)
        if any(v > 1 for v in c.values()):
            index_reason = True

    reason_codes: list[str] = []
    if missing:
        reason_codes.append("KEYFRAME_IMAGE_DECODE_PARTIAL")
    if index_reason:
        reason_codes.append("KEYFRAME_INDEX_COLLAPSE")
    if duplicate_pairs:
        reason_codes.append("KEYFRAME_VISUAL_DUPLICATE")
    if len(grays) >= 4 and global_min < _GLOBAL_MIN_DIFF:
        reason_codes.append("LATE_STRIP_COLLAPSED")

    passed = len(reason_codes) == 0
    min_pair_distance = min(adjacent_diffs) if adjacent_diffs else 0.0

    out = {
        "passed": passed,
        "duplicate_pairs": duplicate_pairs,
        "min_pair_distance": round(min_pair_distance, 5),
        "adjacent_diffs": [round(x, 5) for x in adjacent_diffs],
        "global_min_pair_distance": round(global_min, 5),
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "missing_phases_for_gate": missing,
    }
    logger.info(
        "[PRO_V2][KEYFRAME_VISUAL_GATE] passed=%s min_adj_diff=%.5f global_min=%.5f dup_pairs=%s reasons=%s",
        passed,
        min_pair_distance,
        global_min,
        duplicate_pairs[:8],
        out["reason_codes"],
    )
    return out
