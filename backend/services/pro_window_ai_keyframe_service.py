"""Pro Stage 4: AI picks one candidate index per phase inside pre-cut motion windows only."""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from services.gemini_service import extract_json_from_response
from services.keyframe_service import PHASE_ORDER
from services.video_utils import get_video_rotation

logger = logging.getLogger(__name__)

_PRO_WINDOW_AI_TIMEOUT_S = float(os.getenv("STELLAR_PRO_WINDOW_AI_TIMEOUT_S", "75"))


def _linspace_indices(lo: int, hi: int, k: int) -> list[int]:
    if hi < lo:
        lo, hi = hi, lo
    n = hi - lo + 1
    if n <= k:
        return list(range(lo, hi + 1))
    xs = np.linspace(lo, hi, k)
    out = sorted({int(round(float(x))) for x in xs})
    return out


def _encode_jpeg_b64(frame: np.ndarray, w: int = 320) -> str:
    import base64

    h0, w0 = frame.shape[:2]
    if w0 > w:
        sc = w / w0
        frame = cv2.resize(frame, (w, int(h0 * sc)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_candidates_for_window(
    analysis_video_path: str,
    poses: list[dict],
    lo: int,
    hi: int,
    max_candidates: int,
) -> tuple[list[str], list[int]]:
    cap = cv2.VideoCapture(analysis_video_path)
    if not cap.isOpened():
        return [], []
    rotation = get_video_rotation(analysis_video_path)
    idxs = _linspace_indices(lo, hi, max_candidates)
    images: list[str] = []
    pose_idx_out: list[int] = []
    from services.video_utils import read_frame_pose_pipeline

    for pi in idxs:
        if not (0 <= pi < len(poses)):
            continue
        fi = int(poses[pi].get("frame_index", pi))
        frame = read_frame_pose_pipeline(cap, fi, rotation)
        if frame is None:
            continue
        b64 = _encode_jpeg_b64(frame)
        if b64:
            images.append(b64)
            pose_idx_out.append(pi)
    cap.release()
    return images, pose_idx_out


async def select_frames_with_window_ai(
    windows: list[dict[str, Any]],
    poses: list[dict],
    analysis_video_path: str,
    *,
    region: str,
    max_candidates: int = 5,
) -> dict[str, dict[str, Any]]:
    """One batched vision call for all ``PHASE_ORDER`` windows (same Modal run as the rest of Pro chain)."""
    _ = region
    from services.gemini_service import _call_vision_ai

    win_by_phase = {str(w.get("phase") or ""): w for w in windows}
    bundles: list[dict[str, Any]] = []
    all_images: list[str] = []

    for phase in PHASE_ORDER:
        w = win_by_phase.get(phase)
        if not w:
            bundles.append({"phase": phase, "empty": True, "n": 0, "pis": []})
            continue
        lo = int(w["start_pose_idx"])
        hi = int(w["end_pose_idx"])
        imgs, pis = _extract_candidates_for_window(
            analysis_video_path, poses, lo, hi, max_candidates,
        )
        if not imgs or not pis:
            center = int(w.get("center_pose_idx", (lo + hi) // 2))
            imgs2, pis2 = _extract_candidates_for_window(
                analysis_video_path, poses, center, center, 1,
            )
            if imgs2 and pis2:
                st = len(all_images)
                all_images.extend(imgs2)
                bundles.append({"phase": phase, "start": st, "n": len(imgs2), "pis": pis2, "empty": False})
            else:
                bundles.append({"phase": phase, "empty": True, "n": 0, "pis": [center]})
            continue
        st = len(all_images)
        all_images.extend(imgs)
        bundles.append({"phase": phase, "start": st, "n": len(imgs), "pis": pis, "empty": False})

    out: dict[str, dict[str, Any]] = {}

    def _center_fallback(phase: str) -> dict[str, Any]:
        w = win_by_phase.get(phase) or {}
        lo = int(w.get("start_pose_idx", 0))
        hi = int(w.get("end_pose_idx", 0))
        center = int(w.get("center_pose_idx", (lo + hi) // 2))
        return {
            "pose_idx": center,
            "confidence": 0.35,
            "short_reason": "no_candidates_used_center",
            "candidate_pose_indices": [center],
        }

    if not all_images:
        for ph in PHASE_ORDER:
            out[ph] = _center_fallback(ph)
        return out

    lines: list[str] = []
    for b in bundles:
        ph = str(b["phase"])
        if b.get("empty"):
            lines.append(f'- "{ph}": no candidate JPEGs — respond best_candidate_index=0 (local index).')
        else:
            s, n = int(b["start"]), int(b["n"])
            lines.append(
                f'- "{ph}": candidate JPEGs are concatenated images[{s}]..[{s + n - 1}] '
                f"({n} frames, temporal order). Local indices 0..{n - 1}."
            )
    manifest = "\n".join(lines)
    order_csv = ", ".join(PHASE_ORDER)
    prompt = f"""You are selecting the best keyframe (one pose index) per golf swing phase from candidate thumbnails.

{manifest}

Return ONLY valid JSON:
{{"picks": [
  {{"phase": "<phase_id>", "best_candidate_index": <int, 0-based within that phase's block>, "confidence": <0..1>, "short_reason": "<brief English>"}}
]}}
You MUST output exactly {len(PHASE_ORDER)} objects in picks, in this phase order: {order_csv}.
For phases marked "no candidate JPEGs", still output that phase with best_candidate_index 0 and confidence around 0.35.
"""

    by_phase: dict[str, dict[str, Any]] = {}
    try:
        batched_timeout = max(_PRO_WINDOW_AI_TIMEOUT_S * 2.5, 180.0)
        text, _prov, _slot = await _call_vision_ai(
            prompt,
            all_images,
            1536,
            0.15,
            "pro_window_batched",
            timeout_s=batched_timeout,
        )
        js = extract_json_from_response(text)
        picks_raw = js.get("picks")
        if isinstance(picks_raw, list):
            for item in picks_raw:
                if isinstance(item, dict) and item.get("phase"):
                    by_phase[str(item["phase"])] = item
    except Exception as exc:
        logger.warning("[PRO][window_ai] batched_call_failed err=%s — center fallbacks", exc)

    for b in bundles:
        phase = str(b["phase"])
        w = win_by_phase.get(phase) or {}
        lo = int(w.get("start_pose_idx", 0))
        hi = int(w.get("end_pose_idx", 0))
        if b.get("empty"):
            pis_fb = list(b.get("pis") or [])
            if pis_fb:
                c0 = int(pis_fb[0])
                out[phase] = {
                    "pose_idx": c0,
                    "confidence": 0.35,
                    "short_reason": "no_candidates_used_center",
                    "candidate_pose_indices": pis_fb,
                }
            else:
                out[phase] = _center_fallback(phase)
            logger.info("[PRO][window_ai] phase=%s fallback=center (empty bundle)", phase)
            continue

        pis = list(b["pis"])
        n = int(b["n"])
        pobj = by_phase.get(phase) or {}
        bi = int(pobj.get("best_candidate_index", n // 2))
        bi = max(0, min(n - 1, bi))
        chosen_pi = pis[bi]
        conf = float(pobj.get("confidence", 0.7))
        reason = str(pobj.get("short_reason", "batched_ai_pick"))[:200]
        out[phase] = {
            "pose_idx": int(chosen_pi),
            "confidence": round(conf, 4),
            "short_reason": reason,
            "candidate_pose_indices": pis,
        }
        logger.info(
            "[PRO][window_ai] phase=%s batched picked_pose_idx=%s conf=%.2f",
            phase,
            chosen_pi,
            conf,
        )

    for ph in PHASE_ORDER:
        if ph not in out:
            out[ph] = _center_fallback(ph)

    return out
