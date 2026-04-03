"""Pro Stage 4: AI picks one candidate index per phase inside pre-cut motion windows only."""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from services.gemini_service import extract_json_from_response
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
    """Returns phase -> {pose_idx, confidence, short_reason, candidate_pose_indices}."""
    from services.gemini_service import _call_vision_ai

    out: dict[str, dict[str, Any]] = {}
    for w in windows:
        phase = str(w.get("phase") or "")
        lo = int(w["start_pose_idx"])
        hi = int(w["end_pose_idx"])
        imgs, pis = _extract_candidates_for_window(analysis_video_path, poses, lo, hi, max_candidates)
        if not imgs or not pis:
            center = int(w.get("center_pose_idx", (lo + hi) // 2))
            out[phase] = {
                "pose_idx": center,
                "confidence": 0.35,
                "short_reason": "no_candidates_used_center",
                "candidate_pose_indices": [center],
            }
            logger.info("[PRO][window_ai] phase=%s fallback=center idx=%s", phase, center)
            continue

        n = len(imgs)
        prompt = (
            f'Golf swing phase "{phase}". Frames are in temporal order with candidate indices '
            f"0..{n - 1}.\n"
            "Pick the single frame that best represents this phase for coaching.\n"
            "Return ONLY valid JSON: "
            '{"best_candidate_index": <int 0..'
            f"{n - 1}"
            '>, "confidence": <0..1>, "short_reason": "<brief English>"}\n'
            "Do not redefine phases or describe other parts of the swing."
        )
        try:
            text, _prov, _slot = await _call_vision_ai(
                prompt,
                imgs,
                512,
                0.15,
                f"pro_window_{phase}",
                timeout_s=_PRO_WINDOW_AI_TIMEOUT_S,
            )
            js = extract_json_from_response(text)
            bi = int(js.get("best_candidate_index", n // 2))
            bi = max(0, min(n - 1, bi))
            chosen_pi = pis[bi]
            conf = float(js.get("confidence", 0.7))
            reason = str(js.get("short_reason", "ai_pick"))[:200]
        except Exception as exc:
            logger.warning("[PRO][window_ai] phase=%s ai_fail=%s", phase, exc)
            bi = n // 2
            chosen_pi = pis[bi]
            conf = 0.45
            reason = "ai_error_center_bias"

        out[phase] = {
            "pose_idx": int(chosen_pi),
            "confidence": round(conf, 4),
            "short_reason": reason,
            "candidate_pose_indices": pis,
        }
        logger.info(
            "[PRO][window_ai] phase=%s stage=done picked_pose_idx=%s conf=%.2f",
            phase,
            chosen_pi,
            conf,
        )

    return out
