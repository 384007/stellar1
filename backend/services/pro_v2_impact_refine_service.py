"""Pro v2 — OpenCV-only impact refine with two-pass window and late-strip constraints."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.pro_v2_frame_read import read_frame_bgr_seek

logger = logging.getLogger(__name__)


def _jpeg_b64(frame: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _roi_gray(frame: np.ndarray, ratio: float = 0.42) -> np.ndarray:
    h, w = frame.shape[:2]
    rw = max(16, int(w * ratio))
    rh = max(16, int(h * ratio))
    x0 = max(0, (w - rw) // 2)
    y0 = max(0, (h - rh) // 2)
    g = cv2.cvtColor(frame[y0 : y0 + rh, x0 : x0 + rw], cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def _lap_var(gray_f: np.ndarray) -> float:
    g8 = np.clip(gray_f, 0, 255).astype(np.uint8)
    return float(cv2.Laplacian(g8, cv2.CV_64F).var())


def _raw_impact_components(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    next_bgr: np.ndarray,
) -> tuple[float, float, float, float]:
    """burst sum, symmetry [0,1], direction consistency [0,1], transition magnitude."""
    gp, gc, gn = _roi_gray(prev_bgr), _roi_gray(curr_bgr), _roi_gray(next_bgr)
    d01 = float(np.mean(np.abs(gc - gp)))
    d12 = float(np.mean(np.abs(gn - gc)))
    burst = d01 + d12
    sym = 1.0 - min(1.0, abs(d01 - d12) / (burst + 1e-6))

    g8p = np.clip(gp, 0, 255).astype(np.uint8)
    g8c = np.clip(gc, 0, 255).astype(np.uint8)
    g8n = np.clip(gn, 0, 255).astype(np.uint8)
    flow1 = cv2.calcOpticalFlowFarneback(g8p, g8c, None, 0.5, 3, 9, 3, 5, 1.1, 0)
    flow2 = cv2.calcOpticalFlowFarneback(g8c, g8n, None, 0.5, 3, 9, 3, 5, 1.1, 0)
    m1 = float(np.mean(np.sqrt(flow1[..., 0] ** 2 + flow1[..., 1] ** 2)))
    m2 = float(np.mean(np.sqrt(flow2[..., 0] ** 2 + flow2[..., 1] ** 2)))
    dir_cons = 1.0 - min(1.0, abs(m1 - m2) / (m1 + m2 + 1e-6))

    lp, lc, ln = _lap_var(gp), _lap_var(gc), _lap_var(gn)
    trans = abs(lc - 0.5 * (lp + ln))
    return burst, sym, dir_cons, trans


def _pick_best_in_window(
    video_path: str,
    *,
    center_idx: int,
    radius_frames: int,
    fps: float,
    total_frames: int,
    lower_bound_idx: int,
    upper_bound_idx: int,
    max_samples: int = 32,
) -> dict[str, Any] | None:
    path = str(Path(video_path))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    lo = max(0, center_idx - radius_frames)
    hi = min(max(total_frames - 1, 0), center_idx + radius_frames)
    if hi - lo < 3:
        cap.release()
        return None

    span = hi - lo
    step = max(1, span // max_samples)
    indices: list[int] = []
    x = lo
    while x <= hi:
        indices.append(int(x))
        x += step
    if indices[-1] != hi:
        indices.append(hi)
    indices = sorted(set(indices))
    frames: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        frames.append((int(idx), fr.copy()))
    cap.release()

    if len(frames) < 3:
        return None

    bursts: list[float] = []
    syms: list[float] = []
    dirs: list[float] = []
    trans: list[float] = []
    triples: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    for i in range(1, len(frames) - 1):
        _, prev_bgr = frames[i - 1]
        idx, curr_bgr = frames[i]
        _, next_bgr = frames[i + 1]
        b, s, d, t = _raw_impact_components(prev_bgr, curr_bgr, next_bgr)
        bursts.append(b)
        syms.append(s)
        dirs.append(d)
        trans.append(t)
        triples.append((idx, curr_bgr))

    bm = max(bursts) + 1e-9
    tm = max(trans) + 1e-9
    best_i = 0
    best_score = -1.0
    candidate_rows: list[dict[str, Any]] = []
    for i in range(len(triples)):
        bn = bursts[i] / bm
        tn = trans[i] / tm
        grad = 1.0 - min(1.0, abs(bursts[i] - (0.5 * bm)) / (bm + 1e-9))
        idx_i = int(triples[i][0])
        in_band = 1.0 if lower_bound_idx < idx_i < upper_bound_idx else 0.15
        score = 0.33 * bn + 0.20 * syms[i] + 0.19 * dirs[i] + 0.16 * min(1.0, tn) + 0.12 * grad
        score *= in_band
        candidate_rows.append({"frame_index": idx_i, "score": score, "burst_n": bn, "sym": syms[i], "dir": dirs[i]})
        if score > best_score:
            best_score = score
            best_i = i

    idx, curr_bgr = triples[best_i]
    spread = float(np.percentile(bursts, 75) - np.percentile(bursts, 25))
    median_b = float(np.median(bursts))
    return {
        "frame_index": idx,
        "timestamp": round(idx / fps, 5) if fps > 0 else 0.0,
        "score": best_score,
        "image_base64": _jpeg_b64(curr_bgr),
        "weak_window": best_score < 0.26 or (max(bursts) < 1.15 * median_b and spread < 0.1 * bm),
        "burst_spread": spread,
        "candidates": sorted(candidate_rows, key=lambda x: float(x["score"]), reverse=True)[:5],
    }


def refine_impact_keyframe_only(
    analysis_video_path: str,
    keyframes: list[dict[str, Any]],
    *,
    window_s: float = 0.12,
) -> list[dict[str, Any]]:
    """Replace only impact using two-pass OpenCV score; enforce downswing<impact<follow-through."""
    out = [dict(k) for k in keyframes]
    imp_i = next((i for i, k in enumerate(out) if k.get("phase") == "impact"), None)
    if imp_i is None:
        return out

    rough_t = float(out[imp_i].get("timestamp") or 0.0)
    path = str(Path(analysis_video_path))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return out
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 240.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    phase_to_frame = {str(k.get("phase")): int(k.get("frame_index") or 0) for k in out}
    ds_idx = int(phase_to_frame.get("downswing", 0))
    ft_idx = int(phase_to_frame.get("follow_through", max(0, total - 1)))
    center_idx = max(0, min(total - 1, int(round(rough_t * fps))))
    lower_bound_idx = max(ds_idx + 1, 1)
    upper_bound_idx = min(ft_idx - 1, max(2, total - 2))
    if upper_bound_idx <= lower_bound_idx:
        upper_bound_idx = min(max(lower_bound_idx + 2, center_idx + 3), max(2, total - 2))
    r0 = max(2, int(round(window_s * fps)))

    pass1 = _pick_best_in_window(
        analysis_video_path,
        center_idx=center_idx,
        radius_frames=r0,
        fps=fps,
        total_frames=total,
        lower_bound_idx=lower_bound_idx,
        upper_bound_idx=upper_bound_idx,
    )
    picked = pass1
    pass2 = None
    if pass1 and pass1.get("weak_window"):
        r1 = max(r0 + 3, int(round(min(0.20, window_s * 1.7) * fps)))
        pass2 = _pick_best_in_window(
            analysis_video_path,
            center_idx=center_idx,
            radius_frames=r1,
            fps=fps,
            total_frames=total,
            lower_bound_idx=lower_bound_idx,
            upper_bound_idx=upper_bound_idx,
        )
        if pass2 and pass2.get("score", 0) >= float(pass1.get("score", 0)) * 0.9:
            picked = pass2

    best = picked
    if not best or not best.get("image_base64"):
        logger.info("[PRO_V2][IMPACT_REFINE] skipped no_candidate")
        return out

    fi = int(best["frame_index"])
    # Keep semantic ordering; if too close to follow-through, fallback to best viable candidate in-band.
    min_gap = max(3, int(round(0.015 * fps)))
    if fi >= ft_idx - min_gap:
        for cand in best.get("candidates", []):
            cidx = int(cand.get("frame_index", fi))
            if lower_bound_idx <= cidx <= ft_idx - min_gap:
                fi = cidx
                break
    fi = max(lower_bound_idx, min(fi, max(lower_bound_idx, ft_idx - min_gap)))

    logger.info(
        "[PRO_V2][IMPACT_REFINE] rough=(t=%.5f,f=%s) pass1=%s pass2=%s final=(f=%s,score=%.4f,status=%s)",
        rough_t,
        center_idx,
        None if not pass1 else {"f": pass1.get("frame_index"), "s": round(float(pass1.get("score", 0)), 4)},
        None if not pass2 else {"f": pass2.get("frame_index"), "s": round(float(pass2.get("score", 0)), 4)},
        fi,
        float(best.get("score", 0)),
        "expanded" if pass2 and picked is pass2 else "narrow",
    )

    frame = read_frame_bgr_seek(analysis_video_path, fi)
    out[imp_i]["timestamp"] = round(fi / fps, 5) if fps > 0 else float(best["timestamp"])
    out[imp_i]["frame_index"] = fi
    out[imp_i]["source_frame_index"] = fi
    out[imp_i]["source_pose_idx"] = fi
    out[imp_i]["image_base64"] = _jpeg_b64(frame) if frame is not None else str(best["image_base64"])
    return out
