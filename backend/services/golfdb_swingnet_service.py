"""Pro v3 **A / B** inference using **SwingNet** from **wmcnally/golfdb**.

- **A-path:** full-sequence SwingNet logits → eight event keyframes (+ top-k).
- **B-path:** local probability-peak refinement using A-path caches.

**Timeline:** inference runs on ``analysis_240fps.mp4`` (constant **240fps** from ``build_analysis_timeline``).
``frame_index`` on each keyframe is the **decoded frame number in that analysis file** (0 … N−1), same as
``generate_analysis_frames`` / UI strips. Wall-clock time ≈ ``frame_index / 240``.

Weights: baked at ``/opt/stellar-weights`` on Modal image build; volume ``/models``; local ``backend/models/``.
Non-Modal: auto-download from Google Drive on first model load unless ``STELLAR_SWINGNET_AUTO_DOWNLOAD=0``.

Upstream: https://github.com/wmcnally/golfdb — preprocessing matches ``test_video.py`` (160² letterbox, ImageNet norm).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.prov3.keyframes.constants import EVENT_SEQUENCE, TOP_K
from lib.golfdb_swingnet.event_detector import EventDetector
from services.golfdb_swingnet_paths import resolve_swingnet_checkpoint_path
from services.provider_registry import role_log

logger = logging.getLogger(__name__)

# Pro v3 A engine identifier (logs / debugging).
PROV3_A_ENGINE_ID = "wmcnally/golfdb:SwingNet"

# Viterbi: max distinct time rows considered per event (wider than TOP_K for joint search).
_VITERBI_ROWS_PER_EVENT = 14

# Minimum decode-frame gaps between consecutive events (Address→…→Finish).
_PAIR_DECODE_GAPS = (1, 1, 2, 4, 4, 2, 2)
# Global: Top→Impact >= 9, Impact→Finish >= 6 (decode indices).
_TOP_TO_IMPACT_MIN = 9
_IMPACT_TO_FINISH_MIN = 6

_NEG_INF = -1.0e18

# Populated by A-path for B-path local peak refinement (per analysis_id).
_REFINE_CTX: dict[str, dict[str, Any]] = {}
_CTX_LOCK = threading.Lock()

_MODEL: EventDetector | None = None
_MODEL_CKPT: str | None = None
_MODEL_DEVICE: torch.device | None = None
_MODEL_LOCK = threading.Lock()
_DOWNLOAD_LOCK = threading.Lock()
_DOWNLOAD_ATTEMPTED = False

_SWINGNET_DRIVE_URL = "https://drive.google.com/uc?id=1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW"


def _runtime_is_modal() -> bool:
    return (os.getenv("STELLAR_RUNTIME") or "").strip().lower() == "modal" or bool(
        (os.getenv("MODAL_REGION") or "").strip()
    )


def _auto_download_swingnet_allowed() -> bool:
    if _runtime_is_modal():
        return False
    v = (os.getenv("STELLAR_SWINGNET_AUTO_DOWNLOAD") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _ensure_swingnet_weights_download() -> None:
    """First Pro v3 A inference on non-Modal: fetch weights into backend/models if missing."""
    global _DOWNLOAD_ATTEMPTED
    if not _auto_download_swingnet_allowed():
        return
    if resolve_swingnet_checkpoint_path():
        return
    with _DOWNLOAD_LOCK:
        if resolve_swingnet_checkpoint_path():
            return
        if _DOWNLOAD_ATTEMPTED:
            return
        _DOWNLOAD_ATTEMPTED = True
        dest = Path(__file__).resolve().parents[1] / "models" / "swingnet_1800.pth.tar"
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "[SwingNet] no checkpoint — auto-download to %s (set STELLAR_SWINGNET_AUTO_DOWNLOAD=0 to skip)",
            dest,
        )
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gdown>=5.2"],
                check=True,
                timeout=300,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import gdown; gdown.download({_SWINGNET_DRIVE_URL!r}, {str(dest)!r}, quiet=False)",
                ],
                check=True,
                timeout=900,
            )
        except Exception as exc:
            logger.error("[SwingNet] auto-download failed: %s", exc)
            return
        if dest.is_file() and dest.stat().st_size > 50_000_000:
            logger.info("[SwingNet] auto-download OK bytes=%s", dest.stat().st_size)
        else:
            logger.error("[SwingNet] auto-download invalid or too small: %s", dest)


def swingnet_checkpoint_path() -> str:
    """Resolved weight file (env → baked/volume/local paths)."""
    return resolve_swingnet_checkpoint_path()


def swingnet_enabled() -> bool:
    return bool(swingnet_checkpoint_path())


def _device() -> torch.device:
    d = (os.getenv("STELLAR_SWINGNET_DEVICE") or "").strip().lower()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")


def _load_model() -> tuple[EventDetector, torch.device]:
    global _MODEL, _MODEL_CKPT, _MODEL_DEVICE
    _ensure_swingnet_weights_download()
    ckpt = swingnet_checkpoint_path()
    dev = _device()
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_CKPT == ckpt and _MODEL_DEVICE == dev:
            return _MODEL, dev
        if not ckpt or not os.path.isfile(ckpt):
            raise FileNotFoundError(f"SwingNet checkpoint missing: {ckpt!r}")
        model = EventDetector(
            width_mult=1.0,
            lstm_layers=1,
            lstm_hidden=256,
            bidirectional=True,
            dropout=False,
        )
        try:
            blob = torch.load(ckpt, map_location=dev, weights_only=False)
        except TypeError:
            blob = torch.load(ckpt, map_location=dev)
        state = blob.get("model_state_dict") if isinstance(blob, dict) else blob
        model.load_state_dict(state, strict=True)
        model.to(dev)
        model.eval()
        _MODEL = model
        _MODEL_CKPT = ckpt
        _MODEL_DEVICE = dev
        logger.info(
            "[SwingNet] %s loaded checkpoint=%s device=%s",
            PROV3_A_ENGINE_ID,
            ckpt,
            dev,
        )
        return model, dev


def _read_letterbox_rgb(cap: cv2.VideoCapture, input_size: int) -> np.ndarray | None:
    ok, img = cap.read()
    if not ok or img is None:
        return None
    h, w = img.shape[:2]
    ratio = input_size / max(h, w)
    new_size = (int(w * ratio), int(h * ratio))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    delta_w = input_size - new_size[0]
    delta_h = input_size - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    border = [0.406 * 255, 0.456 * 255, 0.485 * 255]
    b_img = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border
    )
    return cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)


def _video_to_batch(
    video_path: str,
    *,
    input_size: int = 160,
    max_frames: int = 1200,
) -> tuple[torch.Tensor, np.ndarray, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot_open_video:{video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if total <= 0:
        cap.release()
        raise ValueError("video_has_no_frames")
    duration_s = float(total / fps) if fps > 0 else 0.0
    short_target = int(round(duration_s * 200.0)) if duration_s > 0 else total
    mid_target = int(round(duration_s * 150.0)) if duration_s > 0 else total
    long_target = int(round(duration_s * 120.0)) if duration_s > 0 else total
    if duration_s <= 3.0:
        adaptive_target = max(640, short_target)
    elif duration_s <= 6.0:
        adaptive_target = max(800, mid_target)
    else:
        adaptive_target = max(900, long_target)
    n_target = min(total, max(480, min(max_frames, adaptive_target)))
    sample_indices = np.unique(np.linspace(0, total - 1, num=n_target, dtype=np.int64))

    frames: list[np.ndarray] = []
    n_samp = len(sample_indices)
    log_step = max(1, min(150, n_samp // 6 or 1))
    for i, idx in enumerate(sample_indices):
        if i == 0 or (i + 1) % log_step == 0 or i == n_samp - 1:
            role_log(
                f"[ROLE=LITE_PIPELINE] swingnet_frame_decode {i + 1}/{n_samp} "
                f"target_idx={int(idx)} ok_frames={len(frames)}"
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(idx)))
        rgb = _read_letterbox_rgb(cap, input_size)
        if rgb is None:
            continue
        frames.append(rgb)
    cap.release()
    if len(frames) < 8:
        raise ValueError("swingnet_too_few_frames")

    stack = np.stack(frames, axis=0).astype(np.float32) / 255.0
    t = torch.from_numpy(stack).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t = (t - mean) / std
    batch = t.unsqueeze(0)
    # Rebuild sample_indices aligned with successfully read frames — assume 1:1 if none skipped
    if len(frames) != len(sample_indices):
        sample_indices = sample_indices[: len(frames)]
    return batch, sample_indices.astype(np.int64), fps, total


def _validate_true240_timeline(video_path: str, analysis_fps: float) -> tuple[float, int]:
    if abs(float(analysis_fps) - 240.0) > 0.5:
        raise RuntimeError(f"true240_required: analysis_fps={analysis_fps}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot_open_analysis_video:{video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        raise RuntimeError(f"analysis_timeline_empty:{video_path}")
    if fps <= 0:
        raise RuntimeError("analysis_timeline_fps_unknown")
    if abs(fps - 240.0) > 0.5:
        raise RuntimeError(f"true240_required: video_fps={fps:.3f}")
    return fps, total


def _run_forward(model: EventDetector, batch: torch.Tensor, device: torch.device, seq_len: int) -> np.ndarray:
    n = batch.shape[1]
    raw_sl = int(os.getenv("STELLAR_SWINGNET_SEQ_LEN") or "64")
    seq_len = max(16, min(seq_len, raw_sl))
    with torch.no_grad():
        if n <= seq_len:
            logits = model(batch.to(device))
            return F.softmax(logits.float().cpu(), dim=1).numpy()
        parts: list[torch.Tensor] = []
        start = 0
        while start < n:
            chunk = batch[:, start : start + seq_len, :, :, :].to(device)
            if chunk.shape[1] == 0:
                break
            logits = model(chunk)
            parts.append(logits.float().cpu())
            start += seq_len
        full = torch.cat(parts, dim=0)
        return F.softmax(full, dim=1).numpy()


def _log_prob(p: float) -> float:
    x = float(p)
    if x <= 0.0:
        return _NEG_INF
    return float(np.log(max(x, 1e-12)))


def _local_peak_bonus(probs: np.ndarray, row: int, cls: int, radius: int = 4) -> float:
    """Small bonus when this row is a local maximum for class cls (reduces bogus Impact peaks)."""
    t = int(probs.shape[0])
    r = int(row)
    lo = max(0, r - radius)
    hi = min(t - 1, r + radius)
    sub = probs[lo : hi + 1, cls]
    if sub.size == 0:
        return 0.0
    if float(probs[r, cls]) >= float(np.max(sub)) - 1e-9:
        return 0.04
    return 0.0


def _candidate_rows_for_class(probs: np.ndarray, cls: int, max_rows: int) -> list[int]:
    col = probs[:, cls]
    order = np.argsort(-col)
    seen: set[int] = set()
    out: list[int] = []
    for j in order.flat:
        r = int(j)
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
        if len(out) >= max_rows:
            break
    return out


def _keyframes_from_probs_viterbi(
    probs: np.ndarray,
    sample_indices: np.ndarray,
) -> list[dict[str, Any]]:
    """Joint path over eight events with pairwise + global decode gaps (Viterbi on candidate rows)."""
    n_rows = int(probs.shape[0])
    if n_rows < 8:
        return _keyframes_from_probs_argmax_fallback(probs, sample_indices)

    max_c = min(n_rows, _VITERBI_ROWS_PER_EVENT)
    cand_rows: list[list[int]] = [_candidate_rows_for_class(probs, k, max_c) for k in range(8)]
    if any(len(x) == 0 for x in cand_rows):
        return _keyframes_from_probs_argmax_fallback(probs, sample_indices)

    cand_dec = [[int(sample_indices[r]) for r in rows] for rows in cand_rows]
    n_c = [len(x) for x in cand_rows]

    best: list[list[float]] = [[_NEG_INF] * n_c[k] for k in range(8)]
    back: list[list[int]] = [[-1] * n_c[k] for k in range(8)]
    top_dec: list[list[int]] = [[0] * n_c[k] for k in range(8)]
    imp_dec: list[list[int]] = [[0] * n_c[k] for k in range(8)]

    for j in range(n_c[0]):
        r0 = cand_rows[0][j]
        lp = _log_prob(float(probs[r0, 0]))
        best[0][j] = lp

    for k in range(1, 8):
        gap_need = int(_PAIR_DECODE_GAPS[k - 1])
        for j in range(n_c[k]):
            r_k = cand_rows[k][j]
            d_k = cand_dec[k][j]
            lp_k = _log_prob(float(probs[r_k, k]))
            if k in (3, 4, 5):
                lp_k += _local_peak_bonus(probs, r_k, k)
            for i in range(n_c[k - 1]):
                d_prev = cand_dec[k - 1][i]
                if d_k < d_prev + gap_need:
                    continue
                if k == 5:
                    t_top = top_dec[4][i]
                    if d_k < t_top + _TOP_TO_IMPACT_MIN:
                        continue
                if k == 7:
                    i_imp = imp_dec[6][i]
                    if d_k < i_imp + _IMPACT_TO_FINISH_MIN:
                        continue
                cand_score = best[k - 1][i] + lp_k
                slack = (d_k - d_prev) - gap_need
                if slack > 0:
                    cand_score += min(0.03, float(slack) * 1.2e-4)
                if cand_score > best[k][j]:
                    best[k][j] = cand_score
                    back[k][j] = i
                    if k < 3:
                        top_dec[k][j] = 0
                    elif k == 3:
                        top_dec[k][j] = d_k
                    else:
                        top_dec[k][j] = top_dec[k - 1][i]
                    if k < 5:
                        imp_dec[k][j] = 0
                    elif k == 5:
                        imp_dec[k][j] = d_k
                    else:
                        imp_dec[k][j] = imp_dec[k - 1][i]

    last = 7
    best_j = int(np.argmax(best[last]))
    if best[last][best_j] <= _NEG_INF / 2:
        logger.warning("[SwingNet] Viterbi found no feasible path — argmax fallback")
        return _keyframes_from_probs_argmax_fallback(probs, sample_indices)

    pick: list[int] = [0] * 8
    pick[last] = best_j
    for k in range(7, 0, -1):
        pick[k - 1] = back[k][pick[k]]

    keyframes: list[dict[str, Any]] = []
    for k in range(8):
        event_name = EVENT_SEQUENCE[k]
        row = cand_rows[k][pick[k]]
        conf = round(float(probs[row, k]), 4)
        sf_main = int(sample_indices[row])
        top_k: list[dict[str, Any]] = []
        order = np.argsort(probs[:, k])[-TOP_K:][::-1]
        for j in order:
            sf = int(sample_indices[int(j)])
            top_k.append(
                {
                    "event_name": event_name,
                    "frame_index": int(sf),
                    "confidence": round(float(probs[int(j), k]), 4),
                }
            )
        keyframes.append(
            {
                "event_name": event_name,
                "frame_index": int(sf_main),
                "confidence": conf,
                "top_k_candidates": top_k,
            }
        )
    return keyframes


def _keyframes_from_probs_argmax_fallback(
    probs: np.ndarray,
    sample_indices: np.ndarray,
) -> list[dict[str, Any]]:
    """Legacy: per-class argmax with row monotonicity (used only if Viterbi infeasible)."""
    n_rows = int(probs.shape[0])
    rows = [int(np.argmax(probs[:, k])) for k in range(8)]
    for i in range(1, 8):
        if rows[i] <= rows[i - 1]:
            rows[i] = min(rows[i - 1] + 1, n_rows - 1)

    keyframes: list[dict[str, Any]] = []
    for k, row in enumerate(rows):
        event_name = EVENT_SEQUENCE[k]
        top_k: list[dict[str, Any]] = []
        order = np.argsort(probs[:, k])[-TOP_K:][::-1]
        for j in order:
            sf = int(sample_indices[int(j)])
            top_k.append(
                {
                    "event_name": event_name,
                    "frame_index": int(sf),
                    "confidence": round(float(probs[int(j), k]), 4),
                }
            )
        conf = round(float(probs[row, k]), 4)
        sf_main = int(sample_indices[row])
        keyframes.append(
            {
                "event_name": event_name,
                "frame_index": int(sf_main),
                "confidence": conf,
                "top_k_candidates": top_k,
            }
        )
    return keyframes


def _decode_frame_index_to_row(
    frame_index: int,
    sample_indices: np.ndarray,
    total_frames: int,
) -> int:
    """Map a **decode frame index** in the analysis MP4 to the nearest SwingNet sample row."""
    hi = max(0, int(total_frames) - 1)
    di = max(0, min(int(frame_index), hi))
    return int(np.argmin(np.abs(sample_indices.astype(np.float64) - float(di))))


def run_swingnet_extract(
    video_path: str,
    *,
    analysis_id: str,
    analysis_fps: float = 240.0,
    max_extract_frames: int | None = None,
) -> list[dict[str, Any]] | None:
    """Return Pro v3 A-layer keyframe dicts, or None to signal fallback.

    ``frame_index`` values are **decode indices** in ``video_path`` (the 240fps analysis MP4).
    ``analysis_fps`` is accepted for API symmetry with preprocess; indices do not depend on OpenCV fps.

    ``max_extract_frames`` caps SwingNet decode/LSTM length (default: env ``STELLAR_SWINGNET_EXTRACT_MAX_FRAMES`` or 1200).
    Lite passes a lower cap via ``STELLAR_SWINGNET_LITE_MAX_FRAMES`` to finish before ~3–4m ingress cuts.
    """
    if not swingnet_enabled():
        return None
    try:
        role_log(f"[ROLE=LITE_PIPELINE] swingnet_extract_start analysis_id={analysis_id}")
        _validate_true240_timeline(video_path, analysis_fps)
        role_log("[ROLE=LITE_PIPELINE] swingnet_true240_ok loading_model_if_needed")
        model, device = _load_model()
        role_log(f"[ROLE=LITE_PIPELINE] swingnet_model_ready device={device} building_frame_batch")
        cap = max_extract_frames
        if cap is None:
            cap = int(os.getenv("STELLAR_SWINGNET_EXTRACT_MAX_FRAMES", "1200"))
        cap = max(64, min(int(cap), 2400))
        batch, sample_indices, fps, total = _video_to_batch(video_path, max_frames=cap)
        role_log(
            f"[ROLE=LITE_PIPELINE] swingnet_batch_ready T={batch.shape[1]} samples "
            f"running_lstm_forward seq_cap=64"
        )
        probs = _run_forward(model, batch, device, seq_len=64)
        if probs.shape[0] != len(sample_indices):
            logger.warning(
                "[SwingNet] prob_len=%s sample_len=%s — clipping",
                probs.shape[0],
                len(sample_indices),
            )
            m = min(probs.shape[0], len(sample_indices))
            probs = probs[:m]
            sample_indices = sample_indices[:m]
        kfs = _keyframes_from_probs_viterbi(probs, sample_indices)
        for row in kfs:
            fi = int(row.get("frame_index", 0))
            if fi < 0 or fi >= int(total):
                raise RuntimeError(f"analysis_timeline_index_out_of_range:{fi}/{total}")
        logger.info("[SwingNet] A-path joint Viterbi decode (official indices — no preview spread)")
        with _CTX_LOCK:
            _REFINE_CTX[analysis_id] = {
                "probs": probs,
                "sample_indices": sample_indices,
                "source_fps": fps,
                "total_frames": total,
            }
        logger.info(
            "[SwingNet] A-path ok engine=%s frames=%s analysis_id=%s",
            PROV3_A_ENGINE_ID,
            len(sample_indices),
            analysis_id,
        )
        return kfs
    except RuntimeError:
        raise
    except Exception as exc:
        logger.warning("[SwingNet] A-path failed (%s)", exc)
        return None


def swingnet_b_refine(
    analysis_id: str,
    keyframes: list[dict[str, Any]],
    *,
    window: int = 8,
) -> list[dict[str, Any]]:
    """B-path: local per-class peak on SwingNet probs (same wmcnally/golfdb run as A).

    Reads refine context with **peek** (no pop) so recovery passes can reuse the same prob tensor.
    Caller must ``clear_swingnet_ctx(analysis_id)`` when the analysis session ends.
    """
    with _CTX_LOCK:
        ctx = _REFINE_CTX.get(analysis_id)
    if not ctx:
        return keyframes
    probs = ctx["probs"]
    sample_indices = ctx["sample_indices"]
    _fps = float(ctx["source_fps"])
    total = int(ctx["total_frames"])
    t = probs.shape[0]
    _core = {"Top", "Mid-downswing", "Impact", "Finish"}
    out: list[dict[str, Any]] = []
    for item in keyframes:
        try:
            k = EVENT_SEQUENCE.index(str(item.get("event_name")))
        except ValueError:
            out.append(dict(item))
            continue
        evn = str(item.get("event_name"))
        w = int(window)
        if evn in _core:
            w = max(w, 20)
        v = int(item.get("frame_index", 0))
        row = _decode_frame_index_to_row(v, sample_indices, total)
        lo = max(0, row - w)
        hi = min(t - 1, row + w)
        sub = probs[lo : hi + 1, k]
        best_off = int(np.argmax(sub))
        row_new = lo + best_off
        sf = int(sample_indices[row_new])
        sf = max(0, min(sf, total - 1))
        conf = round(float(probs[row_new, k]), 4)
        cloned = dict(item)
        cloned["frame_index"] = int(sf)
        cloned["confidence"] = conf
        top_k = list(cloned.get("top_k_candidates") or [])
        if top_k:
            for c in top_k:
                if str(c.get("event_name")) == str(item.get("event_name")):
                    ri = _decode_frame_index_to_row(int(c.get("frame_index", 0)), sample_indices, total)
                    ri = max(lo, min(hi, ri))
                    sf2 = int(sample_indices[ri])
                    c["frame_index"] = int(sf2)
                    c["confidence"] = round(float(probs[ri, k]), 4)
        cloned["top_k_candidates"] = top_k
        out.append(cloned)
    return out


def clear_swingnet_ctx(analysis_id: str) -> None:
    with _CTX_LOCK:
        _REFINE_CTX.pop(analysis_id, None)
