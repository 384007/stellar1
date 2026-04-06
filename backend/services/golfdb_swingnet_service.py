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
from lib.prov3.keyframes.decode_spacing import spread_keyframes_min_decode_gap
from lib.golfdb_swingnet.event_detector import EventDetector
from services.golfdb_swingnet_paths import resolve_swingnet_checkpoint_path

logger = logging.getLogger(__name__)

# Pro v3 A engine identifier (logs / debugging).
PROV3_A_ENGINE_ID = "wmcnally/golfdb:SwingNet"

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
    max_frames: int = 400,
) -> tuple[torch.Tensor, np.ndarray, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot_open_video:{video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if total <= 0:
        cap.release()
        raise ValueError("video_has_no_frames")
    n_target = min(total, max_frames)
    sample_indices = np.unique(np.linspace(0, total - 1, num=n_target, dtype=np.int64))

    frames: list[np.ndarray] = []
    for idx in sample_indices:
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


def _decode_frame_index_to_row(
    frame_index: int,
    sample_indices: np.ndarray,
    total_frames: int,
) -> int:
    """Map a **decode frame index** in the analysis MP4 to the nearest SwingNet sample row."""
    hi = max(0, int(total_frames) - 1)
    di = max(0, min(int(frame_index), hi))
    return int(np.argmin(np.abs(sample_indices.astype(np.float64) - float(di))))


def _keyframes_from_probs(
    probs: np.ndarray,
    sample_indices: np.ndarray,
) -> list[dict[str, Any]]:
    """Eight events: per-class argmax over time, enforce non-decreasing timeline rows."""
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


def run_swingnet_extract(
    video_path: str,
    *,
    analysis_id: str,
    analysis_fps: float = 240.0,
) -> list[dict[str, Any]] | None:
    """Return Pro v3 A-layer keyframe dicts, or None to signal fallback.

    ``frame_index`` values are **decode indices** in ``video_path`` (the 240fps analysis MP4).
    ``analysis_fps`` is accepted for API symmetry with preprocess; indices do not depend on OpenCV fps.
    """
    _ = analysis_fps
    if not swingnet_enabled():
        return None
    try:
        model, device = _load_model()
        batch, sample_indices, fps, total = _video_to_batch(video_path)
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
        kfs = _keyframes_from_probs(probs, sample_indices)
        span = max(int(total), int(np.max(sample_indices)) + 1) if len(sample_indices) else int(total)
        span = max(span, 1)
        kfs, _spread = spread_keyframes_min_decode_gap(kfs, span)
        if _spread:
            logger.info(
                "[SwingNet] decode-index spread applied span_frames=%s (avoid clustered keyframe thumbs)",
                span,
            )
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
    except Exception as exc:
        logger.warning("[SwingNet] A-path failed (%s) — heuristic fallback", exc)
        return None


def swingnet_b_refine(
    analysis_id: str,
    keyframes: list[dict[str, Any]],
    *,
    window: int = 8,
) -> list[dict[str, Any]]:
    """B-path: local per-class peak on SwingNet probs (same wmcnally/golfdb run as A)."""
    with _CTX_LOCK:
        ctx = _REFINE_CTX.pop(analysis_id, None)
    if not ctx:
        return keyframes
    probs = ctx["probs"]
    sample_indices = ctx["sample_indices"]
    fps = float(ctx["source_fps"])
    total = int(ctx["total_frames"])
    t = probs.shape[0]
    out: list[dict[str, Any]] = []
    for item in keyframes:
        try:
            k = EVENT_SEQUENCE.index(str(item.get("event_name")))
        except ValueError:
            out.append(dict(item))
            continue
        v = int(item.get("frame_index", 0))
        row = _decode_frame_index_to_row(v, sample_indices, total)
        lo = max(0, row - window)
        hi = min(t - 1, row + window)
        sub = probs[lo : hi + 1, k]
        best_off = int(np.argmax(sub))
        row_new = lo + best_off
        sf = int(sample_indices[row_new])
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
