"""Pro v3 **A / B** inference using **SwingNet** from **wmcnally/golfdb**.

- **A-path:** full-sequence SwingNet logits → eight event keyframes (+ top-k).
- **B-path:** local probability-peak refinement using A-path caches.

Weights are not committed. Fetch with ``backend/scripts/download_golfdb_swingnet_weights.sh`` (``pip install gdown``).

Resolution order:

1. ``STELLAR_SWINGNET_CHECKPOINT`` if the path exists.
2. Else ``backend/models/swingnet_1800.pth.tar`` or ``backend/models/swingnet_1800.pth``.

Upstream: https://github.com/wmcnally/golfdb — preprocessing matches ``test_video.py`` (160² letterbox, ImageNet norm).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.prov3.keyframes.constants import EVENT_SEQUENCE, TOP_K
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


def swingnet_checkpoint_path() -> str:
    """Resolved weight file (env → ``backend/models`` → ``/models`` on Modal)."""
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


def _source_to_virtual_frame(sf: int, source_fps: float, analysis_fps: float = 240.0) -> int:
    return int(round(float(sf) * analysis_fps / max(source_fps, 1e-6)))


def _virtual_to_row(
    virtual: int,
    source_fps: float,
    sample_indices: np.ndarray,
    total_frames: int,
) -> int:
    sf = int(round(virtual * max(source_fps, 1e-6) / 240.0))
    sf = max(0, min(sf, total_frames - 1))
    return int(np.argmin(np.abs(sample_indices - sf)))


def _keyframes_from_probs(
    probs: np.ndarray,
    sample_indices: np.ndarray,
    source_fps: float,
    total_frames: int,
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
                    "frame_index": _source_to_virtual_frame(sf, source_fps),
                    "confidence": round(float(probs[int(j), k]), 4),
                }
            )
        conf = round(float(probs[row, k]), 4)
        sf_main = int(sample_indices[row])
        keyframes.append(
            {
                "event_name": event_name,
                "frame_index": _source_to_virtual_frame(sf_main, source_fps),
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
    """Return Pro v3 A-layer keyframe dicts, or None to signal fallback."""
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
        kfs = _keyframes_from_probs(probs, sample_indices, fps, total)
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
        row = _virtual_to_row(v, fps, sample_indices, total)
        lo = max(0, row - window)
        hi = min(t - 1, row + window)
        sub = probs[lo : hi + 1, k]
        best_off = int(np.argmax(sub))
        row_new = lo + best_off
        sf = int(sample_indices[row_new])
        conf = round(float(probs[row_new, k]), 4)
        cloned = dict(item)
        cloned["frame_index"] = _source_to_virtual_frame(sf, fps)
        cloned["confidence"] = conf
        top_k = list(cloned.get("top_k_candidates") or [])
        if top_k:
            for c in top_k:
                if str(c.get("event_name")) == str(item.get("event_name")):
                    ri = _virtual_to_row(int(c.get("frame_index", 0)), fps, sample_indices, total)
                    ri = max(lo, min(hi, ri))
                    sf2 = int(sample_indices[ri])
                    c["frame_index"] = _source_to_virtual_frame(sf2, fps)
                    c["confidence"] = round(float(probs[ri, k]), 4)
        cloned["top_k_candidates"] = top_k
        out.append(cloned)
    return out


def clear_swingnet_ctx(analysis_id: str) -> None:
    with _CTX_LOCK:
        _REFINE_CTX.pop(analysis_id, None)
