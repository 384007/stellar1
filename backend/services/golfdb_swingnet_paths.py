"""SwingNet checkpoint path resolution — no torch import (safe for /health)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_swingnet_checkpoint_path() -> str:
    """First existing file: env, then baked ``/opt/stellar-weights``, volume ``/models``, ``backend/models``."""
    env = (os.getenv("STELLAR_SWINGNET_CHECKPOINT") or "").strip()
    if env:
        if os.path.isfile(env):
            return os.path.abspath(env)
        logger.warning(
            "[SwingNet] STELLAR_SWINGNET_CHECKPOINT missing on disk (%s) — trying defaults",
            env,
        )

    backend_root = Path(__file__).resolve().parents[1]
    # Modal: /models volume overrides image-baked /opt/stellar-weights (ops can swap weights without rebuild).
    candidates = [
        Path("/models/swingnet_1800.pth.tar"),
        Path("/models/swingnet_1800.pth"),
        Path("/opt/stellar-weights/swingnet_1800.pth.tar"),
        Path("/opt/stellar-weights/swingnet_1800.pth"),
        backend_root / "models" / "swingnet_1800.pth.tar",
        backend_root / "models" / "swingnet_1800.pth",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        sz = p.stat().st_size
        if sz < 50_000_000:
            logger.warning("[SwingNet] skip undersized checkpoint candidate %s bytes=%s", p, sz)
            continue
        return str(p.resolve())
    return ""


def swingnet_weights_configured() -> bool:
    return bool(resolve_swingnet_checkpoint_path())
