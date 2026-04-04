"""SwingNet checkpoint path resolution — no torch import (safe for /health)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_swingnet_checkpoint_path() -> str:
    """First hit: ``STELLAR_SWINGNET_CHECKPOINT`` if file exists, then common paths.

    Searches: ``backend/models/*``, then Modal volume ``/models/*``.
    """
    env = (os.getenv("STELLAR_SWINGNET_CHECKPOINT") or "").strip()
    if env:
        if os.path.isfile(env):
            return os.path.abspath(env)
        logger.warning(
            "[SwingNet] STELLAR_SWINGNET_CHECKPOINT missing on disk (%s) — trying defaults",
            env,
        )

    backend_root = Path(__file__).resolve().parents[1]
    candidates = [
        backend_root / "models" / "swingnet_1800.pth.tar",
        backend_root / "models" / "swingnet_1800.pth",
        Path("/models/swingnet_1800.pth.tar"),
        Path("/models/swingnet_1800.pth"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return ""


def swingnet_weights_configured() -> bool:
    return bool(resolve_swingnet_checkpoint_path())
