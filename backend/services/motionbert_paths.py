"""Unified MotionBERT checkpoint search paths (volume + image bake). Env STELLAR_MOTIONBERT_CHECKPOINT wins if file exists."""

from __future__ import annotations

# Order: canonical names first; Modal volume is /models.
MOTIONBERT_CHECKPOINT_CANDIDATES: tuple[str, ...] = (
    "/models/motionbert.pt",
    "/models/motionbert_jit.pt",
    "/models/motionbert.ts",
    "/models/MotionBERT.pt",
    "/opt/stellar-weights/motionbert.pt",
)
