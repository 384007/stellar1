from __future__ import annotations

import os
from pathlib import Path

from services.motionbert_paths import MOTIONBERT_CHECKPOINT_CANDIDATES
from services.provider_registry import role_log
from services.provider_schema import provider_result


def _resolve_motionbert_checkpoint() -> str | None:
    """STELLAR_MOTIONBERT_CHECKPOINT if file exists; else first hit under MOTIONBERT_CHECKPOINT_CANDIDATES."""
    env = (os.getenv("STELLAR_MOTIONBERT_CHECKPOINT") or "").strip()
    if env and Path(env).is_file():
        return env
    for p in MOTIONBERT_CHECKPOINT_CANDIDATES:
        if Path(p).is_file():
            return p
    return None


def run(poses: list[dict]) -> dict:
    try:
        import torch
    except Exception:
        role_log(
            f"[ROLE=MOTIONBERT] status=dependency_missing input_2d_frames={len(poses)} "
            "[STELLAR_PLUS_PIPELINE] motionbert_branch=dependency_missing continuing_with_mediapipe_fallback"
        )
        return provider_result(
            role="pose3d",
            provider_name="motionbert",
            status="dependency_missing",
            frame_count=len(poses),
            payload={},
            error_reason="dependency_missing",
        )
    env_ckpt = (os.getenv("STELLAR_MOTIONBERT_CHECKPOINT") or "").strip()
    ckpt = _resolve_motionbert_checkpoint()
    if env_ckpt and not ckpt:
        role_log(
            f"[ROLE=MOTIONBERT] status=checkpoint_missing path={env_ckpt!r} input_2d_frames={len(poses)} "
            "[STELLAR_PLUS_PIPELINE] motionbert_branch=checkpoint_missing continuing_with_mediapipe_fallback"
        )
        return provider_result(
            role="pose3d",
            provider_name="motionbert",
            status="checkpoint_missing",
            frame_count=len(poses),
            payload={},
            error_reason="checkpoint_missing",
        )
    if not ckpt:
        role_log(
            f"[ROLE=MOTIONBERT] status=checkpoint_unset input_2d_frames={len(poses)} "
            f"(no checkpoint on disk; try {MOTIONBERT_CHECKPOINT_CANDIDATES[0]} or STELLAR_MOTIONBERT_CHECKPOINT) "
            "[STELLAR_PLUS_PIPELINE] motionbert_branch=unset continuing_with_mediapipe_fallback"
        )
        return provider_result(
            role="pose3d",
            provider_name="motionbert",
            status="checkpoint_unset",
            frame_count=len(poses),
            payload={},
            error_reason="checkpoint_unset",
        )
    try:
        model = torch.jit.load(ckpt, map_location=(os.getenv("STELLAR_MOTIONBERT_DEVICE") or "cpu"))
        model.eval()
        seq = []
        for p in poses:
            joints = p.get("joints") or []
            seq.append([[float(j.get("x", 0.0)), float(j.get("y", 0.0))] for j in joints])
        if not seq:
            raise ValueError("empty_input_sequence")
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = model(x)
        y_list = y.squeeze(0).detach().cpu().tolist() if hasattr(y, "detach") else []
    except Exception as exc:
        role_log(
            f"[ROLE=MOTIONBERT] status=inference_failed input_2d_frames={len(poses)} err={exc} "
            "[STELLAR_PLUS_PIPELINE] motionbert_branch=inference_failed continuing_with_mediapipe_fallback"
        )
        return provider_result(
            role="pose3d",
            provider_name="motionbert",
            status="inference_failed",
            frame_count=len(poses),
            payload={},
            error_reason=str(exc),
        )
    role_log(
        f"[ROLE=MOTIONBERT] status=ok ckpt={ckpt!r} input_2d_frames={len(poses)} output_3d_frames={len(y_list)} "
        "[STELLAR_PLUS_PIPELINE] motionbert_branch=ok"
    )
    return provider_result(
        role="pose3d",
        provider_name="motionbert",
        status="ok",
        frame_count=len(poses),
        payload={"joints3d": y_list, "lift_confidence": 0.7},
    )
