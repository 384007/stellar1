#!/usr/bin/env python3
"""Create minimal DeepLabCut project under deeplabcut_workspace/ and write .stellar_dlc_config.

Run from backend/ with deeplabcut + tensorflow importable. First import is slow (~1–2 min).

Training is still required before analyze_videos succeeds; keep STELLAR_RESEARCH_BACKEND=disabled until then.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# backend/
BACKEND = Path(__file__).resolve().parents[1]
# Modal image: STELLAR_DLC_WORKSPACE_ROOT=/opt/deeplabcut_workspace (see modal_app.py).
_workspace_env = (os.environ.get("STELLAR_DLC_WORKSPACE_ROOT") or "").strip()
WORK = Path(_workspace_env) if _workspace_env else BACKEND / "deeplabcut_workspace"
BOOT = WORK / "_bootstrap"
PLACEHOLDER = BOOT / "stellar_bootstrap_64x64.mp4"
MARKER = WORK / ".stellar_dlc_config"
OUT_DEFAULT = WORK / "outputs"


def _write_placeholder_video() -> None:
    import numpy as np

    BOOT.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(PLACEHOLDER), fourcc, 10.0, (64, 64))
        if not w.isOpened():
            raise RuntimeError("VideoWriter failed")
        for _ in range(30):
            w.write(np.zeros((64, 64, 3), dtype=np.uint8))
        w.release()
    except Exception as e:
        print("cv2 VideoWriter failed, trying imageio-ffmpeg:", e, file=sys.stderr)
        import imageio.v2 as imageio

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(30)]
        imageio.mimwrite(str(PLACEHOLDER), frames, fps=10, codec="libx264", quality=8)

    if not PLACEHOLDER.is_file() or PLACEHOLDER.stat().st_size < 50:
        raise SystemExit(f"Failed to create placeholder video at {PLACEHOLDER}")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(str(WORK if _workspace_env else BACKEND))
    if not PLACEHOLDER.is_file():
        print("Creating placeholder video…")
        _write_placeholder_video()

    import deeplabcut as dlc

    if MARKER.is_file():
        existing = MARKER.read_text().strip()
        if existing and Path(existing).is_file():
            print(f"Already bootstrapped: {existing}")
            return

    print("Creating DLC project (may take a minute on first tensorflow import)…")
    cfg_path = dlc.create_new_project(
        "stellar_research",
        "stellar",
        [str(PLACEHOLDER.resolve())],
        working_directory=str(WORK.resolve()),
        copy_videos=True,
        videotype=".mp4",
        multianimal=False,
    )
    if cfg_path == "nothingcreated":
        raise SystemExit("create_new_project failed (no valid videos?)")

    cfg_path = str(Path(cfg_path).resolve())
    MARKER.write_text(cfg_path + "\n", encoding="utf-8")
    OUT_DEFAULT.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {MARKER}")
    print(f"config.yaml: {cfg_path}")
    print("Train a model (DLC workflow) before enabling STELLAR_RESEARCH_BACKEND=deeplabcut.")


if __name__ == "__main__":
    main()
