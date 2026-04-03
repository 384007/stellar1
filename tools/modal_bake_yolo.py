#!/usr/bin/env python3
"""Bake yolo11n.pt into /opt/stellar-weights for Modal (Ultralytics cache layout varies)."""
from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    dest = Path("/opt/stellar-weights/yolo11n.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    import os

    os.chdir(dest.parent)
    from ultralytics import YOLO

    YOLO("yolo11n.pt")

    candidates: list[Path] = list(dest.parent.rglob("yolo11n.pt"))
    hc = Path.home() / ".cache"
    if hc.exists():
        candidates.extend(hc.rglob("yolo11n.pt"))

    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        raise SystemExit("yolo11n.pt not found after ultralytics download")
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    if not dest.is_file():
        raise SystemExit(f"failed to materialize {dest}")
    print("[build] yolo11n.pt ->", dest, "bytes=", dest.stat().st_size)


if __name__ == "__main__":
    main()
