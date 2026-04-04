#!/usr/bin/env python3
"""Bake SwingNet weights into /opt/stellar-weights for Modal (Pro v3 A/B).

Fails the image build if the file is missing or too small (no silent skip).
Upstream: https://github.com/wmcnally/golfdb — CC BY-NC 4.0.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

DEST = Path("/opt/stellar-weights/swingnet_1800.pth.tar")
MIN_BYTES = 50_000_000
FILE_ID = "1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW"
URL = f"https://drive.google.com/uc?id={FILE_ID}"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.is_file() and DEST.stat().st_size >= MIN_BYTES:
        print("[build] swingnet already baked OK bytes=", DEST.stat().st_size, flush=True)
        return

    try:
        import gdown  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(f"gdown required: {e}") from e

    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            print(f"[build] swingnet gdown attempt {attempt} -> {DEST}", flush=True)
            gdown.download(URL, str(DEST), quiet=False)
            if DEST.is_file() and DEST.stat().st_size >= MIN_BYTES:
                print("[build] swingnet OK bytes=", DEST.stat().st_size, flush=True)
                return
            if DEST.is_file():
                DEST.unlink(missing_ok=True)
            raise RuntimeError(
                f"download too small or missing (expected >={MIN_BYTES} bytes)"
            )
        except Exception as exc:
            last_err = exc
            print(f"[build] swingnet attempt {attempt} failed: {exc}", flush=True)
            time.sleep(min(30, 5 * attempt))

    raise SystemExit(f"swingnet bake failed after retries: {last_err}")


if __name__ == "__main__":
    main()
