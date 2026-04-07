"""
Stellar AI — **Lite-only** Modal worker.

Deploy: ``modal deploy modal_app_lite.py``

Resources: cpu=1, memory=8192 MiB, timeout=900s (memory above main 6144: cv2+mediapipe pre-load + lite pipeline peak RSS).

Uses the **same container image** as ``modal_app`` (import ``image`` + ``stellar_models_volume`` from there).
ASGI: ``main_lite:app`` — no Plus / Pro v3 / stellar-pro routers.

Logs: ``modal app logs stellar-ai-lite --follow``
"""
from __future__ import annotations

import os
import sys

import modal

from modal_app import (
    _wire_mmaction2_paths,
    _wire_stellar_model_paths,
    _wire_swingnet_paths,
    image,
    stellar_models_volume,
)

MODAL_LITE_FUNCTION_TIMEOUT_S = 900
# 6144 MiB 仍可能在预加载 cv2+mediapipe + 首次请求时 OOM（Modal: terminated by signal）。仅加内存，不改 CPU/预加载策略。
MODAL_LITE_MEMORY_MIB = 8192

app = modal.App(
    name="stellar-ai-lite",
    image=image,
    secrets=[modal.Secret.from_name("custom-secret")],
)


@app.function(
    cpu=1,
    memory=MODAL_LITE_MEMORY_MIB,
    timeout=MODAL_LITE_FUNCTION_TIMEOUT_S,
    volumes={"/models": stellar_models_volume},
)
@modal.asgi_app()
def fastapi_app_lite():
    os.environ["STELLAR_RUNTIME"] = "modal"
    os.environ["STELLAR_MODAL_LITE_WORKER"] = "1"
    # Intentionally do **not** set STELLAR_MODAL_PRO_V3_ONLY or Pro v3 ffmpeg overrides — lite worker is separate.

    _wire_stellar_model_paths()
    _wire_mmaction2_paths()
    _wire_swingnet_paths()

    _sha = os.environ.get("STELLAR_GIT_SHA", "unknown")
    print(
        f"[modal-lite] asgi=main_lite:app timeout_s={MODAL_LITE_FUNCTION_TIMEOUT_S} "
        f"cpu=1 memory={MODAL_LITE_MEMORY_MIB} git_sha={_sha}",
        flush=True,
        file=sys.stderr,
    )

    if "/backend" not in sys.path:
        sys.path.insert(0, "/backend")

    from main_lite import app as _app  # noqa: PLC0415

    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        print("[modal-lite] cv2 + mediapipe pre-load OK", flush=True)
    except Exception as e:
        print(f"[modal-lite] WARNING: cv2/mediapipe pre-load: {e}", flush=True)

    return _app
