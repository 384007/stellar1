"""
Stellar AI — **Lite-only** Modal worker.

Deploy: ``modal deploy modal_app_lite.py``

Uses a **dedicated slim image** (``lite_image`` + ``backend/requirements-modal-lite.txt``): no MMAction2,
TensorFlow, DeepLabCut, SwingNet bake, or YOLO bake — avoids cold-start OOM / SIGKILL from heavy native
imports in the same process as ASGI startup.

ASGI: ``main_lite:app`` — no Plus / Pro v3 / stellar-pro routers.

Logs: ``modal app logs stellar-ai-lite --follow``
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Build metadata (duplicated from modal_app.py so we never ``import modal_app`` here — that would
# construct the full Pro/Plus image graph on every ``modal deploy modal_app_lite.py``).
# ---------------------------------------------------------------------------


def _modal_build_metadata() -> dict[str, str]:
    root = Path(__file__).resolve().parent

    def from_env(key: str) -> str | None:
        v = os.environ.get(key)
        if v is None:
            return None
        v = v.strip()
        return v or None

    sha = from_env("STELLAR_GIT_SHA")
    branch = from_env("STELLAR_GIT_BRANCH")
    build_time = from_env("STELLAR_BUILD_TIME")

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    if not sha:
        sha = git("rev-parse", "HEAD") or "unknown"
    if not branch:
        branch = git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    if not build_time:
        build_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "STELLAR_GIT_SHA": sha,
        "STELLAR_GIT_BRANCH": branch,
        "STELLAR_BUILD_TIME": build_time,
    }


_MODAL_BUILD = _modal_build_metadata()
_STELLAR_SHA_FULL = str(_MODAL_BUILD.get("STELLAR_GIT_SHA") or "unknown")
_STELLAR_SHA_SHORT = (
    _STELLAR_SHA_FULL[:7] if len(_STELLAR_SHA_FULL) >= 7 else _STELLAR_SHA_FULL
)

# ---------------------------------------------------------------------------
# Lite image — system libs for OpenCV headless + MediaPipe; ffmpeg for Lite pipeline preprocess.
# ---------------------------------------------------------------------------
lite_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "curl",
        "ffmpeg",
    )
    .run_commands(
        "ffmpeg -hide_banner -filters 2>&1 | grep -qw minterpolate || "
        '(echo "[modal-lite][build] FATAL: apt ffmpeg lacks minterpolate" && exit 1)'
    )
    .pip_install_from_requirements("backend/requirements-modal-lite.txt")
    .env({**_MODAL_BUILD})
    .run_commands(
        f'echo "[modal-lite][build] STELLAR_COMMIT_SHORT={_STELLAR_SHA_SHORT} '
        f'STELLAR_COMMIT_FULL={_STELLAR_SHA_FULL} '
        f'BRANCH={_MODAL_BUILD.get("STELLAR_GIT_BRANCH")} '
        f'BUILD_TIME={_MODAL_BUILD.get("STELLAR_BUILD_TIME")}"'
    )
    .add_local_dir("backend", remote_path="/backend")
)

# Same volume name as main app (YOLO / motionbert overrides on Lite if present).
stellar_models_volume = modal.Volume.from_name("stellar-models", create_if_missing=True)


def _wire_stellar_model_paths() -> None:
    """Map volume + baked weights to STELLAR_* env (Lite image has no baked YOLO path — volume only)."""
    baked_yolo = Path("/opt/stellar-weights/yolo11n.pt")
    vol_yolo = Path("/models/yolo11n.pt")
    if vol_yolo.is_file():
        os.environ["STELLAR_YOLO_WEIGHTS"] = str(vol_yolo)
    elif baked_yolo.is_file():
        os.environ.setdefault("STELLAR_YOLO_WEIGHTS", str(baked_yolo))

    if (os.getenv("STELLAR_MOTIONBERT_CHECKPOINT") or "").strip():
        return
    for p in (
        "/models/motionbert.pt",
        "/models/motionbert_jit.pt",
        "/models/motionbert.ts",
        "/models/MotionBERT.pt",
        "/opt/stellar-weights/motionbert.pt",
    ):
        if Path(p).is_file():
            os.environ["STELLAR_MOTIONBERT_CHECKPOINT"] = p
            print(f"[modal-lite] STELLAR_MOTIONBERT_CHECKPOINT={p}", flush=True)
            return


MODAL_LITE_FUNCTION_TIMEOUT_S = 900
# Raised temporarily to validate stability after removing heavy image + eager native preloads.
MODAL_LITE_MEMORY_MIB = 10240

app = modal.App(
    name="stellar-ai-lite",
    image=lite_image,
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

    _wire_stellar_model_paths()

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

    print(
        "[modal-lite] startup completed: ASGI app ready (cv2/mediapipe load on first use, not at startup)",
        flush=True,
        file=sys.stderr,
    )

    return _app
