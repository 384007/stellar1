"""
Stellar AI — **Lite-only** Modal worker.

Deploy: ``modal deploy modal_app_lite.py``

Resources: cpu=1, memory=4096 MiB, timeout=900s. Default Modal scaling: scale to zero when idle (cold start on next request; no keep_warm).

Uses **lite_image** (``backend/requirements-modal-lite.txt`` + CPU torch for SwingNet), not ``modal_app.image``.
SwingNet weights are **baked** into ``/opt/stellar-weights/swingnet_1800.pth.tar`` at image build (same ``tools/modal_bake_swingnet.py`` as main Modal).
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


def _lite_modal_build_metadata() -> dict[str, str]:
    """Values captured when `modal deploy` runs (local machine or CI)."""
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


_MODAL_BUILD = _lite_modal_build_metadata()
_STELLAR_SHA_FULL = str(_MODAL_BUILD.get("STELLAR_GIT_SHA") or "unknown")
_STELLAR_SHA_SHORT = (
    _STELLAR_SHA_FULL[:7] if len(_STELLAR_SHA_FULL) >= 7 else _STELLAR_SHA_FULL
)

MODAL_LITE_FUNCTION_TIMEOUT_S = 900
# Slim image + deferred heavy imports; 4G default. No keep_warm — idle workers scale to zero (cold start).
MODAL_LITE_MEMORY_MIB = 4096

# Persistent model storage: same volume name as main app (YOLO / MotionBERT / SwingNet on /models).
stellar_models_volume = modal.Volume.from_name("stellar-models", create_if_missing=True)


def _wire_stellar_model_paths() -> None:
    """Map volume + baked weights to STELLAR_* env (YOLO/MotionBERT; SwingNet via bake + optional volume override)."""
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


def _wire_swingnet_paths_lite() -> None:
    """Volume ``/models`` overrides image-baked ``/opt/stellar-weights`` SwingNet (same as main Modal)."""
    if (os.getenv("STELLAR_SWINGNET_CHECKPOINT") or "").strip():
        return
    for base in (Path("/models"), Path("/opt/stellar-weights")):
        for name in ("swingnet_1800.pth.tar", "swingnet_1800.pth"):
            p = base / name
            if p.is_file() and p.stat().st_size > 50_000_000:
                os.environ["STELLAR_SWINGNET_CHECKPOINT"] = str(p)
                print(
                    f"[modal-lite] SwingNet checkpoint={p} bytes={p.stat().st_size}",
                    flush=True,
                )
                return
            if p.is_file():
                print(
                    f"[modal-lite] SwingNet skip too-small file={p} bytes={p.stat().st_size}",
                    flush=True,
                )


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
    # SwingNet (lite keyframes) needs torch; keep it out of requirements-modal-lite.txt per ops split.
    .run_commands(
        "python -m pip install --no-cache-dir 'torch==2.1.0+cpu' "
        "--index-url https://download.pytorch.org/whl/cpu"
    )
    .run_commands(
        "python -m pip install --no-cache-dir --no-deps 'numpy==1.26.4' --force-reinstall"
    )
    # SwingNet (golfdb) for Lite A/B — bake ~63MB into /opt/stellar-weights (same script as modal_app.py).
    .run_commands("python -m pip install --no-cache-dir 'gdown>=5.2,<6'")
    .add_local_file(
        local_path=str(Path(__file__).resolve().parent / "tools" / "modal_bake_swingnet.py"),
        remote_path="/root/modal_bake_swingnet.py",
        copy=True,
    )
    .run_commands("python /root/modal_bake_swingnet.py")
    .env(
        {
            **_MODAL_BUILD,
            "MEDIAPIPE_DISABLE_GPU": "1",
            "GLOG_minloglevel": "3",
        }
    )
    .run_commands(
        f'echo "[modal-lite][build] STELLAR_COMMIT_SHORT={_STELLAR_SHA_SHORT} '
        f"STELLAR_COMMIT_FULL={_STELLAR_SHA_FULL} "
        f'BRANCH={_MODAL_BUILD.get("STELLAR_GIT_BRANCH")} '
        f'BUILD_TIME={_MODAL_BUILD.get("STELLAR_BUILD_TIME")} '
        f'memory_mib={MODAL_LITE_MEMORY_MIB}"'
    )
    .add_local_dir("backend", remote_path="/backend")
)

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
    # Default Modal behavior: no keep_warm / min_containers — cold start after idle scale-down.
)
@modal.asgi_app()
def fastapi_app_lite():
    os.environ["STELLAR_RUNTIME"] = "modal"
    os.environ["STELLAR_MODAL_LITE_WORKER"] = "1"
    # Intentionally do **not** set STELLAR_MODAL_PRO_V3_ONLY or Pro v3 ffmpeg overrides — lite worker is separate.

    _wire_stellar_model_paths()
    _wire_swingnet_paths_lite()

    _sha = os.environ.get("STELLAR_GIT_SHA", "unknown")
    print(
        f"[modal-lite] boot asgi=main_lite:app timeout_s={MODAL_LITE_FUNCTION_TIMEOUT_S} "
        f"cpu=1 memory_mib={MODAL_LITE_MEMORY_MIB} git_sha={_sha}",
        flush=True,
        file=sys.stderr,
    )

    if "/backend" not in sys.path:
        sys.path.insert(0, "/backend")

    # Defer cv2 / mediapipe / torch to first request — avoids startup OOM and signal termination.
    from main_lite import app as _app  # noqa: PLC0415

    print(
        "[modal-lite] startup completed main_lite imported (cv2/mediapipe not preloaded)",
        flush=True,
        file=sys.stderr,
    )
    return _app
