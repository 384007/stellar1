"""
Stellar AI on Modal — deploy: `modal deploy modal_app.py` (or `tools/deploy_modal.sh` locally).

Live logs (CLI):
  modal app logs stellar-ai --follow

If the dashboard shows no logs, traffic may be hitting Render instead:
  curl -s "https://<your-modal-url>/health" | jq .runtime   # expect "modal"

Gemini 地区/出口请在运行进程的环境变量中配置（与 Modal 解耦），例如本机或 CI：
  HTTPS_PROXY / GEMINI_HTTPS_PROXY、或 GEMINI_BACKEND=vertex + VERTEX_AI_LOCATION=asia-southeast1

Build identity (baked into the image at `modal deploy` time):
  STELLAR_GIT_SHA, STELLAR_GIT_BRANCH, STELLAR_BUILD_TIME — override via env, else from `git` + UTC clock.
  Build logs also print ``[modal][build] STELLAR_COMMIT_SHORT=…`` (grep-friendly); full SHA is in ``ENV STELLAR_GIT_SHA=…``.

MMAction2 + MMPose + extras: `backend/requirements-modal.txt` (no torch/ultralytics), then `tools/modal_install_mmaction2.sh`
  (torch==2.1.0+cpu, mmcv 2.1 manylinux, mmengine, mmaction2 + TSN, mmpose==1.3.2), then `tools/modal_install_ml_extras.sh`
  (mmdet for MMPoseInferencer/RTMPose; TensorFlow + DeepLabCut + tensorpack/tf-slim; numpy repin).
  Then `pip install ultralytics --no-deps` + `psutil`, then `modal_bake_yolo.py`, then bootstrap DLC under `/opt/deeplabcut_workspace`.
  Build skips full init_recognizer (avoids PyTorch hub ResNet download during image build).
  At runtime `fastapi_app` sets STELLAR_MMACTION2_* unless STELLAR_ACTION_BACKEND=disabled.
  Optional volume: `/models/tsn_kinetics400.pth` (and matching config name) overrides baked weights.

ASGI entry: ``main:app``. Pro v3 路由均在 ``/pro-v3`` 下：``POST /pro-v3/analyze``、``GET /pro-v3/media/...``、``POST /pro-v3/keyframes/*``（见 ``routers.prov3_api``）。``STELLAR_MODAL_PRO_V3_ONLY=1`` 时不加载旧 ``/stellar-pro/analyze``。
``fastapi_app`` 使用 ``timeout=3600``；若未设置 ``STELLAR_PROV3_USE_FAST_240FPS``，默认 ``1``（跳过 minterpolate，避免 300s 级超时与取消后 runner 挂起）。
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal


def _modal_build_metadata() -> dict[str, str]:
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


_MODAL_BUILD = _modal_build_metadata()
# Human-readable in `modal deploy` build logs (ENV step only shows raw hex).
_STELLAR_SHA_FULL = str(_MODAL_BUILD.get("STELLAR_GIT_SHA") or "unknown")
_STELLAR_SHA_SHORT = (
    _STELLAR_SHA_FULL[:7] if len(_STELLAR_SHA_FULL) >= 7 else _STELLAR_SHA_FULL
)

# Single source of truth for @app.function(timeout=...). Modal default is 300s; Pro analyze needs more.
# If logs still say "timeout of 300s", the deployed app was not built from this `modal_app.py` (re-run `modal deploy`).
MODAL_FASTAPI_FUNCTION_TIMEOUT_S = 3600

# ---------------------------------------------------------------------------
# Image – install system libs needed by mediapipe/opencv, then Python deps
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "curl",
        # decord / native wheels occasionally fall back to compile on slim images
        "build-essential",
        "ffmpeg",
    )
    .pip_install_from_requirements("backend/requirements-modal.txt")
    .run_commands(
        'python -c "'
        "import os; os.environ['MEDIAPIPE_DISABLE_GPU']='1'; os.environ['GLOG_minloglevel']='3';"
        "import numpy as np; import mediapipe as mp;"
        "p = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2);"
        "p.process(np.zeros((100,100,3), dtype=np.uint8)); p.close();"
        "print('[build] MediaPipe Pose model pre-warmed OK')"
        '"'
    )
    # MMAction2 (CPU) before YOLO bake: torch 2.1+cpu + mmcv 2.1 wheel (mmaction2 requires mmcv<2.2).
    # copy=True: bake step must run at image build time; Modal forbids run_commands after
    # add_local_* unless the file is copied into the image (see Modal Image.add_local_* docs).
    .add_local_dir(
        local_path=str(Path(__file__).resolve().parent / "backend" / "vendor" / "mmaction2_localizers_drn"),
        remote_path="/root/mmaction2_localizers_drn",
        copy=True,
    )
    .add_local_file(
        local_path=str(
            Path(__file__).resolve().parent
            / "backend"
            / "mmaction_configs"
            / "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
        ),
        remote_path="/opt/mmaction_configs/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py",
        copy=True,
    )
    .add_local_file(
        local_path=str(Path(__file__).resolve().parent / "tools" / "modal_install_mmaction2.sh"),
        remote_path="/root/modal_install_mmaction2.sh",
        copy=True,
    )
    .add_local_file(
        local_path=str(Path(__file__).resolve().parent / "tools" / "modal_install_ml_extras.sh"),
        remote_path="/root/modal_install_ml_extras.sh",
        copy=True,
    )
    .run_commands("chmod +x /root/modal_install_mmaction2.sh && bash /root/modal_install_mmaction2.sh")
    .run_commands("chmod +x /root/modal_install_ml_extras.sh && bash /root/modal_install_ml_extras.sh")
    .run_commands(
        "python -m pip install --no-cache-dir --use-deprecated=legacy-resolver --no-deps ultralytics==8.3.55 && "
        "python -m pip install --no-cache-dir --use-deprecated=legacy-resolver 'psutil>=5.9,<7'"
    )
    .add_local_file(
        local_path=str(Path(__file__).resolve().parent / "tools" / "modal_bake_yolo.py"),
        remote_path="/root/modal_bake_yolo.py",
        copy=True,
    )
    .run_commands("python /root/modal_bake_yolo.py")
    # Pro v3 A/B: SwingNet (wmcnally/golfdb) — bake weights into image (~63MB). Script retries gdown + fails build if missing.
    .run_commands("python -m pip install --no-cache-dir 'gdown>=5.2,<6'")
    .add_local_file(
        local_path=str(Path(__file__).resolve().parent / "tools" / "modal_bake_swingnet.py"),
        remote_path="/root/modal_bake_swingnet.py",
        copy=True,
    )
    .run_commands("python /root/modal_bake_swingnet.py")
    # psutil/ultralytics pip may satisfy numpy from PyPI; keep 1.26.x for torch+mmpy/mediapipe.
    .run_commands(
        "python -m pip install --no-cache-dir --use-deprecated=legacy-resolver --no-deps "
        "'numpy==1.26.4' --force-reinstall"
    )
    .add_local_file(
        local_path=str(
            Path(__file__).resolve().parent / "backend" / "scripts" / "bootstrap_deeplabcut_workspace.py"
        ),
        remote_path="/root/bootstrap_deeplabcut_workspace.py",
        copy=True,
    )
    .run_commands(
        "mkdir -p /opt/deeplabcut_workspace && "
        "STELLAR_DLC_WORKSPACE_ROOT=/opt/deeplabcut_workspace python /root/bootstrap_deeplabcut_workspace.py"
    )
    .env({**_MODAL_BUILD, "STELLAR_YOLO_WEIGHTS": "/opt/stellar-weights/yolo11n.pt"})
    .run_commands(
        f'echo "[modal][build] STELLAR_COMMIT_SHORT={_STELLAR_SHA_SHORT} '
        f'STELLAR_COMMIT_FULL={_STELLAR_SHA_FULL} '
        f'BRANCH={_MODAL_BUILD.get("STELLAR_GIT_BRANCH")} '
        f'BUILD_TIME={_MODAL_BUILD.get("STELLAR_BUILD_TIME")} '
        f"PRO_API=/pro-v3/analyze STELLAR_MODAL_PRO_V3_ONLY=1 "
        f'FN_TIMEOUT_S={MODAL_FASTAPI_FUNCTION_TIMEOUT_S}"'
    )
    # Keep backend mount last so local backend edits do not invalidate the whole image build.
    .add_local_dir("backend", remote_path="/backend")
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App(
    name="stellar-ai",
    image=image,
    secrets=[modal.Secret.from_name("custom-secret")],
)

# Persistent model storage: mount `stellar-models` at /models.
# - YOLO: optional override `/models/yolo11n.pt` (wins over image-baked `/opt/stellar-weights/yolo11n.pt`).
# - MotionBERT (optional): TorchScript JIT at `/models/motionbert.pt` upgrades 3D lift; otherwise Plus uses MediaPipe world landmarks automatically.
# - Pro v3 / SwingNet (wmcnally/golfdb): upload `swingnet_1800.pth.tar` to the volume as `/models/swingnet_1800.pth.tar`
#   (backend resolves this path automatically; optional `STELLAR_SWINGNET_CHECKPOINT` override).
stellar_models_volume = modal.Volume.from_name("stellar-models", create_if_missing=True)


def _wire_mmaction2_paths() -> None:
    """Enable MMAction2 TSN when baked image and/or /models volume provides weights + config."""
    from pathlib import Path

    if (os.getenv("STELLAR_ACTION_BACKEND") or "").strip().lower() == "disabled":
        return
    cfg_baked = Path("/opt/mmaction_configs/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py")
    ck_baked = Path("/opt/mmaction_models/tsn_kinetics400.pth")
    cfg_vol = Path("/models/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py")
    ck_vol = Path("/models/tsn_kinetics400.pth")
    cfg = cfg_vol if cfg_vol.is_file() else cfg_baked
    ckpt = ck_vol if ck_vol.is_file() else ck_baked
    if not cfg.is_file() or not ckpt.is_file():
        return
    os.environ.setdefault("STELLAR_ACTION_BACKEND", "mmaction2")
    os.environ["STELLAR_MMACTION2_CONFIG"] = str(cfg)
    os.environ["STELLAR_MMACTION2_CHECKPOINT"] = str(ckpt)
    os.environ.setdefault("STELLAR_MMACTION2_DEVICE", "cpu")
    os.environ.setdefault("STELLAR_MMACTION2_SLIDING_WINDOWS", "2")
    print(f"[modal] MMAction2 enabled config={cfg} checkpoint={ckpt}", flush=True)


def _wire_stellar_model_paths() -> None:
    """Map volume + baked weights to STELLAR_* env so providers resolve without extra dashboard config."""
    from pathlib import Path

    baked_yolo = Path("/opt/stellar-weights/yolo11n.pt")
    vol_yolo = Path("/models/yolo11n.pt")
    if vol_yolo.is_file():
        os.environ["STELLAR_YOLO_WEIGHTS"] = str(vol_yolo)
    elif baked_yolo.is_file():
        os.environ.setdefault("STELLAR_YOLO_WEIGHTS", str(baked_yolo))

    if (os.getenv("STELLAR_MOTIONBERT_CHECKPOINT") or "").strip():
        return
    # Keep in sync with backend/services/motionbert_paths.py MOTIONBERT_CHECKPOINT_CANDIDATES
    for p in (
        "/models/motionbert.pt",
        "/models/motionbert_jit.pt",
        "/models/motionbert.ts",
        "/models/MotionBERT.pt",
        "/opt/stellar-weights/motionbert.pt",
    ):
        if Path(p).is_file():
            os.environ["STELLAR_MOTIONBERT_CHECKPOINT"] = p
            print(f"[modal] STELLAR_MOTIONBERT_CHECKPOINT={p}", flush=True)
            return


def _wire_swingnet_paths() -> None:
    """Set ``STELLAR_SWINGNET_CHECKPOINT`` when a file exists: volume ``/models`` wins over baked ``/opt/stellar-weights``."""
    from pathlib import Path

    if (os.getenv("STELLAR_SWINGNET_CHECKPOINT") or "").strip():
        return
    for base in (Path("/models"), Path("/opt/stellar-weights")):
        for name in ("swingnet_1800.pth.tar", "swingnet_1800.pth"):
            p = base / name
            if p.is_file() and p.stat().st_size > 50_000_000:
                os.environ["STELLAR_SWINGNET_CHECKPOINT"] = str(p)
                print(f"[modal] Pro v3 SwingNet checkpoint={p} bytes={p.stat().st_size}", flush=True)
                return
            if p.is_file():
                print(f"[modal] Pro v3 SwingNet skip too-small file={p} bytes={p.stat().st_size}", flush=True)


# Scaling: no keep_warm / min_containers — idle workers scale to zero (cold start on next request).
# Do not add warm pools here without an explicit product decision.
# Default Modal function timeout is 300s; Pro v3 (240fps ffmpeg + ML) can exceed that — raise cap (Modal allows up to 24h).
@app.function(
    cpu=1,
    # PyTorch + TensorFlow + MMAction in one worker; 4GiB can OOM on cold import.
    memory=6144,
    timeout=MODAL_FASTAPI_FUNCTION_TIMEOUT_S,
    volumes={"/models": stellar_models_volume},
)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    os.environ["STELLAR_RUNTIME"] = "modal"
    # Pro on Modal: slim router set — no legacy /stellar-pro; HTTP Pro is /pro-v3 only (see routers.prov3_api).
    os.environ["STELLAR_MODAL_PRO_V3_ONLY"] = "1"
    # Single-flight lock (409 on second POST) is off by default on Modal — avoids false 409 from retries / overlap.
    # To enforce one analyze per worker: set Modal secret STELLAR_PROV3_ANALYZE_SINGLE_FLIGHT=1
    if "STELLAR_PROV3_ANALYZE_SINGLE_FLIGHT" not in os.environ:
        os.environ["STELLAR_PROV3_ANALYZE_SINGLE_FLIGHT"] = "0"
    # minterpolate (MCI) on 1 CPU is slow enough to hit Modal's old 300s wall; cancellation leaves ffmpeg in a thread
    # and can block runner shutdown. Fast 240fps (fps dup) is the safe default; set STELLAR_PROV3_USE_FAST_240FPS=0 for MCI.
    if "STELLAR_PROV3_USE_FAST_240FPS" not in os.environ:
        os.environ["STELLAR_PROV3_USE_FAST_240FPS"] = "1"
    _wire_stellar_model_paths()
    _wire_mmaction2_paths()
    _wire_swingnet_paths()

    from pathlib import Path as _Path

    for _p in (
        _Path("/models/swingnet_1800.pth.tar"),
        _Path("/opt/stellar-weights/swingnet_1800.pth.tar"),
    ):
        _ok = _p.is_file()
        _sz = _p.stat().st_size if _ok else 0
        print(f"[modal] swingnet probe path={_p} exists={_ok} bytes={_sz}", flush=True)

    _sha = os.environ.get("STELLAR_GIT_SHA", "unknown")
    _branch = os.environ.get("STELLAR_GIT_BRANCH", "unknown")
    _bt = os.environ.get("STELLAR_BUILD_TIME", "unknown")
    _line = f"[modal] build_info git_sha={_sha} branch={_branch} build_time={_bt}"
    print(_line, flush=True, file=sys.stderr)
    print(
        f"[modal] function timeout in modal_app.py: {MODAL_FASTAPI_FUNCTION_TIMEOUT_S}s "
        "(Modal default is 300s; if tasks still cancel at 300s, redeploy with `modal deploy modal_app.py`.)",
        flush=True,
        file=sys.stderr,
    )
    _sf = os.environ.get("STELLAR_PROV3_ANALYZE_SINGLE_FLIGHT", "0")
    _fast = os.environ.get("STELLAR_PROV3_USE_FAST_240FPS", "1")
    _pro = (
        f"[modal] pro_v3_api POST /pro-v3/analyze GET /pro-v3/media/* POST /pro-v3/keyframes/* asgi=main:app "
        f"STELLAR_MODAL_PRO_V3_ONLY=1 STELLAR_PROV3_ANALYZE_SINGLE_FLIGHT={_sf} "
        f"STELLAR_PROV3_USE_FAST_240FPS={_fast} actual_sha={_sha}"
    )
    print(_pro, flush=True, file=sys.stderr)

    if "/backend" not in sys.path:
        sys.path.insert(0, "/backend")

    from main import app as _app  # noqa: PLC0415

    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        print("[modal] cv2 + mediapipe loaded OK", flush=True)
    except Exception as e:
        print(f"[modal] WARNING: failed to pre-load cv2/mediapipe: {e}", flush=True)

    return _app
