import asyncio
import contextlib
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# Suppress C++ library warnings (absl/glog/TensorFlow) at the OS fd level.
# These are printed before Python-level env vars take effect.
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

try:
    import absl.logging  # noqa: E402
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.use_absl_handler()
except ImportError:
    pass


@contextlib.contextmanager
def _suppress_native_stderr():
    """Redirect fd-level stderr to /dev/null to silence C++ library noise."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.responses import Response  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
import httpx  # noqa: E402

load_dotenv()


def detect_runtime() -> str:
    """
    Which platform is serving this process — no manual curl needed.

    Priority: explicit STELLAR_RUNTIME → Modal-injected env → Render → local dev.
    """
    explicit = (os.getenv("STELLAR_RUNTIME") or "").strip().lower()
    if explicit in ("modal", "render", "local"):
        return explicit
    for key in (
        "MODAL_TASK_ID",
        "MODAL_CONTAINER_ID",
        "MODAL_IS_SUBTASK",
        "MODAL_REGION",
        "MODAL_ENVIRONMENT",
    ):
        if os.getenv(key):
            return "modal"
    if os.getenv("RENDER"):
        return "render"
    return "local"


def _modal_echo(msg: str) -> None:
    """Modal dashboard / `modal app logs` can attach to stdout or stderr depending on version; mirror both."""
    print(msg, flush=True)
    print(msg, flush=True, file=sys.stderr)


def _verify_ffmpeg_at_startup() -> bool:
    """Resolve ffmpeg (PATH, env, or imageio-ffmpeg bundle), log -version once; return False if unavailable."""
    from services.internal.prov3_ffmpeg import FFmpegNotFoundError, ffmpeg_bin

    log = logging.getLogger("startup")
    try:
        exe = ffmpeg_bin()
    except FFmpegNotFoundError as e:
        msg = str(e)
        log.error("[ffmpeg] %s", msg)
        if detect_runtime() in ("modal", "render"):
            raise RuntimeError(msg) from e
        return False
    try:
        proc = subprocess.run(
            [exe, "-version"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        combined = ((proc.stdout or "") + (proc.stderr or "")).strip()
        first_line = combined.splitlines()[0] if combined else "(no output)"
        if proc.returncode != 0:
            msg = f"ffmpeg -version exit={proc.returncode} stderr={proc.stderr!r} stdout={proc.stdout!r}"
            log.error("[ffmpeg] %s", msg)
            if detect_runtime() in ("modal", "render"):
                raise RuntimeError(msg)
            return False
        log.info("[ffmpeg] %s", first_line)
        if detect_runtime() == "modal":
            _modal_echo(f"[ffmpeg] {first_line}")
    except FileNotFoundError:
        msg = "ffmpeg binary missing after resolve"
        log.error("[ffmpeg] %s", msg)
        if detect_runtime() in ("modal", "render"):
            raise RuntimeError(msg) from None
        return False
    except subprocess.TimeoutExpired:
        msg = "ffmpeg -version timed out"
        log.error("[ffmpeg] %s", msg)
        if detect_runtime() in ("modal", "render"):
            raise RuntimeError(msg)
        return False
    return True


def _configure_modal_logging() -> None:
    """Route Python logging to stderr; Modal often surfaces stderr more reliably than stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[stellar-modal] %(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    for name in (
        "routers",
        "services",
        "keepalive",
        "startup",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ):
        logging.getLogger(name).setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


if detect_runtime() == "modal":
    _configure_modal_logging()
    _modal_echo("[stellar-modal] main.py loaded — per-request access lines follow on each HTTP call")


def _verify_ffmpeg_cli() -> None:
    """
    Ensure resolved ffmpeg (PATH / STELLAR_FFMPEG_BINARY / imageio-ffmpeg) is runnable.
    Logs ffmpeg -version output once; raises on failure.
    """
    from services.internal.prov3_ffmpeg import ffmpeg_bin

    log = logging.getLogger("startup")
    try:
        proc = subprocess.run(
            [ffmpeg_bin(), "-version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as e:
        msg = "ffmpeg binary not executable after resolve"
        print(f"[stellar-ai] ERROR {msg}", flush=True, file=sys.stderr)
        log.error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        msg = "ffmpeg -version timed out after 30s"
        print(f"[stellar-ai] ERROR {msg}", flush=True, file=sys.stderr)
        log.error(msg)
        raise RuntimeError(msg) from e

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    blob = err or out
    if proc.returncode != 0:
        detail = (err or out)[:4000] or "(no stdout/stderr)"
        msg = f"ffmpeg -version failed with exit code {proc.returncode}: {detail}"
        print(f"[stellar-ai] ERROR {msg}", flush=True, file=sys.stderr)
        log.error(msg)
        raise RuntimeError(msg)

    lines = blob.splitlines() if blob else []
    head = "\n".join(lines[:8]) if lines else "(no output)"
    banner = f"[stellar-ai] ffmpeg -version:\n{head}"
    print(banner, flush=True)
    log.info("ffmpeg OK (first lines):\n%s", head)
    if detect_runtime() == "modal":
        _modal_echo(f"[stellar-modal] ffmpeg probe OK — first line: {lines[0] if lines else '(empty)'}")


# Modal: deploy manually with `modal deploy modal_app.py` or tools/deploy_modal.sh

_keepalive_logger = logging.getLogger("keepalive")
_KEEPALIVE_INTERVAL = 10 * 60  # ping self every 10 minutes


async def _keepalive_loop(base_url: str) -> None:
    """Periodically ping /health to prevent Render free-tier sleep."""
    await asyncio.sleep(60)  # wait for full startup before first ping
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.get(f"{base_url}/health")
                _keepalive_logger.info("keep-alive ping → HTTP %d", resp.status_code)
            except Exception as exc:
                _keepalive_logger.warning("keep-alive ping failed: %s", exc)
            await asyncio.sleep(_KEEPALIVE_INTERVAL)


app = FastAPI(
    title="Stellar AI API",
    version="1.1.0",
    description="AI-powered golf swing analysis platform — Plus advanced diagnosis",
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


@app.on_event("startup")
async def _startup():
    """Launch background keep-alive loop and pre-warm heavy models."""
    logging.getLogger("services.gemini_service").setLevel(logging.INFO)
    ffmpeg_ok = _verify_ffmpeg_at_startup()
    if ffmpeg_ok:
        await asyncio.to_thread(_verify_ffmpeg_cli)

    render_service_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if render_service_url:
        asyncio.create_task(_keepalive_loop(render_service_url))
        _keepalive_logger.info("Keep-alive started → pinging %s every %ds", render_service_url, _KEEPALIVE_INTERVAL)

    import threading

    def _warmup_mediapipe():
        try:
            with _suppress_native_stderr():
                import numpy as np
                import mediapipe as _mp
                _pose = _mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=0.5,
                )
                _pose.process(np.zeros((100, 100, 3), dtype=np.uint8))
                _pose.close()
            logging.getLogger("startup").info("MediaPipe Pose model pre-warmed")
        except Exception as e:
            logging.getLogger("startup").warning("MediaPipe pre-warm failed: %s", e)

    threading.Thread(target=_warmup_mediapipe, daemon=True).start()

    if detect_runtime() == "modal":
        sha = os.environ.get("STELLAR_GIT_SHA", "unknown")
        branch = os.environ.get("STELLAR_GIT_BRANCH", "unknown")
        bt = os.environ.get("STELLAR_BUILD_TIME", "unknown")
        _modal_echo(
            "[stellar-modal] startup: FastAPI + routers ready "
            f"| git_sha={sha} branch={branch} build_time={bt}"
        )


@app.on_event("shutdown")
async def _shutdown():
    """Drain module-level ThreadPoolExecutors to avoid background-thread warnings on exit."""
    import importlib

    for mod_name in ("routers.plus_analyze",):
        try:
            mod = importlib.import_module(mod_name)
            ex = getattr(mod, "_executor", None)
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

ALLOWED_ORIGIN_PATTERNS = [
    re.compile(r"^https://.*\.pages\.dev$"),
    re.compile(r"^https://.*\.vercel\.app$"),
    re.compile(r"^https://.*\.onrender\.com$"),
    re.compile(r"^https://.*\.modal\.run$"),
    re.compile(r"^http://localhost:\d+$"),
]


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "")
        # Accept any HTTPS origin — the API is protected by JWT, not CORS.
        allowed = bool(origin) and (
            origin == FRONTEND_URL
            or origin.startswith("https://")
            or any(p.match(origin) for p in ALLOWED_ORIGIN_PATTERNS)
        )

        if request.method == "OPTIONS":
            response = Response(status_code=204)
            if allowed:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Max-Age"] = "86400"
            return response

        response = await call_next(request)

        if allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response


app.add_middleware(DynamicCORSMiddleware)


class _ModalAccessLogMiddleware(BaseHTTPMiddleware):
    """One line per request on Modal only — shows up in `modal app logs stellar-ai`."""

    async def dispatch(self, request: Request, call_next):
        if detect_runtime() != "modal":
            return await call_next(request)
        path = request.url.path
        t0 = time.perf_counter()
        _modal_echo(f"[stellar-modal] {request.method} {path}")
        try:
            response = await call_next(request)
        except Exception as exc:
            dt = time.perf_counter() - t0
            _modal_echo(f"[stellar-modal] ERROR {path} {dt:.2f}s: {exc}")
            raise
        dt = time.perf_counter() - t0
        _modal_echo(f"[stellar-modal] {response.status_code} {path} {dt:.2f}s")
        return response


class _RuntimeHeaderMiddleware(BaseHTTPMiddleware):
    """Every response gets X-Stellar-Runtime + JSON /health.runtime — no curl required."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        rt = detect_runtime()
        response.headers["X-Stellar-Runtime"] = rt
        origin = request.headers.get("origin", "")
        allowed = bool(origin) and (
            origin == FRONTEND_URL
            or origin.startswith("https://")
            or any(p.match(origin) for p in ALLOWED_ORIGIN_PATTERNS)
        )
        if allowed and request.method != "OPTIONS":
            response.headers["Access-Control-Expose-Headers"] = "X-Stellar-Runtime"
        return response


app.add_middleware(_ModalAccessLogMiddleware)
app.add_middleware(_RuntimeHeaderMiddleware)

_load_errors: list[str] = []

# Modal workers set STELLAR_MODAL_PRO_V2_ONLY=1 before importing this module — skip legacy /stellar-pro.
_MODAL_PRO_V2_ONLY = (os.getenv("STELLAR_MODAL_PRO_V2_ONLY") or "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_EXPECTED_ROUTER_LOADS = 7 if _MODAL_PRO_V2_ONLY else 8


def _safe_load(module_path: str, prefix: str, tags: list[str]):
    """Import a router module; if it fails, log but keep the app alive."""
    try:
        mod = __import__(module_path, fromlist=["router"])
        app.include_router(mod.router, prefix=prefix, tags=tags)
    except Exception as e:
        msg = f"{module_path}: {e}"
        _load_errors.append(msg)
        print(f"[startup] WARNING — failed to load {msg}", file=sys.stderr)


_safe_load("routers.auth", "/auth", ["Authentication"])
with _suppress_native_stderr():
    _safe_load("routers.analyze", "/analyze", ["Analysis"])
    _safe_load("routers.plus_analyze", "/analyze", ["Plus Analysis"])
    _safe_load("routers.pose", "/pose", ["Pose Detection"])
_safe_load("routers.news", "", ["News"])
if not _MODAL_PRO_V2_ONLY:
    _safe_load("routers.stellar_pro_api", "", ["stellar-pro"])
_safe_load("routers.pro_v2_api", "", [])
_safe_load("routers.prov3_keyframes", "", ["prov3-keyframes"])


@app.get("/health")
async def health_check():
    from services.golfdb_swingnet_paths import resolve_swingnet_checkpoint_path
    from services.internal.prov3_ffmpeg import FFmpegNotFoundError, ffmpeg_has_filter, ffmpeg_bin

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    jwt_secret = os.getenv("JWT_SECRET", "")
    frontend_url = os.getenv("FRONTEND_URL", "")
    gemini_backend = (os.getenv("GEMINI_BACKEND") or "").strip().lower() or "developer_api"
    vertex_loc = (os.getenv("VERTEX_AI_LOCATION") or "").strip()
    proxy_on = bool(
        (os.getenv("GEMINI_HTTPS_PROXY") or "").strip()
        or (os.getenv("HTTPS_PROXY") or "").strip()
        or (os.getenv("https_proxy") or "").strip()
    )
    _sw_ck = resolve_swingnet_checkpoint_path()

    _ff_ok = False
    _mci = False
    try:
        ffmpeg_bin()
        _ff_ok = True
        _mci = bool(ffmpeg_has_filter("minterpolate"))
    except FFmpegNotFoundError:
        _ff_ok = False
        _mci = False

    payload = {
        "status": "healthy",
        "service": "stellar-ai",
        "runtime": detect_runtime(),
        # Log / client verification: Plus YOLO+biomech+chain degraded paths (200, not 422) — see STELLAR_PLUS_PIPELINE logs
        "plus_pipeline": {"code_marker": "plus_degradation_v1", "degraded_http_200": True},
        "routers_loaded": _EXPECTED_ROUTER_LOADS - len(_load_errors),
        "load_errors": _load_errors or None,
        "prov3": {
            "engine": "prov3",
            "route_naming": {
                "primary": "POST /pro-v3/analyze",
                "legacy_alias": "POST /pro-v2/analyze (same handler; media URLs stay under /pro-v2/media/ for those requests)",
                "env_STELLAR_MODAL_PRO_V2_ONLY": "If set, Modal skips registering /stellar-pro. Name is historical; not related to URL v2 alias.",
            },
            "pro_http_route": "POST /pro-v3/analyze",
            "pro_http_note": "Primary product path is /pro-v3; /pro-v2 remains a backward-compatible alias.",
            "pro_http": "POST /pro-v3/analyze -> Pro v3 keyframe pipeline (+ optional Gemini report)",
            "api": "POST /api/prov3/keyframes/analyze",
            "video": {
                "target_analysis_fps": 240,
                "pipeline": "cleanup -> analysis_240fps_timeline (minterpolate when available) -> frame_enhance",
                "ffmpeg_resolvable": _ff_ok,
                "minterpolate_available": _mci,
            },
            "ab_weights": {
                "a_engine": "wmcnally/golfdb:SwingNet (eight events + top-k)",
                "b_engine": "same SwingNet prob tensor - local peak refine (no second checkpoint)",
                "swingnet_weights_present": bool(_sw_ck),
                "swingnet_checkpoint": _sw_ck or None,
            },
            "swingnet_engine": "wmcnally/golfdb:SwingNet",
            "swingnet_weights_present": bool(_sw_ck),
            "swingnet_checkpoint": _sw_ck or None,
        },
        "env": {
            "GEMINI_API_KEY": "set" if gemini_key else "missing",
            "GEMINI_BACKEND": gemini_backend,
            "VERTEX_AI_LOCATION": vertex_loc or None,
            "GEMINI_HTTPS_PROXY": "set" if proxy_on else "unset",
            "JWT_SECRET": "set" if jwt_secret else "using-default",
            "FRONTEND_URL": frontend_url or "http://localhost:3000",
            "GOLF_NEWS_API_KEY": "set" if os.getenv("GOLF_NEWS_API_KEY") else "optional-not-set",
            **(
                {"STELLAR_MODAL_PRO_V2_ONLY": "1"}
                if _MODAL_PRO_V2_ONLY
                else {}
            ),
        },
    }
    if detect_runtime() == "modal":
        payload["modal_region"] = os.getenv("MODAL_REGION") or None
        # build_* = baked at `modal deploy` (image env / secret). server_time = this worker's clock now.
        payload["build"] = {
            "git_sha": os.getenv("STELLAR_GIT_SHA") or None,
            "branch": os.getenv("STELLAR_GIT_BRANCH") or None,
            "build_time": os.getenv("STELLAR_BUILD_TIME") or None,
            "server_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    return payload


@app.get("/")
async def root():
    return {"service": "stellar-ai", "status": "running"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
