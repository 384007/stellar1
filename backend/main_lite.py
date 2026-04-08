import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.responses import Response

load_dotenv()

app = FastAPI(
    title="Stellar AI Lite API",
    version="1.0.0",
    description="Lite-only backend surface. Internal analysis pipeline is fully encapsulated server-side.",
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)


def _safe_load(module_path: str, prefix: str, tags: list[str]):
    mod = __import__(module_path, fromlist=["router"])
    app.include_router(mod.router, prefix=prefix, tags=tags)


_safe_load("routers.auth", "/auth", ["Authentication"])
_safe_load("routers.analyze", "/analyze", ["Analysis"])


@app.get("/")
async def root():
    # 浏览器只打开 base URL 时默认无路由会 404；重定向到 /health 便于自检。
    return RedirectResponse(url="/health", status_code=307)


@app.head("/health")
async def health_check_head():
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "stellar-ai-lite",
        "routes": ["/auth/*", "/analyze/lite", "/analyze/recalculate"],
    }
