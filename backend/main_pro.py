"""Same FastAPI app as `main.py`.

- Local/Render: may include `/stellar-pro/analyze` and `POST /pro-v3/analyze`.
- Modal: `STELLAR_RUNTIME=modal` + `STELLAR_MODAL_PRO_V3_ONLY=1` — no `/stellar-pro`, Pro HTTP is `/pro-v3` only.

Run locally:
  python main_pro.py
or:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

# Re-export app for ASGI servers that target `main_pro:app`.
from main import app  # noqa: F401


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
