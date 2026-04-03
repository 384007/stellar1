"""Same FastAPI app as `main.py` (local/Render: `/stellar-pro/analyze` + `/pro-v2/analyze`).

Modal workers set `STELLAR_MODAL_PRO_V2_ONLY=1` before import — only `/pro-v2/analyze` is registered.

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
