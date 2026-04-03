from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def role_log(msg: str) -> None:
    logger.info(msg)
    print(msg, flush=True)
    print(msg, flush=True, file=sys.stderr)


def get_backend(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip().lower()
