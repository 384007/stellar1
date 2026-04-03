"""
JSON-safe payloads for FastAPI: NaN / inf break stdlib json.dumps (and clients).

Use sanitize_json_floats() before returning large analysis dicts.
Use safe_float() when parsing external / pose numeric fields.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Parse a scalar to float; replace None / NaN / ±inf / bad cast with ``default``.
    Handles numpy.float32 / float64 and numpy integers (converted, then float).
    """
    if x is None:
        return default
    try:
        if np is not None:
            if isinstance(x, np.floating):
                v = float(x)
            elif isinstance(x, np.integer):
                v = float(int(x))
            else:
                v = float(x)
        else:
            v = float(x)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(v):
        return default
    return v


def find_non_finite_paths(obj: Any, prefix: str = "") -> list[str]:
    """
    Return dotted paths to non-finite floats (for debug logging).
    Examples: ``prediction.distance_debug.inputs.peak_x_factor``, ``pose_frames[12].angles.spine_tilt``.
    """
    out: list[str] = []

    def _path(key_suffix: str) -> str:
        if not prefix:
            return key_suffix
        if key_suffix.startswith("["):
            return f"{prefix}{key_suffix}"
        return f"{prefix}.{key_suffix}" if prefix else key_suffix

    if np is not None and isinstance(obj, np.ndarray):
        return find_non_finite_paths(obj.tolist(), prefix)

    if isinstance(obj, bool):
        return out

    if np is not None and isinstance(obj, np.floating):
        v = float(obj)
        if not math.isfinite(v):
            out.append(prefix or "<root>")
        return out

    if isinstance(obj, (float, int)):
        if isinstance(obj, float):
            if not math.isfinite(obj):
                out.append(prefix or "<root>")
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = _path(str(k))
            out.extend(find_non_finite_paths(v, key))
        return out

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = _path(f"[{i}]")
            out.extend(find_non_finite_paths(v, key))
        return out

    return out


def sanitize_json_floats(obj: Any) -> Any:
    """
    Recursively copy structures; replace non-finite floats with None.
    numpy scalars → Python float or None; numpy ndarray → nested lists (sanitized).
    bytes / bytearray left unchanged.
    """
    if obj is None:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj

    if np is not None:
        if isinstance(obj, np.ndarray):
            return sanitize_json_floats(obj.tolist())
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return v if math.isfinite(v) else None

    if isinstance(obj, int):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, dict):
        return {k: sanitize_json_floats(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [sanitize_json_floats(v) for v in obj]

    return obj


def log_non_finite_if_any(log: logging.Logger, obj: Any, tag: str, max_paths: int = 80) -> None:
    paths = find_non_finite_paths(obj)
    if paths:
        joined = "; ".join(paths[:max_paths])
        more = f" …(+{len(paths) - max_paths} more)" if len(paths) > max_paths else ""
        log.warning("[%s] non-finite floats before JSON sanitize: %s%s", tag, joined, more)
