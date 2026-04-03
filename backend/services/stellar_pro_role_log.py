from __future__ import annotations

from typing import Iterable

from services.provider_registry import role_log


ROLE_STELLAR_PRO_API = "STELLAR_PRO_API"
ROLE_FFMPEG_PREP = "FFMPEG_PREP"
ROLE_PRO_CHAIN = "PRO_CHAIN"
ROLE_POSE_BACKEND = "POSE_BACKEND"
ROLE_YOLO11 = "YOLO11"
ROLE_BYTETRACK = "BYTETRACK"
ROLE_MMACTION2 = "MMACTION2"
ROLE_PHASE_WINDOW = "PHASE_WINDOW"
ROLE_TOP_REFINE = "TOP_REFINE"
ROLE_OPENCV_IMPACT = "OPENCV_IMPACT"
ROLE_KEYFRAME_SOLVER = "KEYFRAME_SOLVER"
ROLE_CONTACT_SHEET = "CONTACT_SHEET"
ROLE_GEMINI_REPORT = "GEMINI_REPORT"
ROLE_PRO_PUBLIC_PACK = "PRO_PUBLIC_PACK"
ROLE_STELLAR_PRO_API_DONE = "STELLAR_PRO_API_DONE"


STELLAR_PRO_ROLE_ORDER: list[str] = [
    ROLE_STELLAR_PRO_API,
    ROLE_FFMPEG_PREP,
    ROLE_PRO_CHAIN,
    ROLE_POSE_BACKEND,
    ROLE_YOLO11,
    ROLE_BYTETRACK,
    ROLE_MMACTION2,
    ROLE_PHASE_WINDOW,
    ROLE_TOP_REFINE,
    ROLE_OPENCV_IMPACT,
    ROLE_KEYFRAME_SOLVER,
    ROLE_CONTACT_SHEET,
    ROLE_GEMINI_REPORT,
    ROLE_PRO_PUBLIC_PACK,
    ROLE_STELLAR_PRO_API_DONE,
]


def pro_role_log(role_name: str, message: str) -> None:
    role_log(f"[ROLE={role_name}] {message}")


def log_stage_start(role_name: str, **fields) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    pro_role_log(role_name, ("start " + payload).strip())


def log_stage_done(role_name: str, **fields) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    pro_role_log(role_name, ("done " + payload).strip())


def log_stage_failed(role_name: str, **fields) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    pro_role_log(role_name, ("failed " + payload).strip())


def validate_role_sequence(sequence: Iterable[str]) -> bool:
    seen = list(sequence)
    expected_prefix = STELLAR_PRO_ROLE_ORDER[: len(seen)]
    return seen == expected_prefix
