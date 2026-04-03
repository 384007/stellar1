from __future__ import annotations

import os
from pathlib import Path

from services.provider_registry import role_log
from services.provider_schema import provider_result


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deeplabcut_workspace_bases() -> list[Path]:
    """Local repo workspace first; Modal bake uses /opt/deeplabcut_workspace."""
    return [
        _backend_root() / "deeplabcut_workspace",
        Path("/opt/deeplabcut_workspace"),
    ]


def _deeplabcut_workspace() -> Path:
    for base in _deeplabcut_workspace_bases():
        if (base / ".stellar_dlc_config").is_file():
            return base
    return _backend_root() / "deeplabcut_workspace"


def _default_project_config_from_marker() -> str:
    """Path from scripts/bootstrap_deeplabcut_workspace.py or Modal image bake."""
    for base in _deeplabcut_workspace_bases():
        marker = base / ".stellar_dlc_config"
        if not marker.is_file():
            continue
        line = (marker.read_text(encoding="utf-8").strip().splitlines() or [""])[0].strip()
        if line and Path(line).is_file():
            return line
    return ""


def _resolved_project_config() -> str:
    return (os.getenv("STELLAR_DLC_PROJECT_CONFIG") or "").strip() or _default_project_config_from_marker()


def _resolved_output_dir(video_path: str) -> str:
    env = (os.getenv("STELLAR_DLC_OUTPUT_DIR") or "").strip()
    if env:
        return env
    out = _deeplabcut_workspace() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return str(out.resolve())


def run(video_path: str, frame_count: int) -> dict:
    try:
        import deeplabcut
    except Exception:
        role_log(f"[ROLE=DEEPLABCUT] status=dependency_missing frames={frame_count}")
        return provider_result(
            role="research",
            provider_name="deeplabcut",
            status="dependency_missing",
            frame_count=frame_count,
            payload={},
            error_reason="dependency_missing",
        )
    project_config = _resolved_project_config()
    if not project_config:
        role_log(f"[ROLE=DEEPLABCUT] status=model_config_missing frames={frame_count}")
        return provider_result(
            role="research",
            provider_name="deeplabcut",
            status="model_config_missing",
            frame_count=frame_count,
            payload={},
            error_reason="model_config_missing",
        )
    try:
        out_dir = _resolved_output_dir(video_path)
        deeplabcut.analyze_videos(project_config, [video_path], destfolder=out_dir, save_as_csv=True)
    except Exception as exc:
        role_log(f"[ROLE=DEEPLABCUT] status=inference_failed frames={frame_count} err={exc}")
        return provider_result(
            role="research",
            provider_name="deeplabcut",
            status="inference_failed",
            frame_count=frame_count,
            payload={},
            error_reason=str(exc),
        )
    role_log(f"[ROLE=DEEPLABCUT] status=ok frames={frame_count}")
    return provider_result(
        role="research",
        provider_name="deeplabcut",
        status="ok",
        frame_count=frame_count,
        payload={"refined_keypoints": [], "output_dir": out_dir},
    )
