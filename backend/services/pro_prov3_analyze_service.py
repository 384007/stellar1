"""Pro HTTP product response from the Pro v3 keyframe pipeline + optional Gemini report layer.

Keyframes and thumbnails are Pro v3 only. Long-form summary / suggestions / training_plan are filled
asynchronously in ``enrich_pro_prov3_response`` (see ``pro_prov3_gemini_enrich``) using the same
motion_context contract for text-only Gemini reports.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
from PIL import Image, ImageDraw

from lib.prov3.keyframes.constants import EVENT_SEQUENCE
from services.prov3_keyframe_orchestrator_service import run_keyframe_analyze

logger = logging.getLogger(__name__)

# ── Prov3 **UI 条图专用**：容器旋转元数据 → 显示方向。不 import 其他业务链路；
#    A/B / SwingNet / pose 分析路径完全不动，仅 `_build_ui_keyframes` 使用下列函数。

_PROV3_THUMB_FFPROBE_OK: Optional[bool] = None
_PROV3_THUMB_FFPROBE_WARNED = False


def _prov3_thumb_parse_rotate_tag(val: Any) -> int:
    if val is None:
        return 0
    try:
        return int(float(val)) % 360
    except (TypeError, ValueError):
        return 0


def _prov3_thumb_ffprobe_available() -> bool:
    global _PROV3_THUMB_FFPROBE_OK, _PROV3_THUMB_FFPROBE_WARNED
    if _PROV3_THUMB_FFPROBE_OK is None:
        _PROV3_THUMB_FFPROBE_OK = shutil.which("ffprobe") is not None
    if not _PROV3_THUMB_FFPROBE_OK and not _PROV3_THUMB_FFPROBE_WARNED:
        logger.warning("[PRO_PROV3][thumb] ffprobe missing — rotation metadata may be skipped")
        _PROV3_THUMB_FFPROBE_WARNED = True
    return bool(_PROV3_THUMB_FFPROBE_OK)


def _prov3_thumb_container_rotation_degrees(video_path: str) -> int:
    """Return 0, 90, 180, or 270 for **display** correction (phone portrait MP4)."""
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            prop = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            cap.release()
            if prop in (90, 180, 270):
                return int(prop)
    except Exception:
        pass

    if not _prov3_thumb_ffprobe_available():
        return 0

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                "-select_streams",
                "v:0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return 0
        info = json.loads(result.stdout)
        for stream in info.get("streams", []):
            tags = stream.get("tags", {})
            rot = _prov3_thumb_parse_rotate_tag(tags.get("rotate"))
            if rot in (90, 180, 270):
                return rot
            for sd in stream.get("side_data_list", []):
                if sd.get("side_data_type") == "Display Matrix":
                    rot_val = sd.get("rotation", 0)
                    try:
                        r = (-int(rot_val)) % 360
                    except (TypeError, ValueError):
                        r = 0
                    if r in (90, 180, 270):
                        return r
        fmt_tags = info.get("format", {}).get("tags", {})
        rot = _prov3_thumb_parse_rotate_tag(fmt_tags.get("rotate"))
        if rot in (90, 180, 270):
            return rot
    except (FileNotFoundError, json.JSONDecodeError, ValueError, subprocess.TimeoutExpired) as exc:
        logger.warning("[PRO_PROV3][thumb] ffprobe rotation parse failed: %s", exc)
    return 0


def _prov3_thumb_rotate_bgr(frame: Any, rotation: int) -> Any:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _prov3_thumb_read_frame_bgr(cap: cv2.VideoCapture, frame_idx: int, rotation: int) -> Any | None:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return None
    target = int(min(max(int(frame_idx), 0), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return None
    return _prov3_thumb_rotate_bgr(frame_bgr, rotation)

ANALYSIS_FPS = 240.0

EVENT_NAME_TO_PHASE: dict[str, str] = {
    "Address": "address",
    "Toe-up": "takeaway",
    "Mid-backswing": "backswing",
    "Top": "top",
    "Mid-downswing": "downswing",
    "Impact": "impact",
    "Mid-follow-through": "follow_through",
    "Finish": "finish",
}

EVENT_LABELS_EN: dict[str, str] = {
    "Address": "Address",
    "Toe-up": "Toe-up",
    "Mid-backswing": "Mid-backswing",
    "Top": "Top",
    "Mid-downswing": "Mid-downswing",
    "Impact": "Impact",
    "Mid-follow-through": "Mid-follow-through",
    "Finish": "Finish",
}

EVENT_LABELS_ZH: dict[str, str] = {
    "Address": "站姿",
    "Toe-up": "杆头翘起",
    "Mid-backswing": "上杆中段",
    "Top": "顶点",
    "Mid-downswing": "下杆中段",
    "Impact": "触球",
    "Mid-follow-through": "送杆中段",
    "Finish": "收杆",
}


def _jpeg_b64_bgr(frame_bgr: Any, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _probe_video(path: str) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot_open_video:{path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = float(n / fps) if fps > 0 else 0.0
    cap.release()
    return fps, max(n, 1), dur


def _build_ui_keyframes(raw_keyframes: list[dict[str, Any]], video_path: str) -> list[dict[str, Any]]:
    """Decode JPEG strips from the **analysis 240 Hz** MP4 (``prov3.analysis_video``).

    SwingNet ``frame_index`` is a **native frame index** into that file. We seek by index directly;
    OpenCV ``CAP_PROP_FPS`` on minterpolate/H.264 output is often wrong, so ``time_s * fps`` remapping
    would desync UI strips from true-240 inference.

    Applies **container display rotation** for UI thumbnails only.
    """
    cap = cv2.VideoCapture(video_path)
    opened = cap.isOpened()
    rotation = int(_prov3_thumb_container_rotation_degrees(video_path)) if opened else 0
    if not opened:
        cap.release()
        logger.warning("[PRO_PROV3] cannot open video for thumbnails: %s", video_path)
        nframes = 1
    else:
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        nframes = max(nframes, 1)
        fps_cv = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps_cv > 1.0 and abs(fps_cv - ANALYSIS_FPS) > 5.0:
            logger.warning(
                "[PRO_PROV3] analysis clip OpenCV fps=%.2f (expect ~%.0f) — using native frame_index seeks",
                fps_cv,
                ANALYSIS_FPS,
            )

    order = {name: i for i, name in enumerate(EVENT_SEQUENCE)}
    rows = sorted(raw_keyframes, key=lambda k: order.get(str(k.get("event_name")), 99))

    out: list[dict[str, Any]] = []
    for k in rows:
        ev = str(k.get("event_name") or "")
        phase = EVENT_NAME_TO_PHASE.get(ev, "address")
        fi = int(k.get("frame_index") or 0)
        time_s = float(fi) / ANALYSIS_FPS
        src_idx = max(0, min(fi, nframes - 1))
        frame_bgr = None
        if opened:
            frame_bgr = _prov3_thumb_read_frame_bgr(cap, src_idx, rotation)
        b64 = _jpeg_b64_bgr(frame_bgr) if frame_bgr is not None else ""
        conf = float(k.get("confidence") or 0.0)
        out.append(
            {
                "phase": phase,
                "label_en": EVENT_LABELS_EN.get(ev, ev or phase),
                "label_zh": EVENT_LABELS_ZH.get(ev, phase),
                "timestamp": round(time_s, 4),
                "image_base64": b64,
                "source_frame_index": src_idx,
                "source_pose_idx": src_idx,
                "confidence": round(conf, 4),
                "prov3_event_name": ev,
                "analysis_fps": int(ANALYSIS_FPS),
                "keyframe_source": "analysis_240",
            }
        )
    if opened:
        cap.release()
    return out


def _build_contact_sheet(keyframes: list[dict[str, Any]], output_path: Path) -> str | None:
    rows: list[tuple[Image.Image, str, float]] = []
    for kf in keyframes:
        raw = str(kf.get("image_base64") or "")
        if not raw:
            continue
        try:
            data = base64.b64decode(raw)
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        label = str(kf.get("label_zh") or kf.get("label_en") or kf.get("phase") or "")
        rows.append((img, label, float(kf.get("timestamp") or 0.0)))
    if not rows:
        return None

    columns = 4
    tile_w = 320
    label_h = 34
    th = int(rows[0][0].height * tile_w / rows[0][0].width)
    total_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, (th + label_h) * total_rows), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for idx, (img, label, ts) in enumerate(rows):
        r, c = divmod(idx, columns)
        x, y = c * tile_w, r * (th + label_h)
        thumb = img.resize((tile_w, th))
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y + th, x + tile_w, y + th + label_h), fill=(0, 0, 0))
        draw.text((x + 8, y + th + 8), f"{label}  {ts:.2f}s", fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output_path), quality=92)
    return str(output_path)


def _avg_confidence(keyframes: list[dict[str, Any]]) -> float:
    vals = [float(k.get("confidence") or 0.0) for k in keyframes]
    return sum(vals) / len(vals) if vals else 0.0


def run_pro_video_analyze_via_prov3(
    input_video_path: str,
    work_dir: str,
    *,
    screen_mode: bool = False,
    rough_impact_time_s: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Synchronous end-to-end Pro response for ``POST /pro-v3/analyze`` (see ``routers.prov3_api``)."""
    _ = rough_impact_time_s
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[PRO_PROV3] start path=%s screen_mode=%s",
        input_video_path,
        "true" if screen_mode else "false",
    )

    if cancel_check:
        cancel_check()
    prov3 = run_keyframe_analyze(
        input_video_path,
        str(work),
        screen_mode=screen_mode,
        cancel_check=cancel_check,
    )
    dumped = prov3.model_dump(exclude={"analysis_video", "analysis_fps", "source_fps"})
    raw_kfs = list(dumped.get("keyframes") or [])

    if cancel_check:
        cancel_check()
    av_path = str(prov3.analysis_video or "").strip()
    thumb_ok = bool(av_path and Path(av_path).is_file())
    if not thumb_ok:
        logger.error(
            "[PRO_PROV3] analysis_video missing or not on disk — UI strips cannot be aligned to true-240 analysis",
        )
    # Prefer analysis_240fps.mp4 (same file as A/B); last resort cleanup path (indices may mismatch).
    thumb_src = av_path if thumb_ok else input_video_path
    ui_keyframes = _build_ui_keyframes(raw_kfs, thumb_src)
    avg_c = _avg_confidence(ui_keyframes)
    total_score = round(min(100.0, max(0.0, avg_c * 100.0)), 1)

    trust = str(dumped.get("trust_level") or "low")
    status = str(dumped.get("status") or "low_trust")
    fail_reasons = list(dumped.get("fail_reasons") or [])

    analysis_trust = {
        "high": "high_trust",
        "medium": "medium_trust",
        "low": "low_trust",
    }.get(trust, "low_trust")
    report_mode = "formal" if status == "pass" else "limited"

    if status == "pass":
        summary = (
            "Pro v3 keyframe pipeline completed. Eight swing phases are indexed for review "
            f"(trust={trust})."
        )
        summary_zh = (
            "Pro v3 关键帧链路已完成，已标注八个挥杆相位供查看"
            f"（可信度：{trust}）。"
        )
    else:
        summary = (
            "Pro v3 keyframe pipeline finished with limited trust; "
            f"reasons: {', '.join(fail_reasons) or 'unspecified'}."
        )
        summary_zh = (
            "Pro v3 关键帧链路已完成，但可信度有限；原因："
            f"{ '、'.join(fail_reasons) or '未指定' }。"
        )

    issues = fail_reasons[:3] if fail_reasons else []
    issues_zh = fail_reasons[:3] if fail_reasons else []

    sheet_path: str | None = None
    try:
        p = work / "prov3_contact_sheet.jpg"
        sheet_path = _build_contact_sheet(ui_keyframes, p)
    except Exception as exc:
        logger.warning("[PRO_PROV3] contact_sheet skipped: %s", exc)

    phase_keyframes: dict[str, int] = {}
    for k in ui_keyframes:
        ph = str(k.get("phase") or "")
        spi = k.get("source_frame_index")
        if ph and isinstance(spi, int):
            phase_keyframes[ph] = spi

    minimal: dict[str, Any] = {
        "analysis_id": str(dumped.get("analysis_id") or ""),
        "status": "completed",
        "final_status": status,
        "trust_level": analysis_trust,
        "pipeline": "prov3",
        "keyframes_strip": {
            "timeline": "analysis_240" if thumb_ok else "fallback_source",
            "analysis_fps": int(prov3.analysis_fps or ANALYSIS_FPS),
            "thumbnails_from_analysis_video": thumb_ok,
        },
        "summary": summary,
        "summary_zh": summary_zh,
        "total_score": total_score,
        "keyframes": ui_keyframes,
        "contact_sheet_url": sheet_path or "",
        "video_url": input_video_path,
        "original_video_url": input_video_path,
        "playback_video_url": input_video_path,
        "issues": issues,
        "issues_zh": issues_zh,
        "suggestions": [],
        "suggestions_zh": [],
        "scores": {"overall": total_score},
        "type": "pro",
        "training_plan": {
            "day_1": {
                "focus": "节奏与击球瞬间",
                "focus_en": "Tempo and impact",
                "drills": ["按报告中的关键帧对照自检挥杆顺序"],
            }
        },
        "video_meta": {},
        "prov3_screen_pipeline": bool(screen_mode),
        "screen_mode": bool(screen_mode),
        "analysis_trust": analysis_trust,
        "report_mode": report_mode,
        "review_round": 0,
        "core_frame_scores": {},
        "retry_required": False,
        "retry_reasons": fail_reasons,
        "keyframe_mismatch_notice": status != "pass",
        "warning": "" if status == "pass" else "关键帧可信度有限，结论仅供参考。",
        "screen_keyframe_review_applied": False,
        "screen_mode_user_requested": bool(screen_mode),
        "skeleton_data": {"frames": [], "total_frames": 0},
        "pose_frames": [],
        "phase_keyframes": phase_keyframes,
        "prov3": {
            "status": status,
            "trust_level": trust,
            "fail_reasons": fail_reasons,
            "analysis_fps": int(prov3.analysis_fps or ANALYSIS_FPS),
        },
        "_prov3_motion": {
            "analysis_video": str(prov3.analysis_video or ""),
            "analysis_fps": float(prov3.analysis_fps or ANALYSIS_FPS),
            "source_fps": float(prov3.source_fps or 30.0),
            "screen_mode": bool(screen_mode),
        },
    }

    try:
        fps, nframes, dur = _probe_video(input_video_path)
        minimal["video_meta"] = {
            "fps": fps,
            "duration_s": dur,
            "source_frame_count": nframes,
        }
    except RuntimeError as exc:
        logger.warning("[PRO_PROV3] video_meta probe failed: %s", exc)

    logger.info(
        "[PRO_PROV3] done analysis_id=%s kfs=%s trust=%s",
        minimal["analysis_id"],
        len(ui_keyframes),
        analysis_trust,
    )
    return minimal
