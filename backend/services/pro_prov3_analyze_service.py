"""Pro HTTP product response from the Pro v3 keyframe pipeline + optional Gemini report layer.

Keyframes and thumbnails are Pro v3 only. Long-form summary / suggestions / training_plan are filled
asynchronously in ``enrich_pro_prov3_response`` (see ``pro_prov3_gemini_enrich``) using the same
motion_context contract as Pro v2 text-only reports.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw

from lib.prov3.keyframes.constants import EVENT_SEQUENCE
from services.prov3_keyframe_orchestrator_service import run_keyframe_analyze

logger = logging.getLogger(__name__)

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
    cap = cv2.VideoCapture(video_path)
    opened = cap.isOpened()
    if not opened:
        cap.release()
        logger.warning("[PRO_PROV3] cannot open video for thumbnails: %s", video_path)
        fps, nframes = 30.0, 1
    else:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        nframes = max(nframes, 1)

    order = {name: i for i, name in enumerate(EVENT_SEQUENCE)}
    rows = sorted(raw_keyframes, key=lambda k: order.get(str(k.get("event_name")), 99))

    out: list[dict[str, Any]] = []
    for k in rows:
        ev = str(k.get("event_name") or "")
        phase = EVENT_NAME_TO_PHASE.get(ev, "address")
        fi = int(k.get("frame_index") or 0)
        time_s = float(fi) / ANALYSIS_FPS
        src_idx = int(round(time_s * fps))
        src_idx = max(0, min(src_idx, nframes - 1))
        frame_bgr = None
        if opened:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(src_idx))
            ok, frame_bgr = cap.read()
            if not ok:
                frame_bgr = None
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
) -> dict[str, Any]:
    """Synchronous end-to-end Pro response for `/pro-v2/analyze` using only Pro v3 services."""
    _ = rough_impact_time_s
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[PRO_PROV3] start path=%s screen_mode=%s",
        input_video_path,
        "true" if screen_mode else "false",
    )

    prov3 = run_keyframe_analyze(input_video_path, str(work), screen_mode=screen_mode)
    dumped = prov3.model_dump(exclude={"analysis_video", "analysis_fps", "source_fps"})
    raw_kfs = list(dumped.get("keyframes") or [])

    ui_keyframes = _build_ui_keyframes(raw_kfs, input_video_path)
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
        "pipeline": "prov3",
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
        "pro_v2_screen_pipeline": False,
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
