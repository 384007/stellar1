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
from typing import Any, Callable, Dict, List, Optional

import cv2
from PIL import Image, ImageDraw

from lib.prov3.keyframes.constants import EVENT_SEQUENCE
from lib.prov3.keyframes.decode_spacing import spread_keyframes_for_preview_strip
from services.internal.prov3_ffmpeg import ffmpeg_extract_frames_bgr_by_decode_index, ffprobe_video_meta
from services.internal.frame_enhance_service import persist_final_keyframe_images
from services.pro_ui_bundle_service import build_stellar_pro_ui_bundle
from services.prov3_keyframe_orchestrator_service import run_keyframe_analyze_with_preprocess

logger = logging.getLogger(__name__)

# ── Prov3 **UI 条图**：旋转元数据、ffmpeg 按解码序号抽帧；post-A/B 不再改写 A/B 的 ``frame_index``。

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


def _remap_poses_for_analysis_timeline(
    poses: List[Dict[str, Any]],
    *,
    analysis_decode_span: int,
    analysis_duration_s: float,
) -> List[Dict[str, Any]]:
    """Map source-upload pose samples onto true-240 analysis decode indices for video scrubber alignment."""
    if not poses:
        return []
    n = max(1, int(analysis_decode_span))
    dur = float(analysis_duration_s or 0.0)
    if dur < 0.05:
        dur = max((float(p.get("timestamp") or 0.0) for p in poses), default=0.0)
        if dur < 0.05:
            dur = 1.0
    out: List[Dict[str, Any]] = []
    for p in poses:
        q = {k: v for k, v in p.items() if k != "image_base64"}
        ts = float(p.get("timestamp") or 0.0)
        u = min(1.0, max(0.0, ts / dur))
        q["frame_index"] = int(round(u * (n - 1)))
        out.append(q)
    return out


def _build_ui_keyframes(
    raw_keyframes: list[dict[str, Any]],
    analysis_video_path: str,
    analysis_fps: float,
) -> list[dict[str, Any]]:
    """Decode JPEG strips from the **true 240fps analysis** MP4 (``prov3.analysis_video``).

    Keyframes use **decode frame indices** in that file (same contract as SwingNet / ``generate_analysis_frames``).
    Time ``t ≈ frame_index / analysis_fps`` (240). **Primary path:** ffmpeg ``select=eq(n,…)`` + rawvideo.
    **Fallback:** linear remap into OpenCV when the container under-reports frame count or ffmpeg fails.
    """
    meta_pf: dict[str, Any] = {}
    try:
        meta_pf = ffprobe_video_meta(analysis_video_path)
    except Exception as exc:
        logger.warning("[PRO_PROV3] thumb ffprobe meta failed: %s", exc)

    dur_s = float(meta_pf.get("duration_s") or 0.0)
    nb_probe = int(meta_pf.get("nb_frames") or 0)
    w_pf = int(meta_pf.get("width") or 0)
    h_pf = int(meta_pf.get("height") or 0)
    fps_pf = float(meta_pf.get("fps") or 0.0)

    cap = cv2.VideoCapture(analysis_video_path)
    opened = cap.isOpened()
    if not opened:
        logger.warning("[PRO_PROV3] OpenCV cannot open video for thumbnails: %s", analysis_video_path)
    rotation = int(_prov3_thumb_container_rotation_degrees(analysis_video_path)) if opened else 0
    n_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    fps_cv = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
    w_cv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
    h_cv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0

    stream_fps = fps_pf if fps_pf > 1e-6 else (fps_cv if fps_cv > 1e-6 else 30.0)
    nb_eff = nb_probe
    if nb_eff <= 0 and dur_s > 0 and stream_fps > 0:
        nb_eff = max(1, int(round(dur_s * stream_fps)))
    n_decode_max = max(nb_eff, n_cv, 1)
    max_di = max(n_decode_max - 1, 0)
    w_meta = w_pf or w_cv
    h_meta = h_pf or h_cv

    fi_max = max((int(k.get("frame_index") or 0) for k in raw_keyframes), default=0)
    order = {name: i for i, name in enumerate(EVENT_SEQUENCE)}
    rows = sorted(raw_keyframes, key=lambda k: order.get(str(k.get("event_name")), 99))

    per_row: list[tuple[dict[str, Any], int]] = []
    for k in rows:
        fi = int(k.get("frame_index") or 0)
        # True-240 analysis: frame_index is already the demuxer frame number; clamp to probed range.
        di = max(0, min(fi, max_di))
        per_row.append((k, di))
    uniq_decode = sorted({di for _, di in per_row})

    bgr_map: dict[int, Any] = {}
    ffmpeg_ok = False
    if uniq_decode and w_meta > 0 and h_meta > 0:
        try:
            bgr_map = ffmpeg_extract_frames_bgr_by_decode_index(
                analysis_video_path,
                uniq_decode,
                width=w_meta,
                height=h_meta,
                timeout_s=300,
            )
            ffmpeg_ok = len(bgr_map) == len(uniq_decode)
            if not ffmpeg_ok:
                logger.warning(
                    "[PRO_PROV3] thumb ffmpeg extract incomplete keys=%s expected=%s",
                    len(bgr_map),
                    len(uniq_decode),
                )
                bgr_map = {}
        except Exception as exc:
            logger.warning("[PRO_PROV3] thumb ffmpeg extract failed, OpenCV fallback: %s", exc)
            bgr_map = {}

    out: list[dict[str, Any]] = []
    try:
        for k, decode_idx in per_row:
            ev = str(k.get("event_name") or "")
            phase = EVENT_NAME_TO_PHASE.get(ev, "address")
            fi = int(k.get("frame_index") or 0)
            time_s = float(fi) / max(float(analysis_fps), 1.0)
            frame_bgr = None
            src_idx = decode_idx
            if ffmpeg_ok and decode_idx in bgr_map:
                frame_bgr = _prov3_thumb_rotate_bgr(bgr_map[decode_idx], rotation)
            elif opened:
                src_idx = fi
                nframes = max(n_cv, 1)
                src_idx = max(0, min(src_idx, nframes - 1))
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
                    "frame_index": fi,
                    "confidence": round(conf, 4),
                    "analysis_fps": int(round(float(analysis_fps))),
                    "keyframe_image_source": "analysis_video",
                }
            )
    finally:
        try:
            cap.release()
        except Exception:
            pass
    n_ffmpeg = sum(1 for _, d in per_row if ffmpeg_ok and d in bgr_map)
    if per_row:
        logger.info("[PRO_PROV3] thumb strips: ffmpeg_frames=%s/%s", n_ffmpeg, len(per_row))
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


def _semantic_acceptance_gate(keyframes: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    by_phase = {str(k.get("phase") or ""): k for k in keyframes}
    fails: list[str] = []
    for phase in ("address", "top", "impact", "finish"):
        if phase not in by_phase:
            fails.append(f"{phase}_semantic_fail")
    if fails:
        return False, fails

    addr = int(by_phase["address"].get("frame_index") or 0)
    top = int(by_phase["top"].get("frame_index") or 0)
    mid_down = int(by_phase.get("downswing", {}).get("frame_index") or top)
    impact = int(by_phase["impact"].get("frame_index") or 0)
    finish = int(by_phase["finish"].get("frame_index") or 0)
    conf_top = float(by_phase["top"].get("confidence") or 0.0)
    conf_impact = float(by_phase["impact"].get("confidence") or 0.0)

    if not (addr < top < impact < finish):
        fails.append("semantic_event_order_fail")
    if top - addr < 6:
        fails.append("top_semantic_fail")
    if impact - top < 6:
        fails.append("impact_semantic_fail")
    if finish - impact < 8:
        fails.append("finish_semantic_fail")
    if not (top < mid_down < impact):
        fails.append("mid_downswing_semantic_fail")
    if conf_top < 0.58:
        fails.append("top_semantic_fail")
    if conf_impact < 0.58:
        fails.append("impact_semantic_fail")

    return (len(fails) == 0), sorted(set(fails))


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
    prov3, pre = run_keyframe_analyze_with_preprocess(
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
        raise RuntimeError("analysis_video_missing: true240 analysis video is required")

    # Official strip uses true A/B ``frame_index``; preview/contact sheet may use ``spread_keyframes_for_preview_strip`` only.

    persisted_kfs = persist_final_keyframe_images(av_path, raw_kfs, str(work / "keyframe_images"))
    event_to_phase = {v: k for k, v in EVENT_NAME_TO_PHASE.items()}
    persisted_by_phase = {
        event_to_phase.get(str(x["event_name"]), "").strip(): x
        for x in persisted_kfs
        if event_to_phase.get(str(x["event_name"]), "").strip()
    }

    fi_max = max((int(k.get("frame_index") or 0) for k in raw_kfs), default=0)
    dur_pf = 0.0
    try:
        _m = ffprobe_video_meta(av_path)
        nb_pf = int(_m.get("nb_frames") or 0)
        dur_pf = float(_m.get("duration_s") or 0.0)
        fps_pf = float(_m.get("fps") or 240.0)
        if nb_pf <= 0 and dur_pf > 0 and fps_pf > 1e-6:
            nb_pf = max(1, int(round(dur_pf * fps_pf)))
        span_timeline = max(nb_pf, fi_max + 1, 1)
    except Exception:
        span_timeline = max(fi_max + 1, 1)

    ui_official = _build_ui_keyframes(raw_kfs, av_path, float(prov3.analysis_fps or ANALYSIS_FPS))
    preview_raw = spread_keyframes_for_preview_strip(raw_kfs, span_timeline)
    ui_preview = _build_ui_keyframes(preview_raw, av_path, float(prov3.analysis_fps or ANALYSIS_FPS))

    for row in ui_official:
        phase = str(row.get("phase") or "")
        saved = persisted_by_phase.get(phase)
        if saved:
            row["keyframe_image_path"] = str(saved["file_path"])
            row["keyframe_image_file"] = str(saved["file_name"])
            row["keyframe_image_source"] = "analysis_video"
    avg_c = _avg_confidence(ui_official)
    total_score = round(min(100.0, max(0.0, avg_c * 100.0)), 1)

    trust = str(dumped.get("trust_level") or "low")
    status = str(dumped.get("status") or "low_trust")
    fail_reasons = list(dumped.get("fail_reasons") or [])

    # Semantic gate: validation only — never reorder or rewrite keyframe rows or indices.
    semantic_ok, semantic_fails = _semantic_acceptance_gate(ui_official)
    logger.info(
        "[PRO_PROV3][semantic_gate] ok=%s fails=%s",
        int(semantic_ok),
        semantic_fails,
    )
    if not semantic_ok:
        status = "low_trust"
        trust = "low"
        fail_reasons = sorted(set([*fail_reasons, *semantic_fails]))

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

    sheet_path: str | None = None
    try:
        p = work / "prov3_contact_sheet.jpg"
        sheet_path = _build_contact_sheet(ui_preview, p)
    except Exception as exc:
        logger.warning("[PRO_PROV3] contact_sheet skipped: %s", exc)

    low_trust_preview_only = status != "pass" or analysis_trust == "low_trust"
    if low_trust_preview_only and "low_trust_preview_only" not in fail_reasons:
        fail_reasons = [*fail_reasons, "low_trust_preview_only"]
    preview_keyframes = list(ui_preview)
    official_phase_keyframes = [] if low_trust_preview_only else list(ui_official)
    # Top-level ``keyframes`` = official strip only (never alias preview when low trust).
    product_keyframes = list(official_phase_keyframes)
    issues = fail_reasons[:3] if fail_reasons else []
    issues_zh = fail_reasons[:3] if fail_reasons else []

    phase_keyframes: dict[str, int] = {}
    for k in official_phase_keyframes:
        ph = str(k.get("phase") or "")
        spi = k.get("frame_index")
        if ph and isinstance(spi, int):
            phase_keyframes[ph] = spi

    dur_analysis = float(dur_pf or 0.0)
    if dur_analysis < 0.05:
        try:
            _m_dur = ffprobe_video_meta(av_path)
            dur_analysis = float(_m_dur.get("duration_s") or 0.0)
        except Exception:
            dur_analysis = 0.0
    poses_src = list(pre.poses or [])
    poses_remapped = _remap_poses_for_analysis_timeline(
        poses_src,
        analysis_decode_span=int(span_timeline),
        analysis_duration_s=dur_analysis,
    )
    ui_bundle: dict[str, Any] = {}
    if poses_remapped:
        try:
            ui_bundle = build_stellar_pro_ui_bundle(
                poses_remapped,
                phase_keyframes,
                fps=float(prov3.analysis_fps or ANALYSIS_FPS),
                source_frame_count=int(span_timeline),
                analysis_frame_count=int(span_timeline),
                detected_club=None,
            )
        except Exception as exc:
            logger.exception("[PRO_PROV3] build_stellar_pro_ui_bundle failed: %s", exc)
            ui_bundle = {}
    logger.info(
        "[PRO_PROV3] pose_chain preprocess_poses=%d remapped=%d bundle_pose_frames=%d",
        len(poses_src),
        len(poses_remapped),
        len((ui_bundle.get("pose_frames") or []) if ui_bundle else []),
    )

    skel = ui_bundle.get("skeleton_data") if ui_bundle else None
    pframes = ui_bundle.get("pose_frames") if ui_bundle else None
    pred = ui_bundle.get("prediction") if ui_bundle else None

    minimal: dict[str, Any] = {
        "analysis_id": str(dumped.get("analysis_id") or ""),
        "status": "completed",
        "final_status": status,
        "trust_level": analysis_trust,
        "pipeline": "prov3",
        "keyframes_strip": {
            "timeline": "analysis_240",
            "analysis_fps": int(prov3.analysis_fps or ANALYSIS_FPS),
            "thumbnails_from_analysis_video": True,
        },
        "summary": summary,
        "summary_zh": summary_zh,
        "total_score": total_score,
        "keyframes": product_keyframes,
        "official_phase_keyframes": official_phase_keyframes,
        "preview_keyframes": preview_keyframes,
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
        "low_trust_preview_only": low_trust_preview_only,
        "keyframe_mismatch_notice": status != "pass",
        "warning": "" if status == "pass" else "关键帧可信度有限，结论仅供参考。",
        "screen_keyframe_review_applied": False,
        "screen_mode_user_requested": bool(screen_mode),
        "skeleton_data": skel
        if isinstance(skel, dict)
        else {"frames": [], "total_frames": 0},
        "pose_frames": pframes if isinstance(pframes, list) else [],
        **({"prediction": pred} if isinstance(pred, dict) else {}),
        "phase_keyframes": phase_keyframes,
        "keyframe_images": persisted_kfs,
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

    uvm = ui_bundle.get("video_meta") if ui_bundle else None
    if isinstance(uvm, dict) and uvm.get("total_pose_frames", 0):
        minimal["video_meta"] = {
            "fps": float(prov3.analysis_fps or ANALYSIS_FPS),
            "duration_s": float(uvm.get("duration_s") or dur_analysis or 0.0),
            "source_frame_count": int(uvm.get("source_frame_count") or span_timeline),
            "total_pose_frames": int(uvm.get("total_pose_frames") or 0),
        }
    else:
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
        len(ui_official),
        analysis_trust,
    )
    return minimal
