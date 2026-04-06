"""Pro v3 关键帧之后的 Gemini 文案报告（motion_context + 多 pass + 本地兜底）。

Motion 报告与 ``prov3_text_report_service`` / Gemini prompt 对齐；对外统一 ``prov3`` 命名与日志标签。
若浏览器先于服务端结束而断开，Modal 上仍可能打满日志但前端看不到报告——请对齐客户端超时与 Modal task 超时。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2

from services.prov3_text_report_service import (
    _synthetic_keyframe_evaluations,
    pop_prov3_report_meta,
    write_prov3_ai_report,
)
from services.prov3_report_motion import DenseFrame, dense_scan_swing_region, find_swing_window_seconds

logger = logging.getLogger(__name__)


def _merge_keyframe_ai_into_strips(minimal: dict[str, Any], evals: list[Any]) -> None:
    """Attach Gemini per-phase scores/text onto keyframe row dicts (by phase)."""
    if not isinstance(evals, list) or not evals:
        return
    by_phase: dict[str, dict[str, Any]] = {}
    for e in evals:
        if not isinstance(e, dict):
            continue
        ph = str(e.get("phase") or "").strip().lower()
        if ph:
            by_phase[ph] = e

    def _apply(rows: list[Any]) -> None:
        if not isinstance(rows, list):
            return
        for k in rows:
            if not isinstance(k, dict):
                continue
            ph = str(k.get("phase") or "").strip().lower()
            ev = by_phase.get(ph)
            if not ev:
                continue
            try:
                sc = ev.get("score")
                if sc is not None:
                    k["ai_phase_score"] = float(sc)
            except (TypeError, ValueError):
                pass
            en = ev.get("action_assessment_en")
            zh = ev.get("action_assessment_zh")
            if isinstance(en, str) and en.strip():
                k["ai_action_assessment_en"] = en.strip()
            if isinstance(zh, str) and zh.strip():
                k["ai_action_assessment_zh"] = zh.strip()

    _apply(list(minimal.get("official_phase_keyframes") or []))
    _apply(list(minimal.get("preview_keyframes") or []))
    _apply(list(minimal.get("keyframes") or []))


def _nearest_dense_motion(dense: list[DenseFrame], frame_index: int) -> float:
    if not dense:
        return 0.0
    best = dense[0]
    bd = abs(best.frame_index - frame_index)
    for d in dense[1:]:
        dd = abs(d.frame_index - frame_index)
        if dd < bd:
            bd = dd
            best = d
    return float(best.motion_energy_smooth)


def _build_motion_context_local(
    *,
    fps: float,
    swing_t0: float,
    swing_t1: float,
    dense: list[DenseFrame],
    keyframes: list[dict[str, Any]],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for k in keyframes:
        fi = int(k.get("frame_index") or 0)
        rows.append(
            {
                "phase": k.get("phase"),
                "label_en": k.get("label_en"),
                "timestamp_s": float(k.get("timestamp") or 0.0),
                "frame_index": fi,
                "dense_motion_proxy": round(_nearest_dense_motion(dense, fi), 6),
            }
        )
    ctx: dict[str, Any] = {
        "pipeline": "prov3",
        "motion_authority": "prov3_keyframes_plus_dense_opencv",
        "fps": fps,
        "swing_window_s": [round(swing_t0, 4), round(swing_t1, 4)],
        "dense_frame_count": len(dense),
        "keyframes": rows,
    }
    if extras:
        ctx.update(extras)
    return ctx


def _prov3_gemini_enabled() -> bool:
    v = (os.getenv("STELLAR_PROV3_GEMINI") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _normalize_training_plan(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, Any] = {}
    for k, v in raw.items():
        ks = str(k)
        if ks.startswith("day") and len(ks) <= 6:
            suf = ks[3:]
            if suf.isdigit():
                out[f"day_{suf}"] = v
                continue
        out[ks] = v
    return out or None


def _keyframes_for_motion(ui_keyframes: list[dict[str, Any]], analysis_fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    af = max(analysis_fps, 1.0)
    for k in ui_keyframes:
        phase = k.get("phase")
        if not phase:
            continue
        ts = float(k.get("timestamp") or 0.0)
        afi = int(round(ts * af))
        rows.append(
            {
                "phase": phase,
                "label_en": k.get("label_en"),
                "timestamp": round(ts, 6),
                "frame_index": max(0, afi),
            }
        )
    return rows


async def enrich_pro_prov3_response(
    minimal: dict[str, Any],
    *,
    region: str = "global",
) -> dict[str, Any]:
    """Mutates and returns ``minimal``: fill summary / issues / suggestions / training_plan from Gemini when enabled."""
    from services.prov3_analyze_control import prov3_cancel_requested

    if not _prov3_gemini_enabled():
        # Keep _prov3_motion — prov3_api copies analysis_video to media after enrich.
        return minimal

    if prov3_cancel_requested():
        logger.info("[PRO_PROV3][GEMINI] skip — cancel requested before enrich")
        return minimal

    block = minimal.pop("_prov3_motion", None) or {}
    av = str(block.get("analysis_video") or "").strip()
    if not av or not Path(av).is_file():
        logger.warning("[PRO_PROV3][GEMINI] skip — no analysis_video at %s", av or "(empty)")
        minimal["_prov3_motion"] = block
        return minimal

    screen_mode = bool(block.get("screen_mode"))
    analysis_fps = float(block.get("analysis_fps") or 240)
    report_mode = str(minimal.get("report_mode") or "formal").strip().lower()

    cap = cv2.VideoCapture(av)
    if not cap.isOpened():
        logger.warning("[PRO_PROV3][GEMINI] skip — cannot open analysis video")
        cap.release()
        minimal["_prov3_motion"] = block
        return minimal
    try:
        fps_v = float(cap.get(cv2.CAP_PROP_FPS) or analysis_fps)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = float(n / fps_v) if fps_v > 0 else 0.0
    finally:
        cap.release()

    try:
        t0, t1 = find_swing_window_seconds(
            av,
            fps=fps_v,
            duration_s=dur,
            screen_mode=screen_mode,
        )
    except Exception as exc:
        logger.warning("[PRO_PROV3][GEMINI] swing window fallback full clip: %s", exc)
        t0, t1 = 0.0, max(dur, 0.01)

    try:
        dense = dense_scan_swing_region(
            av,
            fps=fps_v,
            t_start_s=t0,
            t_end_s=t1,
            screen_mode=screen_mode,
        )
    except Exception as exc:
        logger.warning("[PRO_PROV3][GEMINI] dense scan failed: %s", exc)
        dense = []

    kf_rows = list(minimal.get("official_phase_keyframes") or [])
    if not kf_rows:
        kf_rows = list(minimal.get("preview_keyframes") or [])
    if not kf_rows:
        kf_rows = list(minimal.get("keyframes") or [])
    kf_motion = _keyframes_for_motion(kf_rows, analysis_fps)
    extras: dict[str, Any] = {
        "prov3_analysis_trust": minimal.get("analysis_trust"),
        "prov3_fail_reasons": list(minimal.get("retry_reasons") or [])[:12],
        "prov3_screen_pipeline": bool(minimal.get("prov3_screen_pipeline")),
        "prov3_final_status": str(minimal.get("final_status") or ""),
        "low_trust_preview_only": bool(minimal.get("low_trust_preview_only")),
    }
    motion_context = _build_motion_context_local(
        fps=fps_v,
        swing_t0=t0,
        swing_t1=t1,
        dense=dense,
        keyframes=kf_motion,
        extras=extras,
    )

    if prov3_cancel_requested():
        logger.info("[PRO_PROV3][GEMINI] skip — cancel requested before AI report")
        minimal["_prov3_motion"] = block
        return minimal

    try:
        rep = await write_prov3_ai_report(
            motion_context,
            region=region,
            report_mode="limited" if report_mode == "limited" else "formal",
        )
        meta = pop_prov3_report_meta(rep)
    except Exception as exc:
        logger.exception("[PRO_PROV3][GEMINI] write_prov3_ai_report failed: %s", exc)
        minimal["_prov3_motion"] = block
        return minimal

    if isinstance(rep.get("summary"), str) and rep["summary"].strip():
        minimal["summary"] = rep["summary"].strip()
    if isinstance(rep.get("summary_zh"), str) and rep["summary_zh"].strip():
        minimal["summary_zh"] = rep["summary_zh"].strip()

    ke = rep.get("keyframe_evaluations")
    if not isinstance(ke, list) or not ke:
        lim_ctx = str(minimal.get("report_mode") or "").strip().lower() == "limited"
        ke = _synthetic_keyframe_evaluations(motion_context, low_trust=lim_ctx)
    if isinstance(ke, list) and ke:
        _merge_keyframe_ai_into_strips(minimal, ke)

    for key in ("issues", "issues_zh", "suggestions", "suggestions_zh"):
        val = rep.get(key)
        if isinstance(val, list) and val:
            minimal[key] = [str(x) for x in val if str(x).strip()]

    if isinstance(rep.get("scores"), dict) and rep["scores"]:
        minimal["scores"] = rep["scores"]
    if rep.get("total_score") is not None:
        try:
            minimal["total_score"] = float(rep["total_score"])
        except (TypeError, ValueError):
            pass

    tp = _normalize_training_plan(rep.get("training_plan"))
    if tp:
        minimal["training_plan"] = tp

    prov3 = minimal.get("prov3")
    if isinstance(prov3, dict):
        prov3 = dict(prov3)
        prov3["gemini_report"] = {
            "ai_provider": rep.get("ai_provider"),
            "meta": meta,
        }
        minimal["prov3"] = prov3

    logger.info(
        "[PRO_PROV3][GEMINI] merged report ai_provider=%s mode=%s",
        rep.get("ai_provider"),
        report_mode,
    )
    return minimal
