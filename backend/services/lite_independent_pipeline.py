from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.gemini_service import LITE_AI_TIMEOUT_S, analyze_swing_lite, cap_confidence
from services.handedness_service import detect_handedness
from services.internal.frame_enhance_service import persist_final_keyframe_images
from services.prov3_dense_scan_service import dense_scan_swing_region
from services.prov3_keyframe_a_extractor_service import run_a_extract
from services.prov3_keyframe_preprocess_service import run_preprocess
from services.shot_predictor import predict_shot

_EVENT_TO_LITE_PHASE = {
    "Address": ("address", "Address", "准备"),
    "Toe-up": ("takeaway", "Takeaway", "起杆"),
    "Mid-backswing": ("backswing", "Backswing", "上杆"),
    "Top": ("top", "Top of Swing", "顶点"),
    "Mid-downswing": ("downswing", "Downswing", "下杆"),
    "Impact": ("impact", "Impact", "击球"),
    "Mid-follow-through": ("follow_through", "Follow-Through", "送杆"),
    "Finish": ("finish", "Finish", "收杆"),
}


async def run_lite_independent_pipeline(video_path: str, *, region: str = "global") -> dict[str, Any]:
    """Lite-only backend pipeline with internal stages hidden from API callers."""
    with tempfile.TemporaryDirectory(prefix="lite_pipeline_") as work_dir:
        pre = await asyncio.to_thread(run_preprocess, video_path, work_dir, screen_mode=False)
        analysis_video = str(pre.analysis_video)
        analysis_frames = list(pre.analysis_frames or [])

        fake_timeline = _build_fake_400_timeline(analysis_frames)
        preloc = _motion_prelocalize(analysis_video)

        a_result = await asyncio.to_thread(
            run_a_extract,
            analysis_id=f"lite_{uuid.uuid4().hex[:12]}",
            analysis_video=analysis_video,
            preprocess_meta=dict(pre.preprocess_meta.model_dump()),
            analysis_frames=fake_timeline,
        )
        rows = list(a_result.keyframes or [])
        if len(rows) != 8:
            raise RuntimeError("lite_keyframe_extract_incomplete")

        final_rows = _adjust_rows_with_preloc(rows, preloc)
        out_dir = str(Path(work_dir) / "lite_keyframes")
        saved = await asyncio.to_thread(persist_final_keyframe_images, analysis_video, final_rows, out_dir)
        keyframes = _build_public_keyframes(final_rows, saved)
        if len(keyframes) != 8:
            raise RuntimeError("lite_keyframe_export_incomplete")

        keyframe_images = [k["image_base64"] for k in keyframes]
        poses = list(pre.poses or [])
        rep_pose = poses[len(poses) // 2] if poses else {"angles": {}}

        ai_result = await asyncio.wait_for(
            analyze_swing_lite(
                pose_data={
                    "angles": rep_pose.get("angles", {}),
                    "all_frame_angles": [p.get("angles", {}) for p in poses if isinstance(p, dict)],
                },
                keyframe_images=keyframe_images,
                region=region,
                phase_images_reliable=True,
            ),
            timeout=LITE_AI_TIMEOUT_S + 45.0,
        )

        hand_info = detect_handedness(poses) if poses else {"hand": "UNKNOWN", "confidence": 0.0}
        hand = str(hand_info.get("hand") or "UNKNOWN")
        all_angles = [p.get("angles", {}) for p in poses if p.get("angles")]
        swing_dur = (poses[-1].get("timestamp", 1.2) - poses[0].get("timestamp", 0.0)) if len(poses) >= 2 else 1.2
        prediction = predict_shot(
            rep_pose,
            swing_duration=swing_dur,
            all_frame_angles=all_angles,
            hand=hand,
            hand_confidence=float(hand_info.get("confidence") or 0.0),
            poses=poses,
            impact_pose_idx=max(0, len(poses) // 2),
        )

        tracking_quality = 1.0 if len(poses) >= 40 else (0.6 if len(poses) >= 20 else 0.3)
        analysis_reliability = cap_confidence(
            ai_result,
            phase_validation={"passed": True},
            hand=hand,
            tracking_quality=tracking_quality,
            phase_vision_reliable=True,
        )

        return {
            "analysis_id": str(uuid.uuid4()),
            "type": "lite",
            "analysis_mode": "standard",
            "keyframes": keyframes,
            "summary": ai_result.get("summary", ""),
            "summary_zh": ai_result.get("summary_zh", ""),
            "issues": ai_result.get("issues", []),
            "issues_zh": ai_result.get("issues_zh", []),
            "suggestions": ai_result.get("suggestions", []),
            "suggestions_zh": ai_result.get("suggestions_zh", []),
            "scores": ai_result.get("scores", {}),
            "total_score": ai_result.get("total_score", 0),
            "analysis_reliability": analysis_reliability,
            "prediction": prediction,
            # Internal-only details (must be removed by packer).
            "_internal_pipeline": {
                "timeline": fake_timeline,
                "prelocalization": preloc,
                "a_status": a_result.a_status,
                "a_fail_reasons": list(a_result.fail_reasons or []),
            },
        }


def _build_fake_400_timeline(analysis_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not analysis_frames:
        return []
    ordered = sorted(analysis_frames, key=lambda x: int(x.get("frame_index", 0)))
    if len(ordered) <= 400:
        return ordered
    picks = np.linspace(0, len(ordered) - 1, num=400, dtype=np.int64)
    return [ordered[int(i)] for i in picks]


def _motion_prelocalize(analysis_video: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(analysis_video)
    if not cap.isOpened():
        return {"impact_hint_s": 0.0, "window_s": [0.0, 0.0]}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 240.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        return {"impact_hint_s": 0.0, "window_s": [0.0, 0.0]}

    dur = total / max(fps, 1e-6)
    dense = dense_scan_swing_region(
        analysis_video,
        fps=fps,
        t_start_s=0.0,
        t_end_s=dur,
        max_frames=2400,
        pose_priority=True,
    )
    if not dense:
        hint = round(dur * 0.72, 3)
        return {"impact_hint_s": hint, "window_s": [max(0.0, hint - 0.9), min(dur, hint + 0.9)]}
    best = max(dense, key=lambda x: float(x.motion_energy_smooth))
    hint = float(best.timestamp_s)
    return {
        "impact_hint_s": round(hint, 3),
        "window_s": [round(max(0.0, hint - 0.9), 3), round(min(dur, hint + 0.9), 3)],
    }


def _adjust_rows_with_preloc(rows: list[dict[str, Any]], preloc: dict[str, Any]) -> list[dict[str, Any]]:
    if len(rows) != 8:
        return rows
    impact_hint_s = float(preloc.get("impact_hint_s") or 0.0)
    fps = 240.0
    hint_fi = int(round(impact_hint_s * fps))
    adjusted: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        if str(copied.get("event_name")) == "Impact":
            top_k = list(copied.get("top_k_candidates") or [])
            if top_k:
                best = min(top_k, key=lambda c: abs(int(c.get("frame_index", 0)) - hint_fi))
                copied["frame_index"] = int(best.get("frame_index", copied.get("frame_index", 0)))
                copied["confidence"] = float(best.get("confidence", copied.get("confidence", 0.0)))
        adjusted.append(copied)
    adjusted.sort(key=lambda x: int(x.get("frame_index", 0)))
    # keep event order by original sequence to avoid accidental swap
    event_order = [
        "Address", "Toe-up", "Mid-backswing", "Top", "Mid-downswing", "Impact", "Mid-follow-through", "Finish",
    ]
    idx_map = {name: i for i, name in enumerate(event_order)}
    adjusted.sort(key=lambda x: idx_map.get(str(x.get("event_name")), 99))
    return adjusted


def _build_public_keyframes(rows: list[dict[str, Any]], saved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path_map = {str(s.get("event_name")): str(s.get("image_path")) for s in saved}
    out: list[dict[str, Any]] = []
    for row in rows:
        event = str(row.get("event_name") or "")
        if event not in _EVENT_TO_LITE_PHASE:
            continue
        image_path = path_map.get(event)
        if not image_path or not os.path.isfile(image_path):
            continue
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        phase, label_en, label_zh = _EVENT_TO_LITE_PHASE[event]
        fi = int(row.get("frame_index", 0))
        out.append(
            {
                "phase": phase,
                "label_en": label_en,
                "label_zh": label_zh,
                "frame_index": fi,
                "source_pose_idx": None,
                "source_frame_index": fi,
                "timestamp": round(fi / 240.0, 4),
                "confidence": float(row.get("confidence", 0.0)),
                "image_base64": b64,
            }
        )
    return out
