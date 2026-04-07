from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from services.gemini_service import LITE_AI_TIMEOUT_S, analyze_swing_lite, cap_confidence
from services.handedness_service import detect_handedness
from services.internal.a_gate_service import run_a_gate
from services.lite_keyframe_export import lite_persist_keyframe_images
from services.lite_keyframe_heuristic import (
    lite_build_eight_keyframe_rows,
    lite_enforce_monotonic_frame_indices,
    lite_refine_impact_row,
)
from services.lite_preview_sample import lite_sample_preview_bgr
from services.lite_timeline_motion import (
    lite_build_uniform_timeline,
    lite_impact_hint_from_timeline,
    lite_motion_along_timeline,
)
from services.lite_video_cleanup import lite_light_clean_video
from services.pose_backend_service import extract_pose_stream
from services.shot_predictor import calibrate_prediction, predict_shot

logger = logging.getLogger(__name__)

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

_LITE_POSE_PREVIEW_FRAMES = int(os.getenv("STELLAR_LITE_POSE_PREVIEW_FRAMES", "40"))


async def _lite_club_from_previews(frames: list[Any], region: str) -> dict[str, Any]:
    from services.club_detector import detect_club

    results: list[dict[str, Any]] = []
    for f in frames:
        try:
            r = await detect_club(f, region)
            results.append(dict(r))
        except Exception as exc:
            logger.warning("[lite] club preview frame failed: %s", exc)
    valid = [r for r in results if str(r.get("club_type") or "").upper() not in ("", "UNKNOWN")]
    if not valid:
        return {"club_type": "UNKNOWN", "club_group": "IRON", "confidence": 0.0}
    votes: dict[str, dict[str, Any]] = {}
    for r in valid:
        ct = str(r.get("club_type") or "").upper()
        cg = str(r.get("club_group") or "IRON").upper()
        conf = float(r.get("confidence") or 0.0)
        if ct not in votes:
            votes[ct] = {"count": 0, "total_conf": 0.0, "group": cg}
        votes[ct]["count"] += 1
        votes[ct]["total_conf"] += conf
    winner = max(votes.items(), key=lambda x: (x[1]["count"], x[1]["total_conf"]))[0]
    agg = votes[winner]
    avg_conf = agg["total_conf"] / max(agg["count"], 1)
    return {
        "club_type": winner,
        "club_group": str(agg.get("group") or "IRON"),
        "confidence": round(min(1.0, avg_conf), 4),
    }


def _build_public_keyframes(
    rows: list[dict[str, Any]],
    saved: list[dict[str, Any]],
    vfps: float,
) -> list[dict[str, Any]]:
    path_map: dict[str, str] = {}
    for s in saved:
        ev = str(s.get("event_name") or "")
        p = str(s.get("file_path") or s.get("image_path") or "")
        if ev and p:
            path_map[ev] = p
    vfps = max(float(vfps), 1e-6)
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
                "timestamp": round(fi / vfps, 4),
                "image_base64": b64,
            }
        )
    return out


async def run_lite_independent_pipeline(video_path: str, *, region: str = "global") -> dict[str, Any]:
    """Single-chain lite pipeline: clean → preview/club → pose → hand → ≤400 timeline → motion → 8 KF → AI → predict."""
    with tempfile.TemporaryDirectory(prefix="lite_pipeline_") as work_dir:
        clean_meta = await asyncio.to_thread(lite_light_clean_video, video_path, work_dir)
        cleaned_path = str(clean_meta["path"])
        vfps = float(clean_meta["fps"])
        total_frames = int(clean_meta["total_frames"])
        duration_s = float(clean_meta.get("duration_s") or (total_frames / max(vfps, 1e-6)))

        preview_bgr = await asyncio.to_thread(
            lite_sample_preview_bgr,
            cleaned_path,
            (0.25, 0.4, 0.6),
        )
        club_info = await _lite_club_from_previews(preview_bgr, region)
        ct = str(club_info.get("club_type") or "UNKNOWN").upper()
        cg = str(club_info.get("club_group") or "IRON").upper()
        cconf = float(club_info.get("confidence") or 0.0)

        pose_bundle = await asyncio.to_thread(
            extract_pose_stream,
            cleaned_path,
            _LITE_POSE_PREVIEW_FRAMES,
        )
        poses = list(pose_bundle.get("poses") or [])
        hand_info = detect_handedness(poses, None) if poses else {"hand": "UNKNOWN", "confidence": 0.0}
        hand = str(hand_info.get("hand") or "UNKNOWN")

        timeline = lite_build_uniform_timeline(total_frames, vfps)
        if len(timeline) < 8:
            raise RuntimeError("lite_timeline_too_short")
        indices = [int(t["frame_index"]) for t in timeline]
        motions = await asyncio.to_thread(lite_motion_along_timeline, cleaned_path, indices)
        preloc = lite_impact_hint_from_timeline(indices, motions, vfps, duration_s)
        hint_fi = int(round(float(preloc.get("impact_hint_s") or 0.0) * vfps))

        rows = lite_build_eight_keyframe_rows(indices, motions)
        rows = lite_refine_impact_row(rows, hint_fi)
        rows = lite_enforce_monotonic_frame_indices(rows, max(0, total_frames - 1))
        a_status, _fail_reasons = run_a_gate(rows)
        phase_ok = a_status == "pass"

        out_dir = str(Path(work_dir) / "lite_keyframes")
        saved = await asyncio.to_thread(lite_persist_keyframe_images, cleaned_path, rows, out_dir)
        keyframes = _build_public_keyframes(rows, saved, vfps)
        if len(keyframes) != 8:
            raise RuntimeError("lite_keyframe_export_incomplete")

        keyframe_images = [k["image_base64"] for k in keyframes]
        rep_pose = poses[len(poses) // 2] if poses else {"angles": {}}

        # Always ask for full Lite JSON (scores / issues / summaries). Keyframe gate state is surfaced
        # only via analysis_reliability.phase_validation — not by switching to the "unreliable strip" prompt,
        # which tends to produce thin or overly cautious copy while users still expect a normal report.
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

        all_angles = [p.get("angles", {}) for p in poses if p.get("angles")]
        swing_dur = (
            (poses[-1].get("timestamp", 1.2) - poses[0].get("timestamp", 0.0)) if len(poses) >= 2 else 1.2
        )
        prediction = predict_shot(
            rep_pose,
            swing_duration=swing_dur,
            all_frame_angles=all_angles,
            club_type=ct if ct != "UNKNOWN" else None,
            club_group=cg if ct != "UNKNOWN" else None,
            hand=hand,
            hand_confidence=float(hand_info.get("confidence") or 0.0),
            poses=poses,
            impact_pose_idx=max(0, len(poses) // 2),
        )
        if ct != "UNKNOWN":
            prediction["club_type"] = ct
            prediction["club_group"] = cg
            prediction["club_detection_confidence"] = cconf
            prediction = calibrate_prediction(prediction, club_type=ct, club_group=cg)
        else:
            prediction.setdefault("club_type", "UNKNOWN")
            prediction.setdefault("club_group", "IRON")

        tracking_quality = 1.0 if len(poses) >= 30 else (0.65 if len(poses) >= 15 else 0.35)
        # Single soft-fail signal for keyframe gate (avoid stacking phase_vision_unreliable on top of it).
        analysis_reliability = cap_confidence(
            ai_result,
            phase_validation={"passed": phase_ok},
            hand=hand,
            tracking_quality=tracking_quality,
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
        }
