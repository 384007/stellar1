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
from services.lite_ab_select_service import select_lite_ab_final_rows
from services.lite_a_gate_service import run_lite_a_gate
from services.lite_b_gate_service import run_lite_b_gate
from services.lite_keyframe_candidate_a import lite_build_candidate_a_rows
from services.lite_keyframe_candidate_b import lite_build_candidate_b_rows
from services.lite_keyframe_export import lite_persist_keyframe_images
from services.lite_keyframe_heuristic import (
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
_LOG_AB = "[lite_ab]"

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


def _closest_pose_index(poses: list[dict[str, Any]], impact_fi: int, vfps: float) -> int:
    if not poses:
        return 0
    target_t = float(impact_fi) / max(float(vfps), 1e-6)
    best_i = 0
    best_d = 1e9
    for i, p in enumerate(poses):
        if not isinstance(p, dict):
            continue
        t = float(p.get("timestamp", 0))
        d = abs(t - target_t)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _impact_frame_index(rows: list[dict[str, Any]]) -> int:
    for r in rows:
        if str(r.get("event_name")) == "Impact":
            return int(r.get("frame_index", 0))
    return 0


async def run_lite_independent_pipeline(video_path: str, *, region: str = "global") -> dict[str, Any]:
    """Mobile-first Lite: normalize → dual internal candidates → gates → select → export → AI → predict."""
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

        max_fi = max(0, total_frames - 1)

        rows_a0 = lite_build_candidate_a_rows(indices, motions)
        rows_a0 = lite_refine_impact_row(rows_a0, hint_fi)
        rows_a = lite_enforce_monotonic_frame_indices(rows_a0, max_fi)
        status_a, reasons_a = run_lite_a_gate(rows_a, impact_hint_frame_index=hint_fi)
        logger.info("%s candidate A built frames=%d", _LOG_AB, len(rows_a))
        logger.info("%s candidate A gate result status=%s reasons=%s", _LOG_AB, status_a, reasons_a)

        rows_b0 = lite_build_candidate_b_rows(indices, motions)
        rows_b0 = lite_refine_impact_row(rows_b0, hint_fi)
        rows_b = lite_enforce_monotonic_frame_indices(rows_b0, max_fi)
        status_b, reasons_b = run_lite_b_gate(
            rows_b,
            impact_hint_frame_index=hint_fi,
            frame_indices=indices,
            motions=motions,
        )
        logger.info("%s candidate B built frames=%d", _LOG_AB, len(rows_b))
        logger.info("%s candidate B gate result status=%s reasons=%s", _LOG_AB, status_b, reasons_b)

        final_rows, phase_ok, path_tag = select_lite_ab_final_rows(
            rows_a,
            rows_b,
            status_a=status_a,
            reasons_a=reasons_a,
            status_b=status_b,
            reasons_b=reasons_b,
            impact_hint_frame_index=hint_fi,
            logger=logger,
        )
        logger.info("%s final phase validation passed=%s path=%s", _LOG_AB, phase_ok, path_tag)

        out_dir = str(Path(work_dir) / "lite_keyframes")
        saved = await asyncio.to_thread(lite_persist_keyframe_images, cleaned_path, final_rows, out_dir)
        keyframes = _build_public_keyframes(final_rows, saved, vfps)
        if len(keyframes) != 8:
            raise RuntimeError("lite_keyframe_export_incomplete")

        keyframe_images = [k["image_base64"] for k in keyframes]
        rep_pose = poses[len(poses) // 2] if poses else {"angles": {}}

        impact_fi = _impact_frame_index(final_rows)
        impact_pose_idx = _closest_pose_index(poses, impact_fi, vfps)

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
            impact_pose_idx=impact_pose_idx,
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
