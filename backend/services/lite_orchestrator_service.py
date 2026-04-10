"""Lite orchestrator: fake-240 preprocess → SwingNet A/B (Lite mirror, no prov3) → export → AI → prediction."""

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
from services.golfdb_swingnet_paths import swingnet_weights_configured
from services.handedness_service import detect_handedness
from services.club_detector import detect_club_three_frames_from_video
from services.lite_a_extractor_service import run_lite_a_extract as run_lite_heuristic_a_extract
from services.lite_ab_mirror.orchestrator import run_lite_ab_after_preprocess
from services.lite_b_refiner_service import run_lite_b_refine as run_lite_heuristic_b_refine
from services.lite_keyframe_export import lite_persist_keyframe_images
from services.lite_preprocess_service import run_lite_preprocess
from services.provider_registry import role_log
from services.lite_timeline_motion import lite_build_uniform_timeline
from services.shot_predictor import calibrate_prediction, predict_shot

logger = logging.getLogger(__name__)
_LOG = "[lite_orch]"

_MIN_HAND_CONF = 0.30
_MIN_CLUB_CONF = 0.40

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


async def run_lite_orchestrator(video_path: str, *, region: str = "global") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lite_orch_") as work_dir:
        pre = await asyncio.to_thread(run_lite_preprocess, video_path, work_dir)
        vfps = float(pre["analysis_fps"])
        analysis_video = str(pre["analysis_video_path"])
        poses = list(pre["poses"])

        if swingnet_weights_configured():
            role_log("[ROLE=LITE_PIPELINE] swingnet_checkpoint_ok entering_ab_path (decode+infer may take minutes on CPU)")
            final_rows, trust_tier, ab_phase_pass, ab_reasons = await asyncio.to_thread(
                run_lite_ab_after_preprocess,
                pre,
            )
        else:
            role_log("[ROLE=LITE_PIPELINE] swingnet_disabled using_heuristic_ab")
            logger.warning(
                "%s SwingNet disabled (no checkpoint) — heuristic Lite A/B fallback",
                _LOG,
            )
            pre_h = {
                **pre,
                "timeline": lite_build_uniform_timeline(
                    int(pre["total_frames"]),
                    float(pre["analysis_fps"]),
                ),
            }
            a_out = await run_lite_heuristic_a_extract(pre_h, region=region)
            ab_reasons = list(a_out.get("fail_reasons") or [])
            if a_out["a_pass"]:
                final_rows = a_out["rows"]
                trust_tier = "high"
                ab_phase_pass = True
            else:
                b_out = await asyncio.to_thread(run_lite_heuristic_b_refine, pre_h, a_out)
                final_rows = b_out["rows"]
                if b_out["b_pass"]:
                    trust_tier = "medium"
                    ab_phase_pass = True
                else:
                    trust_tier = "low"
                    ab_phase_pass = False
                    ab_reasons = ab_reasons + list(b_out.get("fail_reasons") or [])

        if len(final_rows) < 8:
            logger.error("%s incomplete keyframes count=%s reasons=%s", _LOG, len(final_rows), ab_reasons)
            raise RuntimeError("lite_keyframes_incomplete")

        if ab_phase_pass and trust_tier == "high":
            logger.info("%s path=A_high_trust", _LOG)
        elif ab_phase_pass and trust_tier == "medium":
            logger.info("%s path=B_medium_trust ab_reasons=%s", _LOG, ab_reasons)
        else:
            logger.info("%s path=B_low_trust ab_reasons=%s", _LOG, ab_reasons)
        role_log(
            f"[ROLE=LITE_PIPELINE] ab_done rows={len(final_rows)} trust={trust_tier} "
            f"phase_pass={ab_phase_pass} next=club_vision_then_gemini"
        )

        hand_info = detect_handedness(poses, None) if poses else {"hand": "UNKNOWN", "confidence": 0.0}
        hand = str(hand_info.get("hand") or "UNKNOWN")
        hconf = float(hand_info.get("confidence") or 0.0)
        hand_ok = hand != "UNKNOWN" and hconf >= _MIN_HAND_CONF

        # 单次 Lite 分析内只跑 1 次杆型视觉（与主流程同一进程，无额外 Modal HTTP）
        club_info = await detect_club_three_frames_from_video(analysis_video, region=region)
        ct = str(club_info.get("club_type") or "UNKNOWN").upper()
        cg = str(club_info.get("club_group") or "IRON").upper()
        cconf = float(club_info.get("confidence") or 0.0)
        club_ok = ct != "UNKNOWN" and cconf >= _MIN_CLUB_CONF

        phase_passed_product = ab_phase_pass and hand_ok and club_ok

        out_dir = str(Path(work_dir) / "lite_keyframes")
        saved = await asyncio.to_thread(
            lite_persist_keyframe_images,
            analysis_video,
            final_rows,
            out_dir,
            poses=poses,
            analysis_fps=vfps,
        )
        keyframes = _build_public_keyframes(final_rows, saved, vfps)
        if len(keyframes) != 8:
            raise RuntimeError("lite_keyframe_export_incomplete")

        keyframe_images = [k["image_base64"] for k in keyframes]
        rep_pose = poses[len(poses) // 2] if poses else {"angles": {}}

        impact_fi = _impact_frame_index(final_rows)
        impact_pose_idx = _closest_pose_index(poses, impact_fi, vfps)

        role_log("[ROLE=LITE_PIPELINE] gemini_lite_start (may take 30–120s)")
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
            phase_validation={"passed": phase_passed_product},
            hand=hand,
            tracking_quality=tracking_quality,
            lite_trust_tier=trust_tier,
            club_type=ct if ct != "UNKNOWN" else None,
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
