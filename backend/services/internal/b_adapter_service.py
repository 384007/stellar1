from __future__ import annotations

from typing import Dict, List, Optional

from services.golfdb_swingnet_service import swingnet_b_refine

FOCUS_EVENTS = {"Top", "Impact", "Finish", "Mid-downswing"}


def refine_with_b_layer(
    keyframes: List[Dict[str, object]],
    enhanced_local_frames: List[dict],
    *,
    analysis_id: Optional[str] = None,
) -> List[Dict[str, object]]:
    # B-path: local peak on SwingNet probs when A used wmcnally/golfdb (ctx per analysis_id).
    refined_in = (
        swingnet_b_refine(analysis_id or "", list(keyframes))
        if analysis_id
        else list(keyframes)
    )
    frame_indices = {int(item.get("frame_index", 0)) for item in enhanced_local_frames}
    refined: List[Dict[str, object]] = []
    for item in refined_in:
        event_name = str(item.get("event_name"))
        frame_idx = int(item.get("frame_index", 0))
        confidence = float(item.get("confidence", 0.0))
        if event_name in FOCUS_EVENTS:
            if (frame_idx + 1) in frame_indices:
                frame_idx = frame_idx + 1
            confidence = min(0.96, confidence + 0.08)
        else:
            confidence = min(0.92, confidence + 0.03)
        cloned = dict(item)
        cloned["frame_index"] = frame_idx
        cloned["confidence"] = round(confidence, 3)
        refined.append(cloned)
    return refined
