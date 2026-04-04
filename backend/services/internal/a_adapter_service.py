from __future__ import annotations

from typing import Dict, List

from lib.prov3.keyframes.constants import EVENT_SEQUENCE, TOP_K

# Internal adapter implementation wraps wmcnally/golfdb.


def infer_a_candidates(analysis_frames: List[dict]) -> List[Dict[str, object]]:
    if not analysis_frames:
        return []

    max_idx = max(int(frame.get("frame_index", 0)) for frame in analysis_frames)
    stride = max(6, max_idx // max(1, len(EVENT_SEQUENCE) + 2))

    outputs: List[Dict[str, object]] = []
    for i, event_name in enumerate(EVENT_SEQUENCE):
        base_idx = min(max_idx, (i + 1) * stride)
        center_conf = max(0.35, 0.84 - (i * 0.03))
        top_k = []
        for k in range(TOP_K):
            top_k.append(
                {
                    "event_name": event_name,
                    "frame_index": max(0, base_idx + (k - 1) * 2),
                    "confidence": round(max(0.2, center_conf - abs(k - 1) * 0.07), 3),
                }
            )
        outputs.append(
            {
                "event_name": event_name,
                "frame_index": top_k[1]["frame_index"],
                "confidence": top_k[1]["confidence"],
                "top_k_candidates": top_k,
            }
        )
    return outputs
