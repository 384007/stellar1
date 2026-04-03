from __future__ import annotations

from services.provider_registry import role_log
from services.provider_schema import provider_result


def run(detections: list[dict]) -> dict:
    try:
        import numpy as np
        import supervision as sv
    except Exception:
        role_log("[ROLE=BYTETRACK] status=dependency_missing")
        return provider_result(role="tracking", provider_name="bytetrack", status="dependency_missing", error_reason="dependency_missing")
    if not detections:
        return provider_result(
            role="tracking",
            provider_name="bytetrack",
            status="ok",
            payload={"person_tracks": [], "club_tracks": [], "ball_tracks": [], "track_confidence": 0.0, "occlusion_flags": {}},
        )
    tracker = sv.ByteTrack()
    by_frame: dict[int, list[dict]] = {}
    for d in detections:
        fidx = int(d.get("frame_index", 0))
        by_frame.setdefault(fidx, []).append(d)
    person_tracks: list[dict] = []
    club_tracks: list[dict] = []
    ball_tracks: list[dict] = []
    for fidx in sorted(by_frame):
        rows = by_frame[fidx]
        xyxy = np.array([r.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]) for r in rows], dtype=np.float32)
        conf = np.array([float(r.get("confidence", 0.0)) for r in rows], dtype=np.float32)
        cls = np.array([0 for _ in rows], dtype=np.int32)
        det_obj = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracked = tracker.update_with_detections(det_obj)
        tids = tracked.tracker_id.tolist() if tracked.tracker_id is not None else [-1] * len(rows)
        for i, row in enumerate(rows):
            out = {
                "frame_index": fidx,
                "track_id": int(tids[i]) if i < len(tids) else -1,
                "bbox_xyxy": list(row.get("bbox_xyxy", [])),
                "confidence": float(row.get("confidence", 0.0)),
            }
            cname = str(row.get("class_name", "")).lower()
            if cname == "person":
                person_tracks.append(out)
            elif cname == "club":
                club_tracks.append(out)
            elif cname == "ball":
                ball_tracks.append(out)
    role_log(
        f"[ROLE=BYTETRACK] status=ok frames={len(by_frame)} person_tracks={len(person_tracks)} club_tracks={len(club_tracks)} ball_tracks={len(ball_tracks)}"
    )
    return provider_result(
        role="tracking",
        provider_name="bytetrack",
        status="ok",
        confidence_summary={"avg_track_len": float(len(person_tracks) + len(club_tracks) + len(ball_tracks)) / max(len(by_frame), 1)},
        payload={
            "person_tracks": person_tracks,
            "club_tracks": club_tracks,
            "ball_tracks": ball_tracks,
            "track_confidence": 0.8 if (person_tracks or club_tracks or ball_tracks) else 0.0,
            "occlusion_flags": {},
        },
    )
