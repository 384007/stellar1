from __future__ import annotations

import os
from pathlib import Path

import cv2

from services.provider_registry import role_log
from services.provider_schema import provider_result


def _resolve_yolo11_weights() -> str:
    """Prefer explicit env, then Modal volume, then image-baked weights, then Ultralytics default name."""
    env = (os.getenv("STELLAR_YOLO_WEIGHTS") or "").strip()
    if env and Path(env).is_file():
        return env
    for p in ("/models/yolo11n.pt", "/opt/stellar-weights/yolo11n.pt"):
        if Path(p).is_file():
            return p
    return "yolo11n.pt"


def run(video_path: str, frame_stride: int = 3) -> dict:
    try:
        from ultralytics import YOLO
    except Exception:
        role_log("[ROLE=YOLO11] status=dependency_missing")
        return provider_result(role="detection", provider_name="yolo11", status="dependency_missing", error_reason="dependency_missing")

    weights = _resolve_yolo11_weights()
    try:
        model = YOLO(weights)
    except Exception as e:
        role_log(f"[ROLE=YOLO11] status=model_load_failed err={e}")
        return provider_result(role="detection", provider_name="yolo11", status="model_load_failed", error_reason=str(e))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return provider_result(role="detection", provider_name="yolo11", status="inference_failed", error_reason="video_open_failed")

    frame_idx = 0
    detections = []
    person = club = ball = 0
    sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        sampled += 1
        results = model.predict(source=frame, verbose=False, conf=0.2)
        for r in results:
            names = r.names
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls_name = str(names.get(cls_id, cls_id))
                mapped = "club" if cls_name in {"baseball bat", "tennis racket"} else ("ball" if cls_name in {"sports ball", "baseball"} else cls_name)
                if mapped == "person":
                    person += 1
                elif mapped == "club":
                    club += 1
                elif mapped == "ball":
                    ball += 1
                detections.append({
                    "frame_index": frame_idx,
                    "class_name": mapped,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                })
        frame_idx += 1
    cap.release()
    role_log(
        f"[ROLE=YOLO11] weights={weights} model=yolo11n frames={frame_idx} "
        f"detected_frames={sampled} person_boxes={person} club_boxes={club} ball_boxes={ball}"
    )
    return provider_result(
        role="detection",
        provider_name="yolo11",
        provider_version="yolo11n",
        status="ok",
        frame_count=frame_idx,
        confidence_summary={"person_boxes": person, "club_boxes": club, "ball_boxes": ball},
        payload={"detections": detections},
    )
