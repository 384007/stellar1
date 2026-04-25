from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Iterable

import cv2
import httpx
import mediapipe as mp
import numpy as np


POSE_KEYS = {
    "head": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

_FALLBACK_WINDOW_FRAMES = 60


@dataclass
class TrackPoint:
    frame_index: int
    timestamp: float
    nx: float
    ny: float
    confidence: float
    source: str


class ClubDetectorAdapter:
    name = "base"
    unavailable_reason: str | None = None

    def detect(self, frame: np.ndarray, frame_index: int, pose: dict[str, Any] | None) -> tuple[TrackPoint | None, dict[str, Any] | None]:
        return None, None


class BallTrackerAdapter:
    name = "base"
    unavailable_reason: str | None = None

    def detect(self, frame: np.ndarray, frame_index: int) -> TrackPoint | None:
        return None


def _load_yolo_model(weights_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed; YOLO adapter disabled") from exc
    return YOLO(weights_path)


class YoloClubDetector(ClubDetectorAdapter):
    name = "yolo_club_head"

    def __init__(self, weights_path: str):
        self.model = _load_yolo_model(weights_path)

    def detect(self, frame: np.ndarray, frame_index: int, pose: dict[str, Any] | None) -> tuple[TrackPoint | None, dict[str, Any] | None]:
        result = self.model.predict(source=frame, verbose=False, conf=0.15)[0]
        club_head = None
        club_shaft = None
        ts = pose["timestamp"] if pose else 0.0
        for b in result.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            nx, ny = cx / frame.shape[1], cy / frame.shape[0]
            if cls_id == 0 and (club_head is None or conf > club_head.confidence):
                club_head = TrackPoint(frame_index, ts, nx, ny, conf, self.name)
            elif cls_id == 1 and (club_shaft is None or conf > club_shaft["confidence"]):
                club_shaft = {
                    "frame_index": frame_index,
                    "timestamp": ts,
                    "x1": x1 / frame.shape[1],
                    "y1": y1 / frame.shape[0],
                    "x2": x2 / frame.shape[1],
                    "y2": y2 / frame.shape[0],
                    "confidence": conf,
                    "source": self.name,
                }
        return club_head, club_shaft


class RoboflowClubDetector(ClubDetectorAdapter):
    name = "roboflow_club_head"

    def __init__(self, api_key: str, model_ref: str):
        self.api_key = api_key
        self.model_ref = model_ref

    def detect(self, frame: np.ndarray, frame_index: int, pose: dict[str, Any] | None) -> tuple[TrackPoint | None, dict[str, Any] | None]:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return None, None
        url = f"https://detect.roboflow.com/{self.model_ref}?api_key={self.api_key}&confidence=15"
        try:
            resp = httpx.post(url, content=encoded.tobytes(), timeout=10.0, headers={"Content-Type": "application/x-image"})
            if resp.status_code != 200:
                return None, None
            data = resp.json()
        except Exception:
            return None, None

        best = None
        for pred in data.get("predictions", []):
            cls_name = str(pred.get("class", "")).lower()
            if "club" in cls_name and "head" in cls_name:
                conf = float(pred.get("confidence", 0.0))
                if best is None or conf > best.confidence:
                    nx = float(pred.get("x", 0.0)) / frame.shape[1]
                    ny = float(pred.get("y", 0.0)) / frame.shape[0]
                    best = TrackPoint(frame_index, pose["timestamp"] if pose else 0.0, nx, ny, conf, self.name)
        return best, None


class PoseProxyClubDetector(ClubDetectorAdapter):
    name = "mediapipe_pose_proxy"

    def detect(self, frame: np.ndarray, frame_index: int, pose: dict[str, Any] | None) -> tuple[TrackPoint | None, dict[str, Any] | None]:
        if not pose:
            return None, None
        lw = pose["landmarks"].get("left_wrist")
        rw = pose["landmarks"].get("right_wrist")
        le = pose["landmarks"].get("left_elbow")
        re = pose["landmarks"].get("right_elbow")
        if not all([lw, rw, le, re]):
            return None, None
        wrist = np.array([(lw["x"] + rw["x"]) * 0.5, (lw["y"] + rw["y"]) * 0.5], dtype=float)
        elbow = np.array([(le["x"] + re["x"]) * 0.5, (le["y"] + re["y"]) * 0.5], dtype=float)
        direction = wrist - elbow
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None, None
        ext = wrist + (direction / norm) * 0.08
        return TrackPoint(frame_index, pose["timestamp"], float(np.clip(ext[0], 0, 1)), float(np.clip(ext[1], 0, 1)), 0.35, self.name), None


class YoloBallTracker(BallTrackerAdapter):
    name = "yolo_golf_ball"

    def __init__(self, weights_path: str):
        self.model = _load_yolo_model(weights_path)

    def detect(self, frame: np.ndarray, frame_index: int) -> TrackPoint | None:
        result = self.model.predict(source=frame, verbose=False, conf=0.1)[0]
        best = None
        for b in result.boxes:
            cls_id = int(b.cls.item())
            if cls_id != 2:
                continue
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            p = TrackPoint(frame_index, 0.0, cx / frame.shape[1], cy / frame.shape[0], conf, self.name)
            if best is None or p.confidence > best.confidence:
                best = p
        return best


class TrackNetApiBallTracker(BallTrackerAdapter):
    name = "tracknet"

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def detect(self, frame: np.ndarray, frame_index: int) -> TrackPoint | None:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = httpx.post(
                self.api_url,
                files={"file": (f"frame_{frame_index}.jpg", encoded.tobytes(), "image/jpeg")},
                headers=headers,
                timeout=8.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            nx = float(data.get("nx", -1))
            ny = float(data.get("ny", -1))
            conf = float(data.get("confidence", 0.0))
            if nx < 0 or ny < 0:
                return None
            return TrackPoint(frame_index, 0.0, nx, ny, conf, self.name)
        except Exception:
            return None


class RoboflowBallTracker(BallTrackerAdapter):
    name = "roboflow_ball"

    def __init__(self, api_key: str, model_ref: str):
        self.api_key = api_key
        self.model_ref = model_ref

    def detect(self, frame: np.ndarray, frame_index: int) -> TrackPoint | None:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        url = f"https://detect.roboflow.com/{self.model_ref}?api_key={self.api_key}&confidence=10"
        try:
            resp = httpx.post(url, content=encoded.tobytes(), timeout=10.0, headers={"Content-Type": "application/x-image"})
            if resp.status_code != 200:
                return None
            data = resp.json()
            best = None
            for pred in data.get("predictions", []):
                cls_name = str(pred.get("class", "")).lower()
                if "ball" not in cls_name:
                    continue
                conf = float(pred.get("confidence", 0.0))
                nx = float(pred.get("x", 0.0)) / frame.shape[1]
                ny = float(pred.get("y", 0.0)) / frame.shape[0]
                cand = TrackPoint(frame_index, 0.0, nx, ny, conf, self.name)
                if best is None or conf > best.confidence:
                    best = cand
            return best
        except Exception:
            return None


def _ema(points: list[TrackPoint], alpha: float = 0.35) -> list[TrackPoint]:
    if not points:
        return points
    out: list[TrackPoint] = []
    px, py = points[0].nx, points[0].ny
    for p in points:
        px = alpha * p.nx + (1 - alpha) * px
        py = alpha * p.ny + (1 - alpha) * py
        out.append(TrackPoint(p.frame_index, p.timestamp, float(px), float(py), p.confidence, p.source))
    return out


def _path_to_svg(points: list[TrackPoint]) -> str:
    if not points:
        return ""
    cmds = [f"M {points[0].nx:.4f} {points[0].ny:.4f}"]
    cmds.extend(f"L {p.nx:.4f} {p.ny:.4f}" for p in points[1:])
    return " ".join(cmds)


def _polyfit_ball(points: list[TrackPoint]) -> list[TrackPoint]:
    if len(points) < 3:
        return points
    xs = np.array([p.nx for p in points])
    ys = np.array([p.ny for p in points])
    coeffs = np.polyfit(xs, ys, deg=2)
    ys_fit = np.polyval(coeffs, xs)
    return [TrackPoint(p.frame_index, p.timestamp, float(p.nx), float(y), p.confidence, p.source) for p, y in zip(points, ys_fit)]


def _motion_ball_fallback(frames: Iterable[tuple[int, np.ndarray]], fps: float) -> list[TrackPoint]:
    iterator = iter(frames)
    try:
        prev_idx, prev_frame = next(iterator)
    except StopIteration:
        return []
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    track: list[TrackPoint] = []
    for frame_idx, frame in iterator:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, th = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(th)
        best = None
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 2 or area > 40:
                continue
            cx, cy = centroids[label]
            if best is None or area > best[0]:
                best = (area, cx, cy)
        if best:
            _, cx, cy = best
            track.append(TrackPoint(frame_idx, frame_idx / max(fps, 1e-6), float(cx / frame.shape[1]), float(cy / frame.shape[0]), 0.25, "opencv_motion_fallback"))
        prev_gray = gray
        prev_idx = frame_idx
    return track


def _to_dict_points(points: list[TrackPoint]) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": p.frame_index,
            "timestamp": round(float(p.timestamp), 4),
            "nx": round(float(p.nx), 6),
            "ny": round(float(p.ny), 6),
            "confidence": round(float(p.confidence), 4),
            "source": p.source,
        }
        for p in points
    ]


def _infer_impact(club: list[TrackPoint], hands: list[TrackPoint], balls: list[TrackPoint]) -> int:
    candidates: list[tuple[int, float]] = []
    for seq, w in ((club, 1.0), (hands, 0.7), (balls, 0.9)):
        for i in range(1, len(seq)):
            d = math.hypot(seq[i].nx - seq[i - 1].nx, seq[i].ny - seq[i - 1].ny)
            candidates.append((seq[i].frame_index, d * w))
    if not candidates:
        return 0
    return int(max(candidates, key=lambda x: x[1])[0])


def _compute_metrics(
    fps: float,
    impact_idx: int,
    hands: list[TrackPoint],
    club: list[TrackPoint],
    ball: list[TrackPoint],
    poses: list[dict[str, Any]],
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    px_to_m = float((calibration or {}).get("px_to_m", 0.0065))

    def top_speed(points: list[TrackPoint]) -> float:
        s = 0.0
        for i in range(1, len(points)):
            d = math.hypot(points[i].nx - points[i - 1].nx, points[i].ny - points[i - 1].ny)
            s = max(s, d)
        return s

    speed_factor = fps * max(poses[0].get("width", 1), 1) * px_to_m * 2.23694 if poses else 0.0
    club_mph = top_speed(club) * speed_factor
    ball_mph = top_speed(ball) * speed_factor * 1.35
    launch = 0.0
    if len(ball) >= 2:
        a = ball[0]
        b = ball[min(4, len(ball) - 1)]
        launch = abs(math.degrees(math.atan2((a.ny - b.ny), max(1e-4, (b.nx - a.nx)))))
    apex = max([1.0 - p.ny for p in ball], default=0.0) * 40
    carry = ball_mph * 1.45 + launch * 1.6
    curve = ((ball[-1].nx - ball[0].nx) * 80) if len(ball) >= 2 else 0.0

    top_idx = min(range(len(hands)), key=lambda i: hands[i].ny) if hands else 0
    impact_ts = impact_idx / max(fps, 1e-6)
    top_ts = hands[top_idx].timestamp if hands else impact_ts * 0.6
    tempo = max(top_ts, 0.01) / max(impact_ts - top_ts, 0.01)

    head_stability = 80.0
    shoulder_turn = 0.0
    hip_turn = 0.0
    if poses:
        heads = [p["landmarks"].get("head") for p in poses if p["landmarks"].get("head")]
        if heads:
            ys = np.array([h["y"] for h in heads])
            head_stability = max(0.0, 100.0 - float(np.std(ys) * 550.0))
        top_pose = poses[min(top_idx, len(poses) - 1)]
        ls = top_pose["landmarks"].get("left_shoulder")
        rs = top_pose["landmarks"].get("right_shoulder")
        lh = top_pose["landmarks"].get("left_hip")
        rh = top_pose["landmarks"].get("right_hip")
        if ls and rs:
            shoulder_turn = abs(math.degrees(math.atan2(rs["z"] - ls["z"], rs["x"] - ls["x"])))
        if lh and rh:
            hip_turn = abs(math.degrees(math.atan2(rh["z"] - lh["z"], rh["x"] - lh["x"])))

    confidence = float(np.clip(np.mean([p.confidence for p in (club + ball + hands)]) if (club or ball or hands) else 0.0, 0.0, 1.0))
    score_path = max(0.0, 100.0 - float(np.std([p.ny for p in club]) * 550.0)) if club else 45.0
    score_tempo = max(0.0, 100.0 - abs(tempo - 3.0) * 28.0)
    score_body = float(np.clip((head_stability + (100 - abs(shoulder_turn - 85)) + (100 - abs(hip_turn - 42))) / 3.0, 0, 100))
    score_impact = max(0.0, 100.0 - abs(curve) * 2.0)
    overall = 0.25 * score_path + 0.20 * score_tempo + 0.20 * score_body + 0.20 * score_impact + 0.15 * (confidence * 100)
    shape = "fade" if curve > 5 else "draw" if curve < -5 else "straight"

    return {
        "estimated_club_head_speed_mph": round(club_mph, 1),
        "estimated_ball_speed_mph": round(ball_mph, 1),
        "estimated_launch_angle_deg": round(launch, 1),
        "estimated_carry_yards": round(carry, 1),
        "estimated_apex_yards": round(apex, 1),
        "estimated_lateral_curve_yards": round(curve, 1),
        "tempo_ratio": f"{tempo:.1f}:1",
        "shoulder_turn_deg": round(shoulder_turn, 1),
        "hip_turn_deg": round(hip_turn, 1),
        "x_factor_deg": round(abs(shoulder_turn - hip_turn), 1),
        "head_stability_score": round(head_stability, 1),
        "swing_path": "inside_to_out" if curve < 0 else "outside_to_in",
        "shot_shape": shape,
        "confidence": round(confidence, 3),
        "estimated_note": "video-based estimate",
        "scores": {
            "path_smoothness_score": round(score_path, 1),
            "club_speed_score": round(min(100.0, club_mph), 1),
            "tempo_score": round(score_tempo, 1),
            "body_stability_score": round(score_body, 1),
            "impact_score": round(score_impact, 1),
            "overall_score": round(overall, 1),
        },
    }


def _extract_calibration(calibration_json: str | None) -> dict[str, Any] | None:
    if not calibration_json:
        return None
    try:
        return json.loads(calibration_json)
    except Exception:
        return None


def _strip_internal_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        hidden = {"source", "providers", "provider", "adapter", "debug", "stack", "traceback"}
        return {k: _strip_internal_fields(v) for k, v in obj.items() if k not in hidden}
    if isinstance(obj, list):
        return [_strip_internal_fields(x) for x in obj]
    return obj


def _append_optional_adapter(target: list[Any], adapter_factory: Any, unavailable: list[str]) -> None:
    try:
        target.append(adapter_factory())
    except RuntimeError as exc:
        unavailable.append(str(exc))
    except Exception as exc:
        unavailable.append(f"{adapter_factory}: {exc}")


def run_shot_tracer_reconstruct(
    video_path: str,
    front_view_path: str | None = None,
    side_view_path: str | None = None,
    calibration_json: str | None = None,
    mode: str = "single_video",
    include_debug: bool = False,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0.0
    calibration = _extract_calibration(calibration_json)

    unavailable_adapters: list[str] = []
    club_adapters: list[ClubDetectorAdapter] = []
    yolo_club_w = os.getenv("STELLAR_YOLO_CLUB_WEIGHTS", "")
    if yolo_club_w and Path(yolo_club_w).exists():
        _append_optional_adapter(club_adapters, lambda: YoloClubDetector(yolo_club_w), unavailable_adapters)
    rf_key = os.getenv("STELLAR_ROBOFLOW_API_KEY", "")
    rf_club = os.getenv("STELLAR_ROBOFLOW_CLUB_MODEL", "")
    if rf_key and rf_club:
        club_adapters.append(RoboflowClubDetector(rf_key, rf_club))
    club_adapters.append(PoseProxyClubDetector())

    ball_adapters: list[BallTrackerAdapter] = []
    track_url = os.getenv("STELLAR_TRACKNET_API_URL", "")
    if track_url:
        ball_adapters.append(TrackNetApiBallTracker(track_url, os.getenv("STELLAR_TRACKNET_API_KEY", "")))
    rf_ball = os.getenv("STELLAR_ROBOFLOW_BALL_MODEL", "")
    if rf_key and rf_ball:
        ball_adapters.append(RoboflowBallTracker(rf_key, rf_ball))
    yolo_ball_w = os.getenv("STELLAR_YOLO_BALL_WEIGHTS", "")
    if yolo_ball_w and Path(yolo_ball_w).exists():
        _append_optional_adapter(ball_adapters, lambda: YoloBallTracker(yolo_ball_w), unavailable_adapters)

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.45, min_tracking_confidence=0.45)
    poses: list[dict[str, Any]] = []
    hands_track: list[TrackPoint] = []
    club_track: list[TrackPoint] = []
    club_shaft_track: list[dict[str, Any]] = []
    ball_track: list[TrackPoint] = []
    fallback_window: Deque[tuple[int, np.ndarray]] = deque(maxlen=max(_FALLBACK_WINDOW_FRAMES, int(max(fps, 1) * 2)))

    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fallback_window.append((i, frame.copy()))
            ts = i / max(fps, 1e-6)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            pose_row = {"frame_index": i, "timestamp": ts, "width": width, "height": height, "landmarks": {}, "world_landmarks": []}
            if result.pose_landmarks:
                for name, idx in POSE_KEYS.items():
                    lm = result.pose_landmarks.landmark[idx]
                    pose_row["landmarks"][name] = {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(lm.visibility)}
                if result.pose_world_landmarks:
                    for name, idx in POSE_KEYS.items():
                        wlm = result.pose_world_landmarks.landmark[idx]
                        pose_row["world_landmarks"].append({"name": name, "x": float(wlm.x), "y": float(wlm.y), "z": float(wlm.z), "visibility": float(wlm.visibility)})
                lw = pose_row["landmarks"].get("left_wrist")
                rw = pose_row["landmarks"].get("right_wrist")
                if lw and rw:
                    hands_track.append(TrackPoint(i, ts, (lw["x"] + rw["x"]) * 0.5, (lw["y"] + rw["y"]) * 0.5, (lw["visibility"] + rw["visibility"]) * 0.5, "mediapipe_wrist"))
            poses.append(pose_row)

            club_p = None
            shaft_p = None
            for adapter in club_adapters:
                club_p, shaft_p = adapter.detect(frame, i, pose_row)
                if club_p is not None:
                    break
            if club_p:
                club_track.append(club_p)
            if shaft_p:
                club_shaft_track.append(shaft_p)

            ball_p = None
            for adapter in ball_adapters:
                ball_p = adapter.detect(frame, i)
                if ball_p:
                    ball_p.timestamp = ts
                    break
            if ball_p:
                ball_track.append(ball_p)
            i += 1
    finally:
        cap.release()
        pose.close()

    hands_track = _ema(hands_track)
    club_track = _ema(club_track)
    ball_track = _ema(ball_track, 0.5)
    impact_idx = _infer_impact(club_track, hands_track, ball_track)
    if len(ball_track) < 6 and fallback_window:
        motion = _motion_ball_fallback(fallback_window, fps)
        if motion:
            ball_track = motion
    ball_track = _polyfit_ball(ball_track)

    if hands_track:
        top_idx = min(range(len(hands_track)), key=lambda x: hands_track[x].ny)
        top_frame_idx = hands_track[top_idx].frame_index
    else:
        top_frame_idx = max(1, int(frame_count * 0.45))
    impact_idx = max(impact_idx, top_frame_idx + 1)
    metrics = _compute_metrics(fps, impact_idx, hands_track, club_track, ball_track, poses, calibration)

    def _pose_body_row(p: dict[str, Any]) -> dict[str, Any]:
        joints = []
        vis = []
        for name in POSE_KEYS:
            lm = p["landmarks"].get(name)
            if not lm:
                continue
            joints.append({"name": name, "nx": round(lm["x"], 6), "ny": round(lm["y"], 6), "visibility": round(lm["visibility"], 4)})
            vis.append(lm["visibility"])
        return {
            "frame_index": p["frame_index"],
            "timestamp": round(p["timestamp"], 4),
            "joints": joints,
            "confidence": round(float(np.mean(vis)) if vis else 0.0, 4),
            "source": "mediapipe_pose",
        }

    club_3d = []
    world_by_frame = {p["frame_index"]: p["world_landmarks"] for p in poses if p["world_landmarks"]}
    for cp in club_track:
        ws = world_by_frame.get(cp.frame_index, [])
        wrist = next((x for x in ws if x["name"] == "right_wrist"), None)
        club_3d.append({
            "frame_index": cp.frame_index,
            "timestamp": round(cp.timestamp, 4),
            "x": round(cp.nx - 0.5, 6),
            "y": round(0.5 - cp.ny, 6),
            "z": round(float((wrist or {}).get("z", 0.0)), 6),
            "confidence": round(cp.confidence, 4),
            "source": cp.source,
        })

    ball_3d = []
    for idx, bp in enumerate(ball_track):
        z = math.sin(min(1.0, idx / max(1, len(ball_track) - 1)) * math.pi) * 0.45
        ball_3d.append({
            "frame_index": bp.frame_index,
            "timestamp": round(bp.timestamp, 4),
            "x": round(bp.nx - 0.5, 6),
            "y": round(0.5 - bp.ny, 6),
            "z": round(z, 6),
            "confidence": round(bp.confidence, 4),
            "source": bp.source,
        })

    providers = {
        "pose": "mediapipe",
        "club_detector": club_track[0].source if club_track else "mediapipe_pose_proxy",
        "ball_tracker": ball_track[0].source if ball_track else "opencv_motion_fallback",
        "3d": "mediapipe_world_landmarks",
        "unavailable_optional_adapters": unavailable_adapters,
    }
    if mode == "3d_scene" and os.getenv("STELLAR_TRELLIS_API_URL"):
        providers["3d_asset"] = "trellis"
    if mode == "3d_scene" and os.getenv("STELLAR_POSTSHOT_API_URL"):
        providers["scene"] = "postshot"

    response = {
        "status": "ok",
        "engine": "stellar_shot_tracer_v1",
        "real_video_reconstruction": True,
        "video": {"fps": round(fps, 3), "duration_sec": round(duration, 3), "frame_count": frame_count, "width": width, "height": height},
        "phases": {
            "address_t": 0.0,
            "top_t": round(top_frame_idx / max(fps, 1e-6), 4),
            "impact_t": round(impact_idx / max(fps, 1e-6), 4),
            "finish_t": round(max(0, frame_count - 1) / max(fps, 1e-6), 4),
            "impact_frame_index": int(impact_idx),
        },
        "paths": {
            "body_2d": [_pose_body_row(p) for p in poses],
            "hands_2d": _to_dict_points(hands_track),
            "club_head_2d": _to_dict_points(club_track),
            "club_shaft_2d": club_shaft_track,
            "ball_flight_2d": _to_dict_points(ball_track),
            "skeleton_3d": [
                {
                    "frame_index": p["frame_index"],
                    "timestamp": round(p["timestamp"], 4),
                    "joints": p["world_landmarks"],
                    "source": "mediapipe_world_landmarks",
                    "confidence": round(float(np.mean([j["visibility"] for j in p["world_landmarks"]])) if p["world_landmarks"] else 0.0, 4),
                }
                for p in poses if p["world_landmarks"]
            ],
            "club_head_3d": club_3d,
            "ball_flight_3d": ball_3d,
            "svg": {"club_head": _path_to_svg(club_track), "ball_flight": _path_to_svg(ball_track)},
        },
        "metrics": metrics,
        "providers": providers,
        "display": {
            "data_label": "Video-based Estimate",
            "accuracy_notice": "Single-camera estimates are not radar measurements.",
        },
        "limitations": [
            "single-camera video estimates are not radar-accurate",
            "professional ball speed/spin requires dual-camera, high-speed video, calibration, or radar hardware",
            "all speed/distance values are estimated from video-based reconstruction",
        ],
        "mode": mode,
        "inputs": {"front_view": bool(front_view_path), "side_view": bool(side_view_path), "calibration": bool(calibration)},
    }
    if include_debug:
        return response
    return _strip_internal_fields(response)
