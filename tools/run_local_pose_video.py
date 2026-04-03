#!/usr/bin/env python3
"""
Smoke-test pose + smart keyframes on a local video (no Gemini).
Run from repo root or anywhere with:
  cd backend && PYTHONPATH=. python ../tools/run_local_pose_video.py /path/to/video.mp4

Requires Python 3.11 + backend/requirements.txt (MediaPipe).

Full stack (Gemini phases + Plus/Pro AI + fusion): run the backend with that same env, then:
  python3 tools/run_api_e2e.py /path/to/video.mp4
See tools/run_api_e2e.py for STELLAR_API_BASE and --save-json-dir.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to MP4/MOV")
    parser.add_argument("--max-poses", type=int, default=45)
    args = parser.parse_args()
    path = os.path.abspath(args.video_path)
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    backend_root = os.path.join(os.path.dirname(__file__), "..", "backend")
    backend_root = os.path.abspath(backend_root)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

    import cv2

    from services.pose_service import extract_poses_from_video
    from services.swing_flow_utils import detect_swing_phases, get_phase_keyframes
    from services.keyframe_service import extract_keyframes_smart
    from services.video_utils import read_frame_pose_pipeline, get_video_rotation

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("ERROR: cannot open video", file=sys.stderr)
        return 3
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"OpenCV: frames={nframes} fps={fps:.3f}")

    print("Extracting poses...")
    poses, _pose_bundle = extract_poses_from_video(
        path, max_frames=args.max_poses, include_images=False, apply_smoothing=False
    )
    print(f"poses: {len(poses)}")
    if not poses:
        print("ERROR: no poses", file=sys.stderr)
        return 4

    fi0 = poses[0]["frame_index"]
    fi1 = poses[-1]["frame_index"]
    print(f"pose frame_index range: {fi0} .. {fi1}")

    swing = detect_swing_phases(poses)
    pk = get_phase_keyframes(swing, poses)
    print(f"phase_keyframes: {pk}")

    rot = get_video_rotation(path)
    cap2 = cv2.VideoCapture(path)
    test_idx = int(poses[len(poses) // 2]["frame_index"])
    a = read_frame_pose_pipeline(cap2, test_idx, rot)
    cap2.release()
    if a is None:
        print(f"ERROR: read_frame_pose_pipeline failed at {test_idx}", file=sys.stderr)
        return 5

    print("Smart keyframes...")
    pk_mut = dict(pk)
    kfs, meta = extract_keyframes_smart(path, poses, swing, pk_mut, 320)
    print(f"keyframes: {len(kfs)} near_duplicates={meta.get('near_duplicates')} time_close={meta.get('time_too_close')}")
    prev = -1
    for kf in kfs:
        fi = int(kf.get("frame_index", -1))
        ok = fi >= prev
        print(f"  {kf['phase']:16} frame_index={fi} pose_idx={kf.get('source_pose_idx')} monotonic={ok}")
        if not ok:
            print("ERROR: keyframe frame_index not non-decreasing", file=sys.stderr)
            return 6
        prev = fi

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
