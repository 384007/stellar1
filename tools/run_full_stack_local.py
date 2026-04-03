#!/usr/bin/env python3
"""
Local full stack: pose → kinematic phases → smart keyframes → (optional) Gemini Plus.

Usage (from repo root or anywhere):
  cd backend && PYTHONPATH=. python ../tools/run_full_stack_local.py /path/to/video.mp4

Loads backend/.env for GEMINI_API_KEY, GEMINI_HTTPS_PROXY, etc.
Exit 0 if pose+keyframes OK; exit 3 if Gemini required (--require-gemini) but AI failed.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="MP4/MOV path")
    parser.add_argument("--max-poses", type=int, default=45)
    parser.add_argument(
        "--require-gemini",
        action="store_true",
        help="Exit non-zero if GEMINI_API_KEY set but analyze_swing_plus does not return gemini provider",
    )
    args = parser.parse_args()

    vid = Path(args.video_path).resolve()
    if not vid.is_file():
        print(f"ERROR: not a file: {vid}", file=sys.stderr)
        return 2

    backend = Path(__file__).resolve().parent.parent / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))

    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

    try:
        from dotenv import load_dotenv

        load_dotenv(backend / ".env")
    except Exception:
        pass

    import cv2

    from services.pose_service import extract_poses_from_video
    from services.swing_flow_utils import detect_swing_phases, get_phase_keyframes
    from services.keyframe_service import extract_keyframes_smart, keyframes_to_ai_images
    from services.gemini_service import analyze_swing_plus

    cap = cv2.VideoCapture(str(vid))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"video: frames={nframes} fps={fps:.3f}")

    print("1) extract_poses_from_video …")
    poses, _pose_bundle = extract_poses_from_video(
        str(vid), max_frames=args.max_poses, include_images=False, apply_smoothing=False
    )
    print(f"   poses={len(poses)} frame_index {poses[0]['frame_index'] if poses else '—'}..{poses[-1]['frame_index'] if poses else '—'}")
    if not poses:
        print("ERROR: no poses", file=sys.stderr)
        return 4

    print("2) detect_swing_phases + get_phase_keyframes …")
    swing = detect_swing_phases(poses)
    pk = dict(get_phase_keyframes(swing, poses))
    print(f"   phase_keyframes={pk}")

    print("3) extract_keyframes_smart …")
    pk_mut = dict(pk)
    kfs, meta = extract_keyframes_smart(str(vid), poses, swing, pk_mut, 320)
    print(
        f"   keyframes={len(kfs)} near_duplicates={meta.get('near_duplicates')} "
        f"time_close={meta.get('time_too_close')} all_passed={meta.get('all_passed')}"
    )
    prev = -1
    for kf in kfs:
        fi = int(kf.get("frame_index", -1))
        ok = fi >= prev
        print(f"   {kf['phase']:<16} fi={fi} mono={ok}")
        if not ok:
            print("ERROR: keyframe frame_index not non-decreasing", file=sys.stderr)
            return 5
        prev = fi

    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        print("4) Gemini: GEMINI_API_KEY missing — skip AI (set key or use GEMINI_HTTPS_PROXY + key for geo).")
        return 0

    print("4) analyze_swing_plus (Gemini) …")

    async def _ai():
        imgs = keyframes_to_ai_images(kfs)
        mid = len(poses) // 2
        rep = poses[mid]
        return await analyze_swing_plus(
            pose_data={
                "angles": rep["angles"],
                "all_frame_angles": [p["angles"] for p in poses],
                "frame_count": len(poses),
            },
            keyframe_images=imgs,
            region="global",
        )

    result = asyncio.run(_ai())
    prov = (result.get("ai_provider") or "").strip().lower()
    print(f"   ai_provider={prov!r} total_score={result.get('total_score')}")
    summ = (result.get("summary_zh") or result.get("summary") or "")[:120]
    if summ:
        print(f"   summary: {summ!r}…")

    if args.require_gemini and prov in ("", "none"):
        print("ERROR: --require-gemini but Gemini did not succeed (check proxy / region / quota).", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
