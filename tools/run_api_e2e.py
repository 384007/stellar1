#!/usr/bin/env python3
"""
End-to-end API check: multipart upload → /analyze/plus and /stellar-pro/analyze (Pro).

Uses the same HTTP surface as the app (Gemini phase + AI, smart keyframes, HUD, prediction;
Pro adds fusion when those modules succeed).

Prerequisites:
  - Backend running with Python 3.11 + backend/requirements.txt (MediaPipe).
  - GEMINI_API_KEY (or Vertex) set for the server process.
  - If Google returns "User location is not supported", use Singapore/Japan egress without changing Modal:
      export GEMINI_HTTPS_PROXY=http://127.0.0.1:7890   # or your SG/JP proxy URL
    (backend forces REST transport when a proxy is set.)
  - Or Vertex: GEMINI_BACKEND=vertex + VERTEX_AI_LOCATION=asia-southeast1 | asia-northeast1
  - Default base URL: http://127.0.0.1:10000 — override with STELLAR_API_BASE or --base-url.

Examples:
  cd backend && uvicorn main:app --host 127.0.0.1 --port 10000
  # other terminal:
  python3 tools/run_api_e2e.py /path/to/video.mp4
  python3 tools/run_api_e2e.py /path/to/video.mp4 --expect-gemini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _summarize_plus_pro(data: dict[str, Any], label: str) -> list[str]:
    lines: list[str] = [f"=== {label} ==="]
    lines.append(f"type={data.get('type')} ai_provider={data.get('ai_provider')}")
    vm = data.get("video_meta") or {}
    lines.append(
        f"video_meta: fps={vm.get('fps')} source_frame_count={vm.get('source_frame_count')} "
        f"poses={vm.get('total_pose_frames')} duration_s={vm.get('duration_s')}"
    )
    lines.append(f"phase_source={data.get('phase_source')!r}")
    pk = data.get("phase_keyframes") or {}
    lines.append(f"phase_keyframes: {pk}")
    kfs = data.get("keyframes") or []
    lines.append(f"keyframes count={len(kfs)}")
    prev = -1
    mono_ok = True
    phases_order: list[str] = []
    for kf in kfs:
        fi = int(kf.get("frame_index") or -1)
        phases_order.append(str(kf.get("phase")))
        if fi < prev:
            mono_ok = False
        prev = fi
    lines.append(f"keyframe phases order: {phases_order}")
    lines.append(f"keyframe frame_index monotonic non-decreasing: {mono_ok}")
    kv = data.get("keyframe_validation")
    if kv is not None:
        lines.append(f"keyframe_validation: {kv}")
    rel = data.get("analysis_reliability")
    if rel:
        lines.append(f"analysis_reliability.level={rel.get('level')}")
    pred = data.get("prediction") or {}
    lines.append(
        f"prediction: ball_speed={pred.get('ball_speed')} carry={pred.get('carry_distance_yards')}"
    )
    if label.upper().startswith("PRO") and data.get("type") == "pro":
        lines.append(
            f"fusion hints: fused_speed={pred.get('fused_speed')} club_type={pred.get('club_type')}"
        )
    sk = data.get("skeleton_data") or {}
    lines.append(f"skeleton_data.frames={sk.get('total_frames')}")
    # AI text sanity (short)
    qz = (data.get("quick_tip_zh") or "")[:80]
    if qz:
        lines.append(f"quick_tip_zh (prefix): {qz!r}…")
    summ = (data.get("summary_zh") or data.get("summary") or "")[:80]
    if summ:
        lines.append(f"summary (prefix): {summ!r}…")
    return lines


def _summarize_stellar_pro(data: dict[str, Any]) -> list[str]:
    lines: list[str] = ["=== STELLAR PRO (/stellar-pro/analyze) ==="]
    lines.append(f"status={data.get('status')} analysis_id={data.get('analysis_id')}")
    kfs = data.get("keyframes") or []
    lines.append(f"keyframes count={len(kfs)}")
    prev = -1
    mono_ok = True
    phases_order: list[str] = []
    for kf in kfs:
        fi = int(kf.get("source_frame_index") or kf.get("frame_index") or -1)
        phases_order.append(str(kf.get("phase")))
        if fi < prev:
            mono_ok = False
        prev = fi
    lines.append(f"keyframe phases order: {phases_order}")
    lines.append(f"source_frame_index monotonic: {mono_ok}")
    summ = (data.get("summary") or "")[:120]
    lines.append(f"summary (prefix): {summ!r}")
    lines.append(f"total_score={data.get('total_score')}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="POST video to Plus/Pro analyze APIs")
    parser.add_argument("video_path", help="Path to MP4/MOV")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("STELLAR_API_BASE", "http://127.0.0.1:10000").rstrip("/"),
        help="Backend origin (default STELLAR_API_BASE or http://127.0.0.1:10000)",
    )
    parser.add_argument("--plus-only", action="store_true")
    parser.add_argument("--pro-only", action="store_true")
    parser.add_argument(
        "--save-json-dir",
        default="",
        help="If set, write stripped responses (no base64 blobs) as JSON files here",
    )
    args = parser.parse_args()
    path = Path(args.video_path).resolve()
    if not path.is_file():
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        return 2

    try:
        import httpx
    except ImportError:
        print("ERROR: pip install httpx", file=sys.stderr)
        return 3

    base = args.base_url

    def strip_b64(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "image_base64" and isinstance(v, str):
                    out[k] = f"<len={len(v)}>"
                else:
                    out[k] = strip_b64(v)
            return out
        if isinstance(obj, list):
            return [strip_b64(x) for x in obj]
        return obj

    async def run() -> int:
        timeout = httpx.Timeout(300.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                h = await client.get(f"{base}/health")
                print(f"GET /health → {h.status_code}")
                if h.status_code == 200:
                    try:
                        hj = h.json()
                        print(f"  runtime={hj.get('runtime')} env={hj.get('env')}")
                    except Exception:
                        pass
            except httpx.RequestError as e:
                print(f"ERROR: cannot reach {base}: {e}", file=sys.stderr)
                return 4

            name = path.name
            data_bytes = path.read_bytes()

            async def post_plus() -> tuple[int, dict[str, Any] | None]:
                files = {"file": (name, data_bytes, "video/mp4")}
                url = f"{base}/analyze/plus"
                resp = await client.post(url, files=files)
                try:
                    body = resp.json()
                except Exception:
                    body = None
                return resp.status_code, body if isinstance(body, dict) else None

            async def post_stellar_pro() -> tuple[int, dict[str, Any] | None]:
                files = {"file": (name, data_bytes, "video/mp4")}
                url = f"{base}/stellar-pro/analyze"
                resp = await client.post(url, files=files)
                try:
                    body = resp.json()
                except Exception:
                    body = None
                return resp.status_code, body if isinstance(body, dict) else None

            rc = 0
            plus_body: dict[str, Any] | None = None
            pro_body: dict[str, Any] | None = None
            if not args.pro_only:
                code, body = await post_plus()
                print(f"POST /analyze/plus → HTTP {code}")
                if code != 200 or not body:
                    print(f"  body: {body!r}" if body else "", file=sys.stderr)
                    rc = 5 if rc == 0 else rc
                else:
                    plus_body = body
                    for line in _summarize_plus_pro(body, "PLUS"):
                        print(line)
                    if args.save_json_dir:
                        outd = Path(args.save_json_dir)
                        outd.mkdir(parents=True, exist_ok=True)
                        (outd / "plus_response.json").write_text(
                            json.dumps(strip_b64(body), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

            if not args.plus_only:
                code, body = await post_stellar_pro()
                print(f"POST /stellar-pro/analyze → HTTP {code}")
                if code != 200 or not body:
                    print(f"  body: {body!r}" if body else "", file=sys.stderr)
                    rc = 6 if rc == 0 else rc
                else:
                    pro_body = body
                    for line in _summarize_stellar_pro(body):
                        print(line)
                    if args.save_json_dir:
                        outd = Path(args.save_json_dir)
                        outd.mkdir(parents=True, exist_ok=True)
                        (outd / "pro_response.json").write_text(
                            json.dumps(strip_b64(body), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

            if args.expect_gemini:

                def _provider_failed(b: dict[str, Any] | None) -> bool:
                    if not b:
                        return True
                    p = str(b.get("ai_provider") or "").strip().lower()
                    return p in ("", "none")

                if not args.pro_only and _provider_failed(plus_body):
                    print(
                        "ERROR: --expect-gemini but Plus ai_provider is missing/none "
                        "(set GEMINI_HTTPS_PROXY to SG/JP egress, or Vertex in asia-southeast1, or QWEN_API_KEY)",
                        file=sys.stderr,
                    )
                    rc = 7 if rc == 0 else rc
                if not args.plus_only and pro_body is not None:
                    if not str(pro_body.get("summary") or "").strip():
                        print(
                            "ERROR: --expect-gemini but Pro summary is empty",
                            file=sys.stderr,
                        )
                        rc = 8 if rc == 0 else rc

            return rc

    import asyncio

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
