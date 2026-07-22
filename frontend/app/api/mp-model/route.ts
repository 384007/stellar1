import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

/**
 * Proxy for MediaPipe model files.
 *
 * Fetch order (per model):
 *   1. R2 (via env MEDIAPIPE_CDN_BASE, server-only)
 *      → CF edge fetches from R2 via Cloudflare backbone, serves via Chinese PoP.
 *   2. Google Storage — fallback if R2 is not configured or fails.
 *      (CF edge node in HK/SG can reach Google; Chinese browsers cannot.)
 *
 * Cache-Control: 7 days so only the first request per PoP hits the origin.
 */

const MODEL_PATHS: Record<string, string> = {
  pose_landmarker_lite: "models/pose_landmarker_lite.task",
  pose_landmarker_full: "models/pose_landmarker_full.task",
};

const GOOGLE_FALLBACK: Record<string, string> = {
  pose_landmarker_lite:
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  pose_landmarker_full:
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
};

export async function GET(req: NextRequest) {
  const model = req.nextUrl.searchParams.get("m");
  if (!model || !MODEL_PATHS[model]) {
    return NextResponse.json({ error: "unknown model" }, { status: 400 });
  }

  const r2Base = (process.env.MEDIAPIPE_CDN_BASE || "").trim().replace(/\/+$/, "");

  // Build candidate list: R2 first (if configured), Google Storage as fallback.
  const candidates: string[] = [];
  if (r2Base) candidates.push(`${r2Base}/${MODEL_PATHS[model]}`);
  candidates.push(GOOGLE_FALLBACK[model]);

  for (const url of candidates) {
    try {
      const upstream = await fetch(url);
      if (!upstream.ok) continue;
      return new NextResponse(upstream.body, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Cache-Control": "public, max-age=604800, immutable",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch { /* try next */ }
  }

  return NextResponse.json({ error: "all upstream sources failed" }, { status: 502 });
}
