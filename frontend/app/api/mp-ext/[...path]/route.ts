import { NextRequest, NextResponse } from "next/server";
import { MEDIAPIPE_TASKS_VISION_VERSION } from "@/lib/mediapipe-assets";

export const runtime = "edge";

const V = MEDIAPIPE_TASKS_VISION_VERSION;

const ALLOWED = new Set([
  "vision_bundle.mjs",
  "wasm/vision_wasm_internal.js",
  "wasm/vision_wasm_internal.wasm",
  "wasm/vision_wasm_module_internal.js",
  "wasm/vision_wasm_module_internal.wasm",
  "wasm/vision_wasm_nosimd_internal.js",
  "wasm/vision_wasm_nosimd_internal.wasm",
]);

function mime(rel: string): string {
  if (rel.endsWith(".wasm")) return "application/wasm";
  if (rel.endsWith(".mjs") || rel.endsWith(".js")) return "application/javascript; charset=utf-8";
  return "application/octet-stream";
}

function upstreamCandidates(relPath: string): string[] {
  return [
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${V}/${relPath}`,
    `https://registry.npmmirror.com/@mediapipe/tasks-vision/${V}/files/${relPath}`,
  ];
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  const relPath = path.join("/");
  if (!ALLOWED.has(relPath)) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  for (const url of upstreamCandidates(relPath)) {
    try {
      const upstream = await fetch(url, { redirect: "follow" });
      if (!upstream.ok) continue;
      return new NextResponse(upstream.body, {
        headers: {
          "Content-Type": mime(relPath),
          "Cache-Control": "public, max-age=604800, immutable",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch {
      /* try next */
    }
  }

  return NextResponse.json({ error: "upstream failed" }, { status: 502 });
}
