import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

/**
 * CF Edge proxy for MediaPipe static assets (bundle + wasm).
 *
 * Why this exists:
 *   R2's pub-xxx.r2.dev public URL is reachable in China via browser but
 *   unreliable for JavaScript fetch() of large binary files (wasm / models).
 *   This route runs on Cloudflare's edge (including Chinese PoPs), fetches
 *   the file from R2 via CF's internal backbone, and streams it back — so
 *   Chinese users get stable, fast delivery without direct R2 exposure.
 *
 * Handles:
 *   /api/mp-assets/vision_bundle.mjs
 *   /api/mp-assets/wasm/vision_wasm_internal.js
 *   /api/mp-assets/wasm/vision_wasm_internal.wasm
 *   /api/mp-assets/wasm/vision_wasm_module_internal.js
 *   /api/mp-assets/wasm/vision_wasm_module_internal.wasm
 *   /api/mp-assets/wasm/vision_wasm_nosimd_internal.js
 *   /api/mp-assets/wasm/vision_wasm_nosimd_internal.wasm
 */

const ALLOWED: Set<string> = new Set([
  "vision_bundle.mjs",
  "wasm/vision_wasm_internal.js",
  "wasm/vision_wasm_internal.wasm",
  "wasm/vision_wasm_module_internal.js",
  "wasm/vision_wasm_module_internal.wasm",
  "wasm/vision_wasm_nosimd_internal.js",
  "wasm/vision_wasm_nosimd_internal.wasm",
]);

function mime(path: string): string {
  if (path.endsWith(".wasm")) return "application/wasm";
  if (path.endsWith(".mjs") || path.endsWith(".js")) return "application/javascript";
  return "application/octet-stream";
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  const filePath = path.join("/");

  if (!ALLOWED.has(filePath)) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  const base = (process.env.MEDIAPIPE_CDN_BASE || "").trim().replace(/\/+$/, "");

  if (!base) {
    return NextResponse.json({ error: "R2 not configured" }, { status: 503 });
  }

  try {
    const upstream = await fetch(`${base}/${filePath}`);
    if (!upstream.ok) {
      return NextResponse.json(
        { error: `upstream ${upstream.status}` },
        { status: upstream.status },
      );
    }
    return new NextResponse(upstream.body, {
      headers: {
        "Content-Type": mime(filePath),
        "Cache-Control": "public, max-age=604800, immutable",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
