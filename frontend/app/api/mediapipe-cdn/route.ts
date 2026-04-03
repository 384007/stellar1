import { NextResponse } from "next/server";

export const runtime = "edge";

/**
 * Exposes MediaPipe self-host base when it could not be inlined at build time
 * (e.g. dashboard only provides Secrets under names without NEXT_PUBLIC_*).
 * The URL is not confidential — it is already sent to every browser that loads assets from it.
 */
export async function GET() {
  const base =
    (process.env.MEDIAPIPE_CDN_BASE ||
      process.env.NEXT_PUBLIC_MEDIAPIPE_CDN_BASE ||
      "")
      .trim()
      .replace(/\/+$/, "") || null;
  return NextResponse.json(
    { base },
    { headers: { "Cache-Control": "private, max-age=120" } },
  );
}
