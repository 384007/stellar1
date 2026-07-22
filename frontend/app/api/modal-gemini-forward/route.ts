/**
 * Legacy Modal -> Gemini forward route.
 *
 * Disabled: Modal now calls NVIDIA/video-capable AI providers directly with Modal secrets.
 */

import { NextResponse } from "next/server";

export const runtime = "edge";

export async function POST() {
  return NextResponse.json(
    { detail: "Gemini forward is disabled; Modal uses NVIDIA video AI keys directly." },
    { status: 410 },
  );
}
