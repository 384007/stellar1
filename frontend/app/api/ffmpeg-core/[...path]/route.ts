import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

const CORE_VER = "0.12.10";

const ALLOWED = new Set(["ffmpeg-core.js", "ffmpeg-core.wasm"]);

function mime(name: string): string {
  if (name.endsWith(".wasm")) return "application/wasm";
  if (name.endsWith(".js")) return "application/javascript; charset=utf-8";
  return "application/octet-stream";
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  const name = path.join("/");
  if (!ALLOWED.has(name)) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  const url = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/umd/${name}`;
  try {
    const upstream = await fetch(url, { redirect: "follow" });
    if (!upstream.ok) {
      return NextResponse.json({ error: "upstream" }, { status: upstream.status });
    }
    return new NextResponse(upstream.body, {
      headers: {
        "Content-Type": mime(name),
        "Cache-Control": "public, max-age=604800, immutable",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch {
    return NextResponse.json({ error: "fetch failed" }, { status: 502 });
  }
}
