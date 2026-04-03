import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") || "10";

  try {
    const res = await fetch(`${BACKEND}/news?limit=${limit}&_=${Date.now()}`, {
      cache: "no-store",
    });

    if (!res.ok) throw new Error("Backend news fetch failed");

    const data = await res.json();
    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
      },
    });
  } catch {
    return NextResponse.json(
      { news: [], source: "error" },
      { status: 502 }
    );
  }
}
