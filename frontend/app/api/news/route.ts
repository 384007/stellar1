import { NextResponse } from "next/server";
import { sanitizeProductJson } from "@/lib/chains/sanitize";

export const runtime = "edge";
export const dynamic = "force-dynamic";

function newsBackendBase(): string {
  const raw =
    process.env.BACKEND_URL ||
    process.env.NEWS_BACKEND_URL ||
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    "http://localhost:8000";
  return raw.trim().replace(/\/+$/, "");
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") || "10";

  try {
    const res = await fetch(`${newsBackendBase()}/news?limit=${limit}&_=${Date.now()}`, {
      cache: "no-store",
    });

    if (!res.ok) throw new Error("Backend news fetch failed");

    const data = await res.json();
    return NextResponse.json(sanitizeProductJson(data, "generic"), {
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
