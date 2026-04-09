import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { resolveLiteAnalyzeUpstreamBase } from "@/lib/server/prov3-upstream";

export const runtime = "edge";

const UPSTREAM_TIMEOUT_MS = 18_000;

const FALLBACK_JSON = {
  club_type: "UNKNOWN",
  club_group: "IRON",
  confidence: 0,
  hand: "R" as const,
};

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

/**
 * Multipart frame in → FastAPI ``POST /analyze/club-detect`` on **Modal only** (see ``resolveLiteAnalyzeUpstreamBase``).
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const frame = formData.get("frame");
    if (!frame || !(frame instanceof Blob) || frame.size === 0) {
      return NextResponse.json(FALLBACK_JSON);
    }

    const cfRaw = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim();
    const base = resolveLiteAnalyzeUpstreamBase(getCfEnv, { clientCountryCode: cfRaw }).replace(/\/+$/, "");
    if (!base) {
      return NextResponse.json(FALLBACK_JSON);
    }

    const fd = new FormData();
    fd.append("frame", frame, "frame.jpg");

    const headers = new Headers();
    const auth = request.headers.get("authorization");
    if (auth) headers.set("Authorization", auth);
    if (cfRaw) headers.set("CF-IPCountry", cfRaw);

    const url = `${base}/analyze/club-detect`;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers,
        body: fd,
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
      if (res.ok) {
        const json = await res.json().catch(() => null);
        if (json && typeof json === "object") {
          return NextResponse.json({
            club_type: String((json as { club_type?: string }).club_type || "UNKNOWN"),
            club_group: String((json as { club_group?: string }).club_group || "IRON"),
            confidence: Number((json as { confidence?: number }).confidence) || 0,
            hand: (json as { hand?: string }).hand === "L" ? "L" : "R",
          });
        }
      }
    } catch {
      /* fall through */
    }

    return NextResponse.json(FALLBACK_JSON);
  } catch {
    return NextResponse.json(FALLBACK_JSON);
  }
}
