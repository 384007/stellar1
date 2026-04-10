import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { resolveLiteAnalyzeUpstreamBase } from "@/lib/server/prov3-upstream";

export const runtime = "edge";

const UPSTREAM_TIMEOUT_MS = 18_000;
/** Three in-process vision calls — allow extra time vs single-frame. */
const UPSTREAM_BATCH_TIMEOUT_MS = 90_000;

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

function normalizeClubJson(json: unknown): Record<string, string | number> {
  if (!json || typeof json !== "object") return { ...FALLBACK_JSON };
  const j = json as { club_type?: string; club_group?: string; confidence?: number; hand?: string };
  return {
    club_type: String(j.club_type || "UNKNOWN"),
    club_group: String(j.club_group || "IRON"),
    confidence: Number(j.confidence) || 0,
    hand: j.hand === "L" ? "L" : "R",
  };
}

/**
 * Lite Modal worker (``LITE_BACKEND_URL`` / default lite origin).
 * Main Lite video flow should use ``/analyze/lite`` only (one run); this route is for standalone club checks.
 * - Single ``frame`` → ``POST /analyze/club-detect``
 * - ``frame_0`` + ``frame_1`` + ``frame_2`` → ``POST /analyze/club-detect-batch`` (one HTTP)
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const f0 = formData.get("frame_0");
    const f1 = formData.get("frame_1");
    const f2 = formData.get("frame_2");
    const single = formData.get("frame");

    const batch =
      f0 instanceof Blob &&
      f0.size > 0 &&
      f1 instanceof Blob &&
      f1.size > 0 &&
      f2 instanceof Blob &&
      f2.size > 0;

    const cfRaw = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim();
    const base = resolveLiteAnalyzeUpstreamBase(getCfEnv, { clientCountryCode: cfRaw }).replace(/\/+$/, "");
    if (!base) {
      return NextResponse.json(FALLBACK_JSON);
    }

    const headers = new Headers();
    const auth = request.headers.get("authorization");
    if (auth) headers.set("Authorization", auth);
    if (cfRaw) headers.set("CF-IPCountry", cfRaw);

    if (batch) {
      const fd = new FormData();
      fd.append("frame_0", f0, "frame_0.jpg");
      fd.append("frame_1", f1, "frame_1.jpg");
      fd.append("frame_2", f2, "frame_2.jpg");
      const url = `${base}/analyze/club-detect-batch`;
      try {
        const res = await fetch(url, {
          method: "POST",
          headers,
          body: fd,
          signal: AbortSignal.timeout(UPSTREAM_BATCH_TIMEOUT_MS),
        });
        if (res.ok) {
          const json = await res.json().catch(() => null);
          return NextResponse.json(normalizeClubJson(json));
        }
      } catch {
        /* fall through */
      }
      return NextResponse.json(FALLBACK_JSON);
    }

    const frame = single instanceof Blob && single.size > 0 ? single : null;
    if (!frame) {
      return NextResponse.json(FALLBACK_JSON);
    }

    const fd = new FormData();
    fd.append("frame", frame, "frame.jpg");

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
        return NextResponse.json(normalizeClubJson(json));
      }
    } catch {
      /* fall through */
    }

    return NextResponse.json(FALLBACK_JSON);
  } catch {
    return NextResponse.json(FALLBACK_JSON);
  }
}
