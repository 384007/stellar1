import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { resolveProModalUpstreamBase } from "@/lib/server/prov3-upstream";
import { sanitizeProductJson } from "@/lib/chains/sanitize";

export const runtime = "edge";

/** Long-running MediaPipe / video generation on Modal */
const UPSTREAM_TIMEOUT_MS = 660_000;

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

/**
 * POST /api/plus/posture-video — forwards JSON to Modal ``POST /analyze/plus/posture-video`` only.
 */
export async function POST(request: NextRequest) {
  const cfRaw = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim();
  const base = resolveProModalUpstreamBase(getCfEnv, { clientCountryCode: cfRaw }).replace(/\/+$/, "");
  if (!base) {
    return NextResponse.json({ detail: "分析服务暂时不可用，请稍后重试。" }, { status: 503 });
  }

  let bodyText: string;
  try {
    bodyText = await request.text();
  } catch {
    return NextResponse.json({ detail: "Invalid body" }, { status: 400 });
  }

  const headers = new Headers({ "Content-Type": "application/json" });
  const auth = request.headers.get("authorization");
  if (auth) headers.set("Authorization", auth);
  if (cfRaw) headers.set("CF-IPCountry", cfRaw);

  const url = `${base}/analyze/plus/posture-video`;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers,
      body: bodyText,
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const text = await res.text();
    const ct = res.headers.get("content-type") || "";
    if (res.ok && ct.includes("application/json")) {
      try {
        const data = JSON.parse(text) as unknown;
        return NextResponse.json(sanitizeProductJson(data, "plus"), {
          status: res.status,
          headers: { "content-type": "application/json; charset=utf-8" },
        });
      } catch {
        /* raw fallback */
      }
    }
    return new NextResponse(text, {
      status: res.status,
      headers: { "content-type": ct || "application/json" },
    });
  } catch {
    return NextResponse.json({ detail: "姿态视频生成失败，请稍后重试。" }, { status: 502 });
  }
}
