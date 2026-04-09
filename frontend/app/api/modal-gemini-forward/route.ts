/**
 * Modal (Pro v3) → Cloudflare Edge → Gemini (same host×key loop as Lite / club-detect).
 *
 * When Modal has no GEMINI_PROXY_* but Pages Secrets do, Modal uses the same host×key loop as Lite
 * (no browser CN header required). Request body ``cn_network_hint: true`` uses Lite’s CN host order
 * (proxies first when GEMINI_PROXY_* exist on Pages).
 *
 * Auth: ``Authorization: Bearer <GEMINI_API_KEY>`` must match one of the configured Gemini keys on Pages.
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { getGeminiHosts, getGeminiKeys, shouldRetryNextGeminiKey } from "@/lib/gemini-proxy";

export const runtime = "edge";

function getCfEnv(key: string): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ((getRequestContext().env as any)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

type ForwardBody = {
  model?: string;
  contents?: unknown;
  generationConfig?: Record<string, unknown>;
  /** When true, use CN host order (proxies first when GEMINI_PROXY_* set on Pages). */
  cn_network_hint?: boolean;
};

export async function POST(request: NextRequest) {
  const auth = request.headers.get("authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
  const keys = getGeminiKeys(getCfEnv);
  if (!token || keys.length === 0 || !keys.includes(token)) {
    return NextResponse.json({ detail: "Unauthorized" }, { status: 401 });
  }

  let body: ForwardBody;
  try {
    body = (await request.json()) as ForwardBody;
  } catch {
    return NextResponse.json({ detail: "Invalid JSON" }, { status: 400 });
  }

  if (!body.contents || typeof body.contents !== "object") {
    return NextResponse.json({ detail: "contents required" }, { status: 400 });
  }

  const model =
    (typeof body.model === "string" && body.model.trim()) ||
    getCfEnv("GEMINI_MODEL") ||
    "gemini-2.5-flash-lite";
  const cn = !!body.cn_network_hint;
  const hosts = getGeminiHosts(getCfEnv, cn);
  const genCfg = body.generationConfig && typeof body.generationConfig === "object"
    ? body.generationConfig
    : {};

  const upstreamBody = JSON.stringify({
    contents: body.contents,
    generationConfig: genCfg,
  });

  const forwardTimeoutMs = Math.min(
    300_000,
    Math.max(30_000, Number(getCfEnv("STELLAR_CF_GEMINI_FORWARD_TIMEOUT_MS") || "240000") || 240_000),
  );

  for (const host of hosts) {
    for (let ki = 0; ki < keys.length; ki++) {
      const key = keys[ki]!;
      try {
        const res = await fetch(
          `${host}/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(key)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: upstreamBody,
            signal: AbortSignal.timeout(forwardTimeoutMs),
          },
        );
        if (res.ok) {
          const data = (await res.json()) as Record<string, unknown>;
          return NextResponse.json({ ...data, _stellar_key_slot: ki + 1 });
        }
        if (shouldRetryNextGeminiKey(res.status)) {
          continue;
        }
        break;
      } catch {
        break;
      }
    }
  }

  return NextResponse.json({ detail: "上游接口不可用，请稍后重试。" }, { status: 502 });
}
