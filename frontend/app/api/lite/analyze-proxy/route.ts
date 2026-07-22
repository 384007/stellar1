import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";

import { createLiteAnalyzeEdgeSseStream } from "@/lib/lite-analyze-edge-sse";
import { LITE_ANALYZE_FETCH_TIMEOUT_MS } from "@/lib/lite-analyze-timeout";
import { resolveLiteAnalyzeUpstreamBase } from "@/lib/server/prov3-upstream";

export const runtime = "edge";

/**
 * CN / same-origin Lite path: multipart in → forward ``POST {ModalBase}/analyze/lite``.
 *
 * **Cold start:** returns ``text/event-stream`` immediately and sends ``: edge-wait-upstream`` comment
 * lines every few seconds while waiting for Modal ``fetch`` headers, then pipes upstream SSE or wraps
 * JSON / errors as a final ``data:`` envelope — same shape the client already parses via ``readLiteAnalyzeResult``.
 */

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

function trimLiteBase(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

async function requireAuth(request: NextRequest): Promise<NextResponse | null> {
  const authHeader = request.headers.get("authorization") || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : authHeader;

  if (!token) {
    return NextResponse.json({ detail: "请先登录后再使用分析功能" }, { status: 401 });
  }
  if (token.startsWith("guest-")) {
    return NextResponse.json(
      { detail: "游客模式已关闭，请注册或登录后使用分析功能" },
      { status: 403 },
    );
  }
  if (token.startsWith("local-")) return null;
  if (!token.includes(".")) {
    return NextResponse.json({ detail: "登录状态无效，请重新登录" }, { status: 401 });
  }

  let jwtSecret = "";
  try {
    jwtSecret = (getRequestContext().env as Record<string, string>).JWT_SECRET || "";
  } catch {
    /* not in CF context */
  }
  if (!jwtSecret) jwtSecret = process.env.JWT_SECRET || "";

  if (!jwtSecret) {
    console.error("[lite/analyze-proxy] JWT_SECRET not configured, skipping JWT verification");
    return null;
  }

  try {
    const secret = new TextEncoder().encode(jwtSecret);
    const { payload } = await jwtVerify(token, secret);
    if (payload.is_guest) {
      return NextResponse.json(
        { detail: "游客模式已关闭，请注册或登录后使用分析功能" },
        { status: 403 },
      );
    }
    return null;
  } catch {
    return NextResponse.json({ detail: "登录已过期，请重新登录" }, { status: 401 });
  }
}

export async function POST(request: NextRequest) {
  const authErr = await requireAuth(request);
  if (authErr) return authErr;

  const cfCountryHdr =
    (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim() || undefined;
  const base = trimLiteBase(
    resolveLiteAnalyzeUpstreamBase(getCfEnv, { clientCountryCode: cfCountryHdr }),
  );
  if (!base) {
    return NextResponse.json(
      {
        detail: "分析服务暂时不可用，请稍后重试。",
        code: "LITE_PROXY_NO_BACKEND",
      },
      { status: 503 },
    );
  }

  const idem =
    request.headers.get("x-stellar-idempotency-key") ||
    request.headers.get("X-Stellar-Idempotency-Key") ||
    "";
  if (!idem.trim()) {
    return NextResponse.json(
      { detail: "Missing idempotency key", code: "LITE_IDEMPOTENCY_KEY_REQUIRED" },
      { status: 400 },
    );
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ detail: "Invalid multipart body" }, { status: 400 });
  }

  const file = form.get("file");
  const requestId = form.get("request_id");
  if (!file || typeof file === "string" || !("arrayBuffer" in file)) {
    return NextResponse.json({ detail: "No file provided" }, { status: 400 });
  }
  const rid = typeof requestId === "string" ? requestId.trim() : "";
  if (!rid) {
    return NextResponse.json({ detail: "Missing request_id" }, { status: 400 });
  }

  const f = file as File;
  const upstreamHeaders: Record<string, string> = {
    "X-Stellar-Idempotency-Key": idem.trim(),
  };
  // Authentication is completed at Pages. Do not forward the product JWT to
  // Modal: its gateway can treat an arbitrary Bearer value as Modal auth.
  if (cfCountryHdr) upstreamHeaders["CF-IPCountry"] = cfCountryHdr;

  const url = `${base}/analyze/lite`;
  const fetchUpstream = () => {
    // FormData bodies are single-use in some Edge runtimes; rebuild for a retry.
    const out = new FormData();
    out.append("file", f, f.name || "video.mp4");
    out.append("request_id", rid);
    return fetch(url, {
      method: "POST",
      headers: upstreamHeaders,
      body: out,
      signal: AbortSignal.timeout(LITE_ANALYZE_FETCH_TIMEOUT_MS),
    });
  };
  const fetchWithColdStartRetry = async () => {
    const first = await fetchUpstream();
    if (first.status !== 524) return first;

    // Modal may still be assigning a scale-to-zero container when its first
    // request reaches the Modal gateway. Keep the browser SSE alive and retry
    // once after the gateway's 524 rather than surfacing a false failure.
    console.warn(`[lite/analyze-proxy] upstream 524; retrying cold start request_id=${rid}`);
    await new Promise((resolve) => setTimeout(resolve, 3_000));
    return fetchUpstream();
  };
  const stream = createLiteAnalyzeEdgeSseStream(fetchWithColdStartRetry);

  return new NextResponse(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
      "x-stellar-lite-stream": "1",
    },
  });
}
