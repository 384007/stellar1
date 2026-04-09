import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";

import { LITE_ANALYZE_FETCH_TIMEOUT_MS } from "@/lib/lite-analyze-timeout";
import { resolveLiteAnalyzeUpstreamBase } from "@/lib/server/prov3-upstream";
import { sanitizeLiteSseStream } from "@/lib/chains";
import { sanitizeProductJson } from "@/lib/chains/sanitize";

export const runtime = "edge";

/**
 * CN / same-origin Lite path: multipart in → forward ``POST {ModalBase}/analyze/lite`` (main Modal by default).
 * No Gemini/Qwen here — only JWT gate + transparent proxy (idempotency + request_id + Authorization + CF-IPCountry).
 *
 * **524:** Cloudflare’s limit on the *client → Pages* request (often ~100s) can fire before upstream finishes;
 * the browser sees HTTP 524 even though upstream ``AbortSignal`` allows 1h. Mitigations: raise CF/proxy
 * timeouts or async Lite (not implemented here).
 *
 * Default upstream is **main Pro Modal** (same as ``MODAL_BACKEND_URL`` / fallbacks).
 * Optional override: ``LITE_BACKEND_URL`` = dedicated Lite-only origin, no trailing slash.
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

  const out = new FormData();
  const f = file as File;
  out.append("file", f, f.name || "video.mp4");
  out.append("request_id", rid);

  const upstreamHeaders: Record<string, string> = {
    "X-Stellar-Idempotency-Key": idem.trim(),
  };
  const authz = request.headers.get("authorization");
  if (authz) upstreamHeaders.Authorization = authz;
  if (cfCountryHdr) upstreamHeaders["CF-IPCountry"] = cfCountryHdr;

  const url = `${base}/analyze/lite`;
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: "POST",
      headers: upstreamHeaders,
      body: out,
      signal: AbortSignal.timeout(LITE_ANALYZE_FETCH_TIMEOUT_MS),
    });
  } catch {
    return NextResponse.json(
      { detail: "分析服务连接失败，请稍后重试。", code: "LITE_PROXY_UPSTREAM_FAILED" },
      { status: 502 },
    );
  }

  const ct = upstream.headers.get("content-type") || "application/json; charset=utf-8";
  if (upstream.ok && ct.includes("text/event-stream") && upstream.body) {
    const outBody = sanitizeLiteSseStream(upstream.body, "analysis");
    return new NextResponse(outBody, {
      status: upstream.status,
      headers: {
        "content-type": ct,
        "cache-control": upstream.headers.get("cache-control") || "no-cache",
        "x-accel-buffering": "no",
      },
    });
  }

  const buf = await upstream.arrayBuffer();
  if (upstream.ok && ct.includes("application/json")) {
    try {
      const text = new TextDecoder().decode(buf);
      const data = JSON.parse(text) as unknown;
      return NextResponse.json(sanitizeProductJson(data, "analysis"), {
        status: upstream.status,
        headers: { "content-type": "application/json; charset=utf-8" },
      });
    } catch {
      /* fall through */
    }
  }
  return new NextResponse(buf, {
    status: upstream.status,
    headers: { "content-type": ct },
  });
}
