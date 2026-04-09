import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";
import { resolveLiteAnalyzeUpstreamBase } from "@/lib/prov3-endpoints";
import { sanitizeProductJson } from "@/lib/chains/sanitize";

export const runtime = "edge";

const UPSTREAM_TIMEOUT_MS = 12_000;

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
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
    console.error("[analyze/recalculate] JWT_SECRET not configured, skipping JWT verification");
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

/**
 * JSON in/out → FastAPI ``POST /analyze/recalculate`` on **Modal only**.
 */
export async function POST(request: NextRequest) {
  const authErr = await requireAuth(request);
  if (authErr) return authErr;

  let bodyText: string;
  try {
    bodyText = await request.text();
  } catch {
    return NextResponse.json({ detail: "Invalid body" }, { status: 400 });
  }
  if (!bodyText.trim()) {
    return NextResponse.json({ detail: "Empty body" }, { status: 400 });
  }

  const cfRaw = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim();
  const base = resolveLiteAnalyzeUpstreamBase(getCfEnv, { clientCountryCode: cfRaw }).replace(/\/+$/, "");
  if (!base) {
    return NextResponse.json({ detail: "分析服务暂时不可用，请稍后重试。" }, { status: 503 });
  }

  const headers = new Headers({
    "Content-Type": "application/json",
  });
  const auth = request.headers.get("authorization");
  if (auth) headers.set("Authorization", auth);
  if (cfRaw) headers.set("CF-IPCountry", cfRaw);

  const url = `${base}/analyze/recalculate`;
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
        return NextResponse.json(sanitizeProductJson(data, "analysis"), {
          status: res.status,
          headers: { "content-type": "application/json; charset=utf-8" },
        });
      } catch {
        /* fall through to raw body */
      }
    }
    return new NextResponse(text, {
      status: res.status,
      headers: { "content-type": ct || "application/json" },
    });
  } catch {
    return NextResponse.json({ detail: "重新计算失败，请稍后重试。" }, { status: 502 });
  }
}
