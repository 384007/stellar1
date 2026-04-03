/**
 * /api/pro/precheck  — lightweight Edge route (<1 s)
 *
 * 1. Validates the Pro JWT (signed token only — rejects local-* tokens)
 * 2. Optionally verifies is_pro against D1 for ground truth
 * 3. Returns { allowed, is_pro, modal_url, backend_url, modal_urls, backend_urls, token }
 *    (multi-endpoint lists for Pro v2 client failover; modal_url/backend_url stay first entries)
 */

import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { buildProV2BackendUrlList, buildProV2ModalUrlList } from "@/lib/pro-v2-endpoints";

export const runtime = "edge";

const BACKEND_FALLBACK = "https://stellar1-backend.onrender.com";

function cfClientCountry(request: NextRequest): string {
  return (
    request.headers.get("cf-ipcountry") ||
    request.headers.get("CF-IPCountry") ||
    ""
  )
    .trim()
    .toUpperCase();
}

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

function getDB() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).DB;
  } catch {
    return null;
  }
}

export async function POST(request: NextRequest) {
  const auth = request.headers.get("authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : auth;

  if (!token || token.startsWith("guest-")) {
    return NextResponse.json({ detail: "请先登录 Pro 账户", allowed: false }, { status: 401 });
  }

  // local-* tokens are from old client-side fallback — require re-login
  if (token.startsWith("local-")) {
    return NextResponse.json(
      { detail: "登录凭证已失效，请重新登录", allowed: false },
      { status: 401 }
    );
  }

  // Must be a real JWT
  if (!token.includes(".")) {
    return NextResponse.json({ detail: "无效凭证", allowed: false }, { status: 401 });
  }

  let jwtSecret = getCfEnv("JWT_SECRET");
  if (!jwtSecret) jwtSecret = process.env.JWT_SECRET || "";
  if (!jwtSecret) {
    return NextResponse.json({ detail: "服务器配置错误", allowed: false }, { status: 500 });
  }

  let isPro = false;
  let userId = "";

  try {
    const { payload } = await jwtVerify(token, new TextEncoder().encode(jwtSecret));
    isPro = !!(payload.is_pro);
    userId = (payload.user_id as string) || "";
  } catch {
    return NextResponse.json({ detail: "登录已过期，请重新登录", allowed: false }, { status: 401 });
  }

  // Verify against D1 for ground truth (if available)
  if (userId) {
    const db = getDB();
    if (db) {
      try {
        const user = await db
          .prepare("SELECT is_pro FROM users WHERE id = ?")
          .bind(userId)
          .first();
        if (user) {
          isPro = !!user.is_pro;
        }
      } catch {
        // D1 lookup failed — fall back to JWT claim
      }
    }
  }

  if (!isPro) {
    return NextResponse.json(
      { detail: "您不是 Pro 用户，请先升级", allowed: false, is_pro: false },
      { status: 403 }
    );
  }

  const mainlandCn = cfClientCountry(request) === "CN";
  const modalUrls = buildProV2ModalUrlList(getCfEnv, mainlandCn);
  const backendUrls = buildProV2BackendUrlList(getCfEnv, mainlandCn, BACKEND_FALLBACK);
  const modalUrl = modalUrls[0] || "";
  const backendUrl = backendUrls[0] || BACKEND_FALLBACK;

  return NextResponse.json({
    allowed: true,
    is_pro: true,
    backend_url: backendUrl,
    modal_url: modalUrl || undefined,
    backend_urls: backendUrls,
    modal_urls: modalUrls,
    ...(mainlandCn ? { network_hint: "cn" as const } : {}),
    token,
  });
}
