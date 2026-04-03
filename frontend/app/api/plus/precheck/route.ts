/**
 * /api/plus/precheck
 *
 * Lightweight Edge route:
 *  1. Validates the JWT
 *  2. Checks + increments the daily Plus usage counter in D1
 *  3. Returns { allowed, user_id, is_pro, remaining, backend_url, modal_url? }
 *
 * When CF-IPCountry is CN, modal_url is omitted (browser→Modal is often blocked);
 * network_hint: "cn" lets the client retry Render more aggressively.
 *
 * The client then calls the Render backend DIRECTLY with this token,
 * completely bypassing Cloudflare's 100-second timeout.
 */

import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";

const PLUS_FREE_DAILY_LIMIT = 3;
const BACKEND_FALLBACK = "https://stellar1-backend.onrender.com";

/** Cloudflare adds CF-IPCountry; browser direct to Modal is often blocked or flaky from mainland China. */
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
    return NextResponse.json({ detail: "请先登录后再使用 Plus 分析", allowed: false }, { status: 401 });
  }

  let userId = "unknown";
  let isPro = false;

  if (token.startsWith("local-")) {
    userId = token;
  } else if (token.includes(".")) {
    let jwtSecret = getCfEnv("JWT_SECRET");
    if (!jwtSecret) jwtSecret = process.env.JWT_SECRET || "";
    if (jwtSecret) {
      try {
        const { payload } = await jwtVerify(token, new TextEncoder().encode(jwtSecret));
        userId = (payload.user_id as string) || "unknown";
        isPro = !!(payload.is_pro);
      } catch {
        return NextResponse.json({ detail: "登录已过期，请重新登录", allowed: false }, { status: 401 });
      }
    }
  }

  const db = getDB();

  if (!isPro) {
    // Check and increment usage
    try {
      if (db) {
        await db.exec(
          "CREATE TABLE IF NOT EXISTS plus_usage (user_id TEXT NOT NULL, usage_date TEXT NOT NULL, count INTEGER DEFAULT 0, PRIMARY KEY (user_id, usage_date))"
        );

        const today = new Date().toISOString().slice(0, 10);
        const row = await db
          .prepare("SELECT count FROM plus_usage WHERE user_id = ? AND usage_date = ?")
          .bind(userId, today)
          .first();

        const used = (row?.count as number) ?? 0;

        if (used >= PLUS_FREE_DAILY_LIMIT) {
          return NextResponse.json({
            allowed: false,
            limit_reached: true,
            used,
            limit: PLUS_FREE_DAILY_LIMIT,
            detail: `今日 Plus 分析次数已达上限（${PLUS_FREE_DAILY_LIMIT}次/天）。升级 Pro 可获得无限次 Plus 分析。`,
          }, { status: 429 });
        }

        await db
          .prepare(
            "INSERT INTO plus_usage (user_id, usage_date, count) VALUES (?, ?, 1) ON CONFLICT(user_id, usage_date) DO UPDATE SET count = count + 1"
          )
          .bind(userId, today)
          .run();
      }
    } catch (e) {
      console.error("[plus/precheck] D1 error:", e);
      // fail-open: allow analysis even if D1 is unavailable
    }
  }

  const backendUrl = getCfEnv("NEXT_PUBLIC_BACKEND_URL") || BACKEND_FALLBACK;
  const modalUrl = getCfEnv("MODAL_BACKEND_URL") || process.env.MODAL_BACKEND_URL || "https://dytsui--stellar-ai-fastapi-app.modal.run";
  const mainlandCn = cfClientCountry(request) === "CN";

  return NextResponse.json({
    allowed: true,
    user_id: userId,
    is_pro: isPro,
    remaining: isPro ? -1 : PLUS_FREE_DAILY_LIMIT,
    backend_url: backendUrl,
    ...(mainlandCn ? {} : { modal_url: modalUrl || undefined }),
    ...(mainlandCn ? { network_hint: "cn" as const } : {}),
    token,
  });
}
