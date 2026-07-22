/**
 * /api/plus/precheck — legacy read-only check (optional).
 *
 * Plus 主流程已改为单次 ``POST /api/plus``（Edge → Modal）完成鉴权与配额；本路由保留以免旧客户端报错。
 * 不再返回 Render URL / token；不单独扣次（扣次在 ``/api/plus``）。
 */

import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";

const PLUS_FREE_DAILY_LIMIT = 3;

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
  let remaining = PLUS_FREE_DAILY_LIMIT;

  if (!isPro && db) {
    try {
      await db.exec(
        "CREATE TABLE IF NOT EXISTS plus_usage (user_id TEXT NOT NULL, usage_date TEXT NOT NULL, count INTEGER DEFAULT 0, PRIMARY KEY (user_id, usage_date))"
      );
      const today = new Date().toISOString().slice(0, 10);
      const row = await db
        .prepare("SELECT count FROM plus_usage WHERE user_id = ? AND usage_date = ?")
        .bind(userId, today)
        .first();
      const used = (row?.count as number) ?? 0;
      remaining = Math.max(0, PLUS_FREE_DAILY_LIMIT - used);
    } catch (e) {
      console.error("[plus/precheck] D1 read error:", e);
    }
  } else if (isPro) {
    remaining = -1;
  }

  const mainlandCn = cfClientCountry(request) === "CN";

  return NextResponse.json({
    allowed: true,
    user_id: userId,
    is_pro: isPro,
    remaining,
    legacy: true,
    detail: "Use POST /api/plus for analysis; this endpoint is read-only.",
    ...(mainlandCn ? { network_hint: "cn" as const } : {}),
  });
}
