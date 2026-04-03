import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";

const PLUS_FREE_DAILY_LIMIT = 3;

function getDB() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).DB;
  } catch {
    return null;
  }
}

function getJwtSecret(): Uint8Array {
  let secret = "";
  try {
    secret = (getRequestContext().env as Record<string, string>).JWT_SECRET || "";
  } catch { /* */ }
  if (!secret) secret = process.env.JWT_SECRET || "";
  return new TextEncoder().encode(secret);
}

export async function GET(request: NextRequest) {
  const auth = request.headers.get("authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : auth;

  if (!token || token.startsWith("guest-")) {
    return NextResponse.json({ detail: "未登录" }, { status: 401 });
  }

  let userId = "unknown";
  let isPro = false;

  if (token.startsWith("local-")) {
    userId = token;
  } else if (token.includes(".")) {
    const secret = getJwtSecret();
    if (secret.length > 0) {
      try {
        const { payload } = await jwtVerify(token, secret);
        userId = (payload.user_id as string) || "unknown";
        isPro = !!(payload.is_pro);
      } catch {
        return NextResponse.json({ detail: "登录已过期" }, { status: 401 });
      }
    }
  }

  if (isPro) {
    return NextResponse.json({
      used: 0,
      remaining: -1,
      limit: null,
      is_pro: true,
    });
  }

  const db = getDB();
  if (!db) {
    return NextResponse.json({
      used: 0,
      remaining: PLUS_FREE_DAILY_LIMIT,
      limit: PLUS_FREE_DAILY_LIMIT,
      is_pro: false,
    });
  }

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

    return NextResponse.json({
      used,
      remaining: Math.max(0, PLUS_FREE_DAILY_LIMIT - used),
      limit: PLUS_FREE_DAILY_LIMIT,
      is_pro: false,
    });
  } catch {
    return NextResponse.json({
      used: 0,
      remaining: PLUS_FREE_DAILY_LIMIT,
      limit: PLUS_FREE_DAILY_LIMIT,
      is_pro: false,
    });
  }
}
