import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";

function getJwtSecret(): Uint8Array {
  let secret = process.env.JWT_SECRET || "";
  if (!secret) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      secret = (getRequestContext().env as any).JWT_SECRET || "";
    } catch { /* ignore */ }
  }
  if (!secret) throw new Error("JWT_SECRET not configured");
  return new TextEncoder().encode(secret);
}

function getDB() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).DB;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  const auth = request.headers.get("authorization");
  if (!auth?.startsWith("Bearer ")) {
    return NextResponse.json({ detail: "未登录" }, { status: 401 });
  }

  const token = auth.slice(7);
  if (token.startsWith("local-")) {
    return NextResponse.json({ detail: "请重新登录" }, { status: 401 });
  }

  let userId: string;
  try {
    const { payload } = await jwtVerify(token, getJwtSecret());
    userId = (payload.user_id as string) || "";
  } catch {
    return NextResponse.json({ detail: "登录已过期" }, { status: 401 });
  }

  if (!userId) {
    return NextResponse.json({ detail: "无效凭证" }, { status: 401 });
  }

  const db = getDB();
  if (!db) {
    return NextResponse.json({ detail: "数据库不可用" }, { status: 503 });
  }

  try {
    const user = await db
      .prepare("SELECT id, email, username, is_pro, daily_count, created_at FROM users WHERE id = ?")
      .bind(userId)
      .first();

    if (!user) {
      return NextResponse.json({ detail: "用户不存在" }, { status: 404 });
    }

    const analysisCount = await db
      .prepare("SELECT COUNT(*) as count FROM analyses WHERE user_id = ?")
      .bind(userId)
      .first();

    return NextResponse.json({
      user: {
        id: user.id,
        email: user.email,
        username: user.username,
        is_pro: !!user.is_pro,
        daily_count: user.daily_count,
        created_at: user.created_at,
      },
      stats: {
        total_analyses: analysisCount?.count || 0,
      },
    });
  } catch (e) {
    return NextResponse.json(
      { detail: "查询失败，请稍后重试。" },
      { status: 500 }
    );
  }
}
