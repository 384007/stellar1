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

async function getUserId(request: NextRequest): Promise<string | null> {
  const auth = request.headers.get("authorization");
  if (!auth?.startsWith("Bearer ")) return null;
  const token = auth.slice(7);
  if (token.startsWith("local-")) return null;
  try {
    const { payload } = await jwtVerify(token, getJwtSecret());
    return (payload.user_id as string) || null;
  } catch {
    return null;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function ensureShareSchema(db: any) {
  try {
    await db.prepare(
      `CREATE TABLE IF NOT EXISTS share_tokens (
        token TEXT PRIMARY KEY,
        analysis_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`
    ).run();
    await db.prepare(
      "CREATE INDEX IF NOT EXISTS idx_share_tokens_analysis ON share_tokens(analysis_id)"
    ).run();
  } catch { /* table already exists */ }
}

/** POST /api/share  — requires auth, returns { token, url } */
export async function POST(request: NextRequest) {
  const userId = await getUserId(request);
  if (!userId) {
    return NextResponse.json({ detail: "未登录" }, { status: 401 });
  }

  const db = getDB();
  if (!db) {
    return NextResponse.json({ detail: "数据库不可用" }, { status: 503 });
  }

  await ensureShareSchema(db);

  let body: { analysis_id?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "请求格式错误" }, { status: 400 });
  }

  const { analysis_id } = body;
  if (!analysis_id) {
    return NextResponse.json({ detail: "缺少 analysis_id" }, { status: 400 });
  }

  const rec = await db
    .prepare("SELECT id FROM analyses WHERE id = ? AND user_id = ? LIMIT 1")
    .bind(analysis_id, userId)
    .first();

  if (!rec) {
    return NextResponse.json({ detail: "记录不存在" }, { status: 404 });
  }

  const existing = await db
    .prepare("SELECT token FROM share_tokens WHERE analysis_id = ? AND user_id = ? LIMIT 1")
    .bind(analysis_id, userId)
    .first();

  const origin = new URL(request.url).origin;

  if (existing?.token) {
    return NextResponse.json({
      token: existing.token as string,
      url: `${origin}/share/${existing.token as string}`,
    });
  }

  const token = crypto.randomUUID().replace(/-/g, "");
  await db
    .prepare(
      "INSERT INTO share_tokens (token, analysis_id, user_id, created_at) VALUES (?, ?, ?, ?)"
    )
    .bind(token, analysis_id, userId, new Date().toISOString())
    .run();

  return NextResponse.json({ token, url: `${origin}/share/${token}` });
}
