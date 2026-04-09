import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";

/** Plus on Modal can exceed 2m; keep below worker max where possible */
const MODAL_PLUS_TIMEOUT_MS = 360_000;
const PLUS_FREE_DAILY_LIMIT = 3;

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

function getModalUrl(): string {
  return (
    getCfEnv("MODAL_BACKEND_URL") ||
    process.env.MODAL_BACKEND_URL ||
    "https://dytsui--stellar-ai-fastapi-app.modal.run"
  )
    .trim()
    .replace(/\/+$/, "");
}

async function fetchPlusModalOnly(file: File): Promise<Response> {
  const modalUrl = getModalUrl();
  const form = new FormData();
  form.append("file", file);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), MODAL_PLUS_TIMEOUT_MS);
  try {
    return await fetch(`${modalUrl}/analyze/plus`, {
      method: "POST",
      body: form,
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
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

function getJwtSecret(): Uint8Array {
  let secret = "";
  try {
    secret = (getRequestContext().env as Record<string, string>).JWT_SECRET || "";
  } catch {
    /* not in CF context */
  }
  if (!secret) secret = process.env.JWT_SECRET || "";
  return new TextEncoder().encode(secret);
}

interface AuthResult {
  user_id: string;
  is_pro: boolean;
}

async function authenticate(request: NextRequest): Promise<AuthResult | NextResponse> {
  const auth = request.headers.get("authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : auth;

  if (!token || token.startsWith("guest-")) {
    return NextResponse.json({ detail: "请先登录后再使用 Plus 分析" }, { status: 401 });
  }

  if (token.startsWith("local-")) {
    return { user_id: token, is_pro: false };
  }

  if (!token.includes(".")) {
    return NextResponse.json({ detail: "登录状态无效" }, { status: 401 });
  }

  const secret = getJwtSecret();
  if (secret.length === 0) {
    return { user_id: "unknown", is_pro: false };
  }

  try {
    const { payload } = await jwtVerify(token, secret);
    return {
      user_id: (payload.user_id as string) || "unknown",
      is_pro: !!(payload.is_pro),
    };
  } catch {
    return NextResponse.json({ detail: "登录已过期，请重新登录" }, { status: 401 });
  }
}

async function checkAndIncrementUsage(
  db: ReturnType<typeof getDB>,
  userId: string,
  isPro: boolean
): Promise<{ allowed: boolean; remaining: number; used: number }> {
  if (isPro) {
    return { allowed: true, remaining: -1, used: 0 };
  }

  if (!db) {
    return { allowed: true, remaining: PLUS_FREE_DAILY_LIMIT, used: 0 };
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

    if (used >= PLUS_FREE_DAILY_LIMIT) {
      return { allowed: false, remaining: 0, used };
    }

    await db
      .prepare(
        "INSERT INTO plus_usage (user_id, usage_date, count) VALUES (?, ?, 1) ON CONFLICT(user_id, usage_date) DO UPDATE SET count = count + 1"
      )
      .bind(userId, today)
      .run();

    return { allowed: true, remaining: PLUS_FREE_DAILY_LIMIT - used - 1, used: used + 1 };
  } catch (e) {
    console.error("[plus] Usage check error:", e);
    return { allowed: true, remaining: PLUS_FREE_DAILY_LIMIT, used: 0 };
  }
}

export async function POST(request: NextRequest) {
  try {
    const authResult = await authenticate(request);
    if (authResult instanceof NextResponse) return authResult;

    const { user_id, is_pro } = authResult;
    const db = getDB();

    const usage = await checkAndIncrementUsage(db, user_id, is_pro);
    if (!usage.allowed) {
      return NextResponse.json(
        {
          detail: "今日 Plus 分析次数已达上限（3次/天）。升级 Pro 可获得无限次 Plus 分析。",
          limit_reached: true,
          used: usage.used,
          limit: PLUS_FREE_DAILY_LIMIT,
        },
        { status: 429 }
      );
    }

    const contentType = request.headers.get("content-type") || "";
    if (!contentType.includes("multipart")) {
      return NextResponse.json({ detail: "请上传视频文件" }, { status: 400 });
    }

    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    if (!file) {
      return NextResponse.json({ detail: "请上传文件" }, { status: 400 });
    }

    let res: Response;
    try {
      res = await fetchPlusModalOnly(file);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        return NextResponse.json({ detail: "Plus 分析超时，请压缩视频后重试" }, { status: 504 });
      }
      return NextResponse.json({ detail: `Modal 不可达: ${e instanceof Error ? e.message : "网络错误"}` }, { status: 502 });
    }

    if (!res.ok) {
      const errBody = await res.text().catch(() => "");
      return NextResponse.json(
        { detail: `Plus 分析失败 [${res.status}]: ${errBody.substring(0, 200)}` },
        { status: res.status >= 400 && res.status < 500 ? res.status : 502 }
      );
    }

    const result = await res.json();

    return NextResponse.json({
      ...result,
      _plus_usage: {
        used: usage.used,
        remaining: usage.remaining,
        limit: is_pro ? null : PLUS_FREE_DAILY_LIMIT,
        is_pro,
      },
    });
  } catch (err) {
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : "Plus 分析错误" },
      { status: 500 }
    );
  }
}
