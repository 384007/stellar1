import { NextRequest, NextResponse } from "next/server";
import { SignJWT } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";

function getCfEnv(key: string): string {
  let val = process.env[key] || "";
  if (!val) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      val = (getRequestContext().env as any)[key] as string || "";
    } catch { /* ignore */ }
  }
  return val;
}

function getJwtSecret(): Uint8Array {
  const secret = getCfEnv("JWT_SECRET");
  if (!secret) throw new Error("JWT_SECRET not configured");
  return new TextEncoder().encode(secret);
}

async function hashPw(
  password: string,
  salt?: string
): Promise<{ hash: string; salt: string }> {
  const s = salt || crypto.randomUUID();
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(password),
    "PBKDF2",
    false,
    ["deriveBits"]
  );
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt: enc.encode(s), iterations: 100000, hash: "SHA-256" },
    key,
    256
  );
  const hex = Array.from(new Uint8Array(bits))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return { hash: hex, salt: s };
}

async function makeToken(payload: Record<string, unknown>): Promise<string> {
  return new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("72h")
    .sign(getJwtSecret());
}

interface D1Binding {
  prepare(query: string): {
    bind(...values: unknown[]): {
      first<T>(): Promise<T | null>;
      run(): Promise<unknown>;
    };
  };
}

function getDB(): D1Binding | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).DB as D1Binding;
  } catch {
    return null;
  }
}

/**
 * Validate invite code (case-insensitive).
 * Checks PRO_INVITE_CODE env var first; falls back to "stellar".
 */
function isValidInviteCode(code: string): boolean {
  if (!code) return false;
  const normalised = code.trim().toLowerCase();
  const raw = getCfEnv("PRO_INVITE_CODE");
  if (raw) {
    return raw.split(",").map((s) => s.trim().toLowerCase()).includes(normalised);
  }
  return normalised === "stellar";
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, email, password, username } = body;

    if (action === "guest") {
      return NextResponse.json(
        { detail: "游客模式已关闭，请注册账户后使用" },
        { status: 403 }
      );
    }

    if (!email || !password) {
      return NextResponse.json(
        { detail: "请输入邮箱和密码" },
        { status: 400 }
      );
    }

    const db = getDB();

    // ── 普通注册 ──
    if (action === "register") {
      if (!db) {
        return NextResponse.json(
          { detail: "数据库未就绪，请确认D1已绑定" },
          { status: 503 }
        );
      }
      if (!username || username.trim().length < 2) {
        return NextResponse.json(
          { detail: "请输入用户名（至少2个字符）" },
          { status: 400 }
        );
      }

      const existing = await db
        .prepare("SELECT id FROM users WHERE email = ?")
        .bind(email)
        .first();
      if (existing) {
        return NextResponse.json(
          { detail: "该邮箱已注册" },
          { status: 409 }
        );
      }

      const userId = crypto.randomUUID();
      const { hash, salt } = await hashPw(password);
      const pwHash = `${salt}:${hash}`;
      const today = new Date().toISOString().split("T")[0];

      await db
        .prepare(
          "INSERT INTO users (id, email, username, password_hash, is_pro, daily_count, last_reset) VALUES (?, ?, ?, ?, 0, 0, ?)"
        )
        .bind(userId, email, username.trim(), pwHash, today)
        .run();

      const token = await makeToken({
        user_id: userId,
        email,
        username: username.trim(),
        is_pro: false,
        is_guest: false,
      });

      return NextResponse.json({
        token,
        user_id: userId,
        email,
        username: username.trim(),
        is_pro: false,
        is_guest: false,
      });
    }

    // ── 普通登录 ──
    if (action === "login") {
      if (!db) {
        return NextResponse.json(
          { detail: "数据库未就绪，请确认D1已绑定" },
          { status: 503 }
        );
      }
      const user = await db
        .prepare("SELECT * FROM users WHERE email = ?")
        .bind(email)
        .first<Record<string, unknown>>();

      if (!user) {
        return NextResponse.json(
          { detail: "邮箱或密码错误" },
          { status: 401 }
        );
      }

      const [salt, storedHash] = (user.password_hash as string).split(":");
      const { hash } = await hashPw(password, salt);
      if (hash !== storedHash) {
        return NextResponse.json(
          { detail: "邮箱或密码错误" },
          { status: 401 }
        );
      }

      const token = await makeToken({
        user_id: user.id,
        email: user.email,
        username: (user.username as string) || (user.email as string),
        is_pro: !!user.is_pro,
        is_guest: false,
      });

      return NextResponse.json({
        token,
        user_id: user.id,
        email: user.email,
        username: (user.username as string) || (user.email as string),
        is_pro: !!user.is_pro,
        is_guest: false,
      });
    }

    // ── Pro 注册 / 升级（邀请码校验，持久化到 D1；D1 不可用时签发会话级 JWT）──
    if (action === "pro-login") {
      const { invite_code } = body as { invite_code?: string };

      if (!isValidInviteCode(invite_code ?? "")) {
        return NextResponse.json(
          { detail: "邀请码无效" },
          { status: 401 }
        );
      }

      // D1 not available — issue a session-only Pro JWT (no persistence)
      if (!db) {
        const displayName = (username as string | undefined)?.trim() || email;
        const sessionId = crypto.randomUUID();
        const token = await makeToken({
          user_id: sessionId,
          email,
          username: displayName,
          is_pro: true,
          is_guest: false,
        });
        return NextResponse.json({
          token,
          user_id: sessionId,
          email,
          username: displayName,
          is_pro: true,
          is_guest: false,
        });
      }

      const existingUser = await db
        .prepare("SELECT * FROM users WHERE email = ?")
        .bind(email)
        .first<Record<string, unknown>>();

      if (existingUser) {
        // Existing account: verify password and upgrade to Pro
        const [salt, storedHash] = (existingUser.password_hash as string).split(":");
        const { hash } = await hashPw(password, salt);
        if (hash !== storedHash) {
          return NextResponse.json({ detail: "密码错误" }, { status: 401 });
        }
        await db
          .prepare("UPDATE users SET is_pro = 1 WHERE id = ?")
          .bind(existingUser.id)
          .run();

        const uname = (existingUser.username as string) || (existingUser.email as string);
        const token = await makeToken({
          user_id: existingUser.id,
          email: existingUser.email,
          username: uname,
          is_pro: true,
          is_guest: false,
        });
        return NextResponse.json({
          token,
          user_id: existingUser.id,
          email: existingUser.email,
          username: uname,
          is_pro: true,
          is_guest: false,
        });
      } else {
        // New account: register as Pro
        const displayName = (username as string | undefined)?.trim() || email;
        const userId = crypto.randomUUID();
        const { hash, salt } = await hashPw(password);
        const pwHash = `${salt}:${hash}`;
        const today = new Date().toISOString().split("T")[0];

        await db
          .prepare(
            "INSERT INTO users (id, email, username, password_hash, is_pro, daily_count, last_reset) VALUES (?, ?, ?, ?, 1, 0, ?)"
          )
          .bind(userId, email, displayName, pwHash, today)
          .run();

        const token = await makeToken({
          user_id: userId,
          email,
          username: displayName,
          is_pro: true,
          is_guest: false,
        });
        return NextResponse.json({
          token,
          user_id: userId,
          email,
          username: displayName,
          is_pro: true,
          is_guest: false,
        });
      }
    }

    return NextResponse.json({ detail: "无效操作" }, { status: 400 });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "认证错误";
    if (msg.includes("no such table")) {
      return NextResponse.json(
        { detail: "数据库表未创建，请先运行D1迁移" },
        { status: 503 }
      );
    }
    return NextResponse.json({ detail: msg }, { status: 500 });
  }
}
