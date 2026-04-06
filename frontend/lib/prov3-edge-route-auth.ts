/**
 * Shared Pro JWT + D1 is_pro check for Edge routes that proxy or read Pro v3 job state.
 * Mirrors ``app/api/prov3/precheck/route.ts`` validation semantics.
 */

import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";

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

function cfClientCountry(request: NextRequest): string {
  return (
    request.headers.get("cf-ipcountry") ||
    request.headers.get("CF-IPCountry") ||
    ""
  )
    .trim()
    .toUpperCase();
}

export type ProEdgeAuthResult =
  | { ok: true; userId: string; cnHint: boolean }
  | { ok: false; response: NextResponse };

export async function requireProUserForProv3Edge(
  request: NextRequest,
): Promise<ProEdgeAuthResult> {
  const auth = request.headers.get("authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : auth;

  if (!token || token.startsWith("guest-")) {
    return { ok: false, response: NextResponse.json({ detail: "请先登录 Pro 账户" }, { status: 401 }) };
  }
  if (token.startsWith("local-")) {
    return {
      ok: false,
      response: NextResponse.json({ detail: "登录凭证已失效，请重新登录" }, { status: 401 }),
    };
  }
  if (!token.includes(".")) {
    return { ok: false, response: NextResponse.json({ detail: "无效凭证" }, { status: 401 }) };
  }

  let jwtSecret = getCfEnv("JWT_SECRET");
  if (!jwtSecret) jwtSecret = process.env.JWT_SECRET || "";
  if (!jwtSecret) {
    return { ok: false, response: NextResponse.json({ detail: "服务器配置错误" }, { status: 500 }) };
  }

  let isPro = false;
  let userId = "";

  try {
    const { payload } = await jwtVerify(token, new TextEncoder().encode(jwtSecret));
    isPro = !!payload.is_pro;
    userId = (payload.user_id as string) || "";
  } catch {
    return { ok: false, response: NextResponse.json({ detail: "登录已过期，请重新登录" }, { status: 401 }) };
  }

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
        /* fall back to JWT */
      }
    }
  }

  if (!isPro) {
    return {
      ok: false,
      response: NextResponse.json(
        { detail: "您不是 Pro 用户，请先升级", is_pro: false },
        { status: 403 },
      ),
    };
  }

  return { ok: true, userId, cnHint: cfClientCountry(request) === "CN" };
}
