/**
 * POST /api/history/upload-video/presign — **deprecated for product use**.
 *
 * The browser no longer receives signed R2 ``upload_url`` or PUTs to ``*.r2.cloudflarestorage.com``.
 * Pro v3 and all uploads use same-origin ``POST /api/history/upload-video`` only.
 *
 * This route remains for compatibility: authenticated callers always get ``{ mode: "proxy" }``.
 */

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
    } catch {
      /* ignore */
    }
  }
  return new TextEncoder().encode(secret);
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

export async function POST(request: NextRequest) {
  const userId = await getUserId(request);
  if (!userId) return NextResponse.json({ detail: "未登录" }, { status: 401 });
  return NextResponse.json({ mode: "proxy" as const });
}
