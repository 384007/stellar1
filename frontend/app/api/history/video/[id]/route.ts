import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";
import {
  historyRetentionCutoffIso,
  purgeExpiredHistoryForUser,
  resolveHistoryRetentionDays,
} from "@/lib/pro-history-retention";

export const runtime = "edge";

function getDB() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).DB;
  } catch {
    return null;
  }
}

function getR2() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).R2_BUCKET || null;
  } catch {
    return null;
  }
}

function cfBindingEnv(): Record<string, unknown> | undefined {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return getRequestContext().env as any;
  } catch {
    return undefined;
  }
}

function getJwtSecret(): Uint8Array {
  let secret = process.env.JWT_SECRET || "";
  if (!secret) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      secret = (getRequestContext().env as any).JWT_SECRET || "";
    } catch { /* ignore */ }
  }
  return new TextEncoder().encode(secret);
}

async function getUserId(request: NextRequest): Promise<string | null> {
  let token = "";
  const auth = request.headers.get("authorization");
  if (auth?.startsWith("Bearer ")) {
    token = auth.slice(7);
  } else {
    token = new URL(request.url).searchParams.get("token") || "";
  }
  if (!token || token.startsWith("local-")) return null;
  try {
    const { payload } = await jwtVerify(token, getJwtSecret());
    return (payload.user_id as string) || null;
  } catch {
    return null;
  }
}

export async function GET(
  request: NextRequest,
  ctx: { params: Promise<{ id: string }> }
) {
  const userId = await getUserId(request);
  if (!userId) return NextResponse.json({ detail: "未登录" }, { status: 401 });

  const db = getDB();
  const r2 = getR2();
  if (!db || !r2) return NextResponse.json({ detail: "服务不可用" }, { status: 503 });

  const retentionDays = resolveHistoryRetentionDays(cfBindingEnv());
  await purgeExpiredHistoryForUser(db, r2, userId, historyRetentionCutoffIso(retentionDays));

  const { id } = await ctx.params;
  const rec = await db
    .prepare("SELECT video_r2_key FROM analyses WHERE id = ? AND user_id = ? LIMIT 1")
    .bind(id, userId)
    .first();

  const videoKey = (rec?.video_r2_key as string) || "";
  if (!videoKey) return NextResponse.json({ detail: "视频不存在" }, { status: 404 });

  const rangeHeader = request.headers.get("range");

  if (rangeHeader) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const head = await (r2 as any).head(videoKey);
    if (!head) return NextResponse.json({ detail: "视频不存在" }, { status: 404 });

    const totalSize: number = head.size ?? 0;
    const contentType: string = head.httpMetadata?.contentType || "video/mp4";

    const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
    if (match && totalSize > 0) {
      const start = parseInt(match[1], 10);
      const end = match[2] ? Math.min(parseInt(match[2], 10), totalSize - 1) : totalSize - 1;
      const chunkSize = end - start + 1;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const obj = await (r2 as any).get(videoKey, { range: { offset: start, length: chunkSize } });
      if (!obj) return NextResponse.json({ detail: "视频不存在" }, { status: 404 });

      return new NextResponse(obj.body, {
        status: 206,
        headers: {
          "Content-Type": contentType,
          "Content-Range": `bytes ${start}-${end}/${totalSize}`,
          "Content-Length": String(chunkSize),
          "Accept-Ranges": "bytes",
          "Cache-Control": "private, max-age=3600",
        },
      });
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const obj = await (r2 as any).get(videoKey);
  if (!obj) return NextResponse.json({ detail: "视频不存在" }, { status: 404 });

  const totalSize: number = obj.size ?? 0;
  const headers: Record<string, string> = {
    "Content-Type": obj.httpMetadata?.contentType || "video/mp4",
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, max-age=3600",
  };
  if (totalSize > 0) headers["Content-Length"] = String(totalSize);

  return new NextResponse(obj.body, { status: 200, headers });
}

