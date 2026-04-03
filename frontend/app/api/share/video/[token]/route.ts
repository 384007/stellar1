import { NextRequest, NextResponse } from "next/server";
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

/** GET /api/share/video/[token]  — streams video from R2 without requiring login */
export async function GET(
  request: NextRequest,
  ctx: { params: Promise<{ token: string }> }
) {
  const db = getDB();
  const r2 = getR2();
  if (!db || !r2) return NextResponse.json({ detail: "服务不可用" }, { status: 503 });

  const { token } = await ctx.params;
  if (!token) return NextResponse.json({ detail: "缺少 token" }, { status: 400 });

  const shareRec = await db
    .prepare("SELECT analysis_id, user_id FROM share_tokens WHERE token = ? LIMIT 1")
    .bind(token)
    .first();

  if (!shareRec) {
    return NextResponse.json({ detail: "分享链接无效" }, { status: 404 });
  }

  const ownerId = String(shareRec.user_id || "");
  if (ownerId) {
    const retentionDays = resolveHistoryRetentionDays(cfBindingEnv());
    await purgeExpiredHistoryForUser(db, r2, ownerId, historyRetentionCutoffIso(retentionDays));
  }

  const rec = await db
    .prepare("SELECT video_r2_key FROM analyses WHERE id = ? AND user_id = ? LIMIT 1")
    .bind(shareRec.analysis_id, shareRec.user_id)
    .first();

  const videoKey = (rec?.video_r2_key as string) || "";
  if (!videoKey) return NextResponse.json({ detail: "视频不存在" }, { status: 404 });

  const rangeHeader = request.headers.get("range");

  if (rangeHeader) {
    // Safari and most browsers require proper Range support to play video.
    // Get metadata first so we can build a correct Content-Range header.
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
          "Cache-Control": "public, max-age=3600",
        },
      });
    }
  }

  // Full response (no Range header, or Range parse failed)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const obj = await (r2 as any).get(videoKey);
  if (!obj) return NextResponse.json({ detail: "视频不存在" }, { status: 404 });

  const totalSize: number = obj.size ?? 0;
  const headers: Record<string, string> = {
    "Content-Type": obj.httpMetadata?.contentType || "video/mp4",
    "Accept-Ranges": "bytes",
    "Cache-Control": "public, max-age=3600",
  };
  if (totalSize > 0) headers["Content-Length"] = String(totalSize);

  return new NextResponse(obj.body, { status: 200, headers });
}
