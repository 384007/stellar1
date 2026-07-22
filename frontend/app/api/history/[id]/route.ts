import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { normalizedTotalScoreForStorage } from "@/lib/safe-analysis-score";
import {
  historyRetentionCutoffIso,
  purgeExpiredHistoryForUser,
  resolveHistoryRetentionDays,
} from "@/lib/pro-history-retention";
import { sanitizeProductJson } from "@/lib/chains/sanitize";

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

export async function GET(
  request: NextRequest,
  ctx: { params: Promise<{ id: string }> }
) {
  const userId = await getUserId(request);
  if (!userId) return NextResponse.json({ detail: "未登录" }, { status: 401 });

  const db = getDB();
  if (!db) return NextResponse.json({ detail: "数据库不可用" }, { status: 503 });

  const { id } = await ctx.params;
  if (!id) return NextResponse.json({ detail: "缺少记录ID" }, { status: 400 });

  const retentionDays = resolveHistoryRetentionDays(cfBindingEnv());
  await purgeExpiredHistoryForUser(db, getR2(), userId, historyRetentionCutoffIso(retentionDays));

  const rec = await db
    .prepare(
      "SELECT id, type, video_url, video_r2_key, result_r2_key, total_score, result_json, created_at FROM analyses WHERE id = ? AND user_id = ? LIMIT 1"
    )
    .bind(id, userId)
    .first();

  if (!rec) return NextResponse.json({ detail: "记录不存在" }, { status: 404 });

  let resultJson = (rec.result_json as string) || "{}";
  const resultR2Key = (rec.result_r2_key as string) || "";
  if (resultR2Key) {
    try {
      const r2 = getR2();
      if (r2) {
        const obj = await r2.get(resultR2Key);
        if (obj) resultJson = await obj.text();
      }
    } catch { /* fallback to d1 result_json */ }
  }

  let safeResultJson = resultJson;
  try {
    safeResultJson = JSON.stringify(sanitizeProductJson(JSON.parse(resultJson), "record"));
  } catch {
    /* keep string */
  }

  return NextResponse.json({
    id: rec.id,
    type: rec.type,
    video_url: rec.video_url || "",
    video_r2_key: rec.video_r2_key || "",
    total_score: normalizedTotalScoreForStorage(rec.total_score),
    result_json: safeResultJson,
    created_at: rec.created_at,
  });
}

/** 用户删除一条分析：删 R2 视频/大 JSON、share_tokens、D1 analyses。 */
export async function DELETE(
  request: NextRequest,
  ctx: { params: Promise<{ id: string }> },
) {
  const userId = await getUserId(request);
  if (!userId) return NextResponse.json({ detail: "未登录" }, { status: 401 });

  const db = getDB();
  if (!db) return NextResponse.json({ detail: "数据库不可用" }, { status: 503 });

  const { id } = await ctx.params;
  if (!id) return NextResponse.json({ detail: "缺少记录ID" }, { status: 400 });

  const rec = await db
    .prepare(
      "SELECT id, video_r2_key, result_r2_key FROM analyses WHERE id = ? AND user_id = ? LIMIT 1",
    )
    .bind(id, userId)
    .first();

  if (!rec) return NextResponse.json({ detail: "记录不存在" }, { status: 404 });

  const r2 = getR2();
  const vk = String((rec as { video_r2_key?: string }).video_r2_key || "").trim();
  const rk = String((rec as { result_r2_key?: string }).result_r2_key || "").trim();
  if (r2 && vk) {
    try {
      await r2.delete(vk);
    } catch {
      /* ignore */
    }
  }
  if (r2 && rk) {
    try {
      await r2.delete(rk);
    } catch {
      /* ignore */
    }
  }

  try {
    await db
      .prepare("DELETE FROM share_tokens WHERE analysis_id = ? AND user_id = ?")
      .bind(id, userId)
      .run();
  } catch {
    /* share_tokens 表可能不存在 */
  }

  await db.prepare("DELETE FROM analyses WHERE id = ? AND user_id = ?").bind(id, userId).run();

  return NextResponse.json({ success: true, id });
}

