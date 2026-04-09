import { NextRequest, NextResponse } from "next/server";
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

function countKeyframeImages(obj: Record<string, unknown>): number {
  const kf = obj.keyframes;
  if (!Array.isArray(kf)) return 0;
  let n = 0;
  for (const item of kf) {
    if (!item || typeof item !== "object") continue;
    const b64 = (item as Record<string, unknown>).image_base64;
    if (typeof b64 === "string" && b64.length > 40) n += 1;
  }
  return n;
}

function anyKeyframeImageMissing(obj: Record<string, unknown>): boolean {
  const kf = obj.keyframes;
  if (!Array.isArray(kf)) return false;
  for (const item of kf) {
    if (!item || typeof item !== "object") continue;
    const b64 = (item as Record<string, unknown>).image_base64;
    if (typeof b64 !== "string" || b64.trim().length < 40) return true;
  }
  return false;
}

function keyframeMetaRichness(obj: Record<string, unknown>): number {
  const kf = obj.keyframes;
  if (!Array.isArray(kf)) return 0;
  let s = 0;
  for (const item of kf) {
    if (!item || typeof item !== "object") continue;
    const k = item as Record<string, unknown>;
    if (typeof k.frame_index === "number") s += 2;
    if (typeof k.confidence === "number") s += 2;
    if (typeof k.fallback_used === "boolean") s += 2;
    if (typeof k.selection_reason === "string" && k.selection_reason.length > 0) s += 1;
  }
  return s;
}

function validationRichness(obj: Record<string, unknown>): number {
  let s = 0;
  if (obj.phase_validation && typeof obj.phase_validation === "object" && !Array.isArray(obj.phase_validation)) {
    s += 6;
  }
  if (obj.keyframe_validation && typeof obj.keyframe_validation === "object" && !Array.isArray(obj.keyframe_validation)) {
    s += 6;
  }
  if (obj.analysis_reliability && typeof obj.analysis_reliability === "object" && !Array.isArray(obj.analysis_reliability)) {
    s += 5;
  }
  if (typeof obj.phase_source === "string" && obj.phase_source.length > 0) s += 3;
  return s;
}

/**
 * Richness score for choosing D1 vs R2. Keyframe images and validation blobs weigh heavily
 * so compact D1 rows do not beat full R2 payloads.
 */
function resultRichnessScore(obj: Record<string, unknown>): number {
  let s = 0;
  const kf = obj.keyframes;
  const kfLen = Array.isArray(kf) ? kf.length : 0;
  if (kfLen > 0) s += Math.min(kfLen, 12) * 2;
  s += countKeyframeImages(obj) * 10;
  s += keyframeMetaRichness(obj);
  s += validationRichness(obj);
  if (obj.primary_diagnosis && typeof obj.primary_diagnosis === "object") s += 5;
  if (obj.prediction && typeof obj.prediction === "object") s += 2;
  if (typeof obj.total_score === "number" && obj.total_score > 0) s += 1;
  if (obj.pose_frames && Array.isArray(obj.pose_frames)) {
    s += Math.min((obj.pose_frames as unknown[]).length, 120);
  }

  const kv = obj.keyframe_validation;
  if (kv && typeof kv === "object" && !Array.isArray(kv)) {
    const nd = (kv as { near_duplicates?: number }).near_duplicates;
    if (typeof nd === "number" && nd > 0) s -= 4 * nd;
  }
  const pv = obj.phase_validation;
  if (pv && typeof pv === "object" && !Array.isArray(pv)) {
    if ((pv as { passed?: boolean }).passed === false) s -= 6;
  }
  if (obj.phase_source === "kinematic_degraded") s -= 4;
  const ar = obj.analysis_reliability;
  if (ar && typeof ar === "object" && !Array.isArray(ar)) {
    if ((ar as { level?: string }).level === "low") s -= 3;
  }
  if (kfLen > 0 && kfLen < 4) s -= 12;
  return s;
}

/**
 * Compare D1 and R2 result JSON and pick the better one.
 * Prefers R2 when it is richer (images, validation, metadata), especially if D1 is image-less compact.
 */
function pickBetterResult(
  d1Json: string,
  r2Json: string
): { json: string; source: "d1" | "r2" } {
  let d1: Record<string, unknown> = {};
  let r2: Record<string, unknown> = {};
  try {
    d1 = JSON.parse(d1Json);
  } catch {
    d1 = {};
  }
  try {
    r2 = JSON.parse(r2Json);
  } catch {
    return { json: d1Json, source: "d1" };
  }

  const dRich = resultRichnessScore(d1);
  const rRich = resultRichnessScore(r2);
  const d1Imgs = countKeyframeImages(d1);
  const r2Imgs = countKeyframeImages(r2);
  const d1Kf = Array.isArray(d1.keyframes) ? d1.keyframes.length : 0;
  const r2Kf = Array.isArray(r2.keyframes) ? r2.keyframes.length : 0;
  const d1Poses = Array.isArray(d1.pose_frames) ? d1.pose_frames.length : 0;
  const r2Poses = Array.isArray(r2.pose_frames) ? r2.pose_frames.length : 0;

  // Video overlay needs pose_frames — compact D1 often omits them while R2 has the full analysis.
  if (r2Poses > 0 && d1Poses === 0) {
    return { json: r2Json, source: "r2" };
  }

  // R2 wins when strictly richer, or when D1 has keyframes but no images while R2 has images.
  if (rRich > dRich) {
    return { json: r2Json, source: "r2" };
  }
  if (d1Kf > 0 && d1Imgs === 0 && r2Imgs > 0) {
    return { json: r2Json, source: "r2" };
  }
  if (r2Kf > 0 && r2Imgs > d1Imgs && validationRichness(r2) >= validationRichness(d1)) {
    return { json: r2Json, source: "r2" };
  }
  // Tie-break: prefer R2 if it adds validation fields D1 lacks.
  if (Math.abs(rRich - dRich) <= 1 && validationRichness(r2) > validationRichness(d1) + 3) {
    return { json: r2Json, source: "r2" };
  }

  return { json: d1Json, source: "d1" };
}

/** GET /api/share/[token]  — no auth required */
export async function GET(
  _request: NextRequest,
  ctx: { params: Promise<{ token: string }> }
) {
  const db = getDB();
  if (!db) return NextResponse.json({ detail: "数据库不可用" }, { status: 503 });

  const { token } = await ctx.params;
  if (!token) return NextResponse.json({ detail: "缺少 token" }, { status: 400 });

  const shareRec = await db
    .prepare("SELECT analysis_id, user_id FROM share_tokens WHERE token = ? LIMIT 1")
    .bind(token)
    .first();

  if (!shareRec) {
    return NextResponse.json({ detail: "分享链接无效或已过期" }, { status: 404 });
  }

  const ownerId = String(shareRec.user_id || "");
  if (ownerId) {
    const retentionDays = resolveHistoryRetentionDays(cfBindingEnv());
    await purgeExpiredHistoryForUser(db, getR2(), ownerId, historyRetentionCutoffIso(retentionDays));
  }

  const rec = await db
    .prepare(
      "SELECT id, type, video_r2_key, result_r2_key, total_score, result_json, created_at FROM analyses WHERE id = ? AND user_id = ? LIMIT 1"
    )
    .bind(shareRec.analysis_id, shareRec.user_id)
    .first();

  if (!rec) {
    return NextResponse.json({ detail: "记录不存在" }, { status: 404 });
  }

  let resultJson = (rec.result_json as string) || "{}";
  const resultR2Key = (rec.result_r2_key as string) || "";
  let resultSource: "d1" | "r2" = "d1";
  let r2Unreadable = false;

  if (resultR2Key) {
    try {
      const r2 = getR2();
      if (r2) {
        const obj = await r2.get(resultR2Key);
        if (obj) {
          const r2Json = await obj.text();
          const chosen = pickBetterResult(resultJson, r2Json);
          resultJson = chosen.json;
          resultSource = chosen.source;
        } else {
          r2Unreadable = true;
        }
      } else {
        r2Unreadable = true;
      }
    } catch {
      r2Unreadable = true;
    }
  }

  // Detect stale/broken results so the UI can show a re-analyze prompt
  let resultStale = false;
  let image_missing = false;
  let result_partial = false;
  try {
    const parsed: Record<string, unknown> = JSON.parse(resultJson);
    const kv = parsed.keyframe_validation as Record<string, unknown> | undefined;
    const nearDups = typeof kv?.near_duplicates === "number" ? kv.near_duplicates : 0;
    const phaseSource = parsed.phase_source as string | undefined;
    if (nearDups >= 4 && phaseSource === "kinematic_degraded") {
      resultStale = true;
    }
    const kf = parsed.keyframes;
    const kfLen = Array.isArray(kf) ? kf.length : 0;
    const imgs = countKeyframeImages(parsed);
    image_missing = (kfLen > 0 && imgs === 0) || anyKeyframeImageMissing(parsed) || parsed.image_missing === true;
    const embeddedPartial = parsed.result_partial === true;
    const gateFail = parsed.final_keyframe_gate_pass === false;
    const degraded = parsed.keyframes_degraded === true || parsed.keyframe_display_mode === "degraded_failed";
    result_partial = embeddedPartial || gateFail || degraded || image_missing || (r2Unreadable && !!resultR2Key);
  } catch {
    result_partial = r2Unreadable && !!resultR2Key;
  }

  let safeResultJson = resultJson;
  try {
    safeResultJson = JSON.stringify(sanitizeProductJson(JSON.parse(resultJson), "share"));
  } catch {
    /* keep raw string */
  }

  return NextResponse.json(
    {
      id: rec.id,
      type: rec.type,
      total_score: normalizedTotalScoreForStorage(rec.total_score),
      result_json: safeResultJson,
      created_at: rec.created_at,
      has_video: !!(rec.video_r2_key as string),
      result_source: resultSource,
      result_stale: resultStale,
      result_partial,
      image_missing,
    },
    {
      headers: {
        // Don't cache stale results — let the browser re-fetch
        "Cache-Control": resultStale ? "no-store" : "public, max-age=300",
      },
    }
  );
}
