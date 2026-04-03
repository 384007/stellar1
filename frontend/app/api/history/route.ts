import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { extractVideoFieldsFromSyncRecord } from "@/lib/history-sync-record";
import { slimPoseFramesForCloudRow, subsamplePoseFramesEven } from "@/lib/analysis-pose-storage";
import { normalizedTotalScoreForStorage } from "@/lib/safe-analysis-score";
import {
  historyRetentionCutoffIso,
  purgeExpiredHistoryForUser,
  resolveHistoryRetentionDays,
} from "@/lib/pro-history-retention";

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

const MAX_RESULT_JSON_LEN = 90_000;

function asStringArray(v: unknown, max = 8): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string").slice(0, max) : [];
}

/** Drop trajectory from prediction (large). */
function predictionWithoutTrajectory(pred: unknown): unknown {
  if (!pred || typeof pred !== "object" || Array.isArray(pred)) return pred;
  const p = { ...(pred as Record<string, unknown>) };
  delete p.trajectory;
  return p;
}

/** Trim long strings inside phase_debug to keep D1 row valid JSON under cap. */
function trimPhaseDebug(pd: unknown): unknown {
  if (!pd || typeof pd !== "object" || Array.isArray(pd)) return pd;
  const o = pd as Record<string, unknown>;
  const out: Record<string, unknown> = { ...o };
  for (const [k, v] of Object.entries(out)) {
    if (typeof v === "string" && v.length > 2000) out[k] = `${v.slice(0, 2000)}…`;
  }
  return out;
}

function slimVideoMeta(vm: unknown): Record<string, unknown> | undefined {
  if (!vm || typeof vm !== "object" || Array.isArray(vm)) return undefined;
  const o = vm as Record<string, unknown>;
  const slim: Record<string, unknown> = {};
  if (typeof o.source_frame_count === "number") slim.source_frame_count = o.source_frame_count;
  if (typeof o.fps === "number") slim.fps = o.fps;
  if (typeof o.duration_s === "number") slim.duration_s = o.duration_s;
  return Object.keys(slim).length ? slim : undefined;
}

function mapKeyframeForStorage(
  kf: Record<string, unknown>,
  includeImage: boolean,
  includePoseSnap: boolean
): Record<string, unknown> {
  const row: Record<string, unknown> = {
    phase: kf.phase,
    label_en: kf.label_en,
    label_zh: kf.label_zh,
    frame_index: kf.frame_index,
    timestamp: kf.timestamp,
    confidence: kf.confidence,
    selection_reason: kf.selection_reason,
    fallback_used: kf.fallback_used,
    source_pose_idx: kf.source_pose_idx,
    source_frame_index: kf.source_frame_index,
    visual_diff_from_prev: kf.visual_diff_from_prev,
    phase_validation_passed: kf.phase_validation_passed,
    reselected: kf.reselected,
    width: typeof kf.width === "number" ? kf.width : undefined,
    height: typeof kf.height === "number" ? kf.height : undefined,
  };
  if (includeImage && typeof kf.image_base64 === "string" && kf.image_base64.length > 0) {
    row.image_base64 = kf.image_base64;
  }
  if (includePoseSnap && kf.pose_snapshot && typeof kf.pose_snapshot === "object") {
    row.pose_snapshot = kf.pose_snapshot;
  }
  return row;
}

type CompactStage = {
  keyframeImages: boolean;
  keyframePoseSnap: boolean;
  trimPredictionTrajectory: boolean;
  trimPhaseDebug: boolean;
};

function buildStorageCompact(result: Record<string, unknown>, stage: CompactStage): Record<string, unknown> {
  const keyframesRaw = Array.isArray(result.keyframes)
    ? (result.keyframes as Array<Record<string, unknown>>)
    : [];
  const keyframes = keyframesRaw.map((kf) =>
    mapKeyframeForStorage(kf, stage.keyframeImages, stage.keyframePoseSnap)
  );

  let prediction: unknown = result.prediction;
  if (stage.trimPredictionTrajectory && prediction && typeof prediction === "object") {
    prediction = predictionWithoutTrajectory(prediction);
  }

  let phase_debug: unknown = result.phase_debug;
  if (stage.trimPhaseDebug && phase_debug !== undefined) {
    phase_debug = trimPhaseDebug(phase_debug);
  }

  const pose_frames_slim =
    Array.isArray(result.pose_frames) && result.pose_frames.length > 0
      ? slimPoseFramesForCloudRow(result.pose_frames)
      : undefined;

  return {
    analysis_id: result.analysis_id,
    type: result.type,
    scores: result.scores && typeof result.scores === "object" ? result.scores : {},
    total_score: normalizedTotalScoreForStorage(result.total_score),
    issues: asStringArray(result.issues),
    issues_zh: asStringArray(result.issues_zh),
    suggestions: asStringArray(result.suggestions),
    suggestions_zh: asStringArray(result.suggestions_zh),
    summary: typeof result.summary === "string" ? result.summary : "",
    summary_zh: typeof result.summary_zh === "string" ? result.summary_zh : "",
    prediction,
    keyframes,
    ...(pose_frames_slim && pose_frames_slim.length > 0 ? { pose_frames: pose_frames_slim } : {}),
    created_at: result.created_at,
    posture_score: typeof result.posture_score === "number" ? result.posture_score : undefined,
    primary_diagnosis:
      result.primary_diagnosis && typeof result.primary_diagnosis === "object"
        ? result.primary_diagnosis
        : undefined,
    additional_issues: Array.isArray(result.additional_issues) ? result.additional_issues : undefined,
    quick_tip_zh: typeof result.quick_tip_zh === "string" ? result.quick_tip_zh : undefined,
    quick_tip_en: typeof result.quick_tip_en === "string" ? result.quick_tip_en : undefined,
    problem_description_zh:
      typeof result.problem_description_zh === "string" ? result.problem_description_zh : undefined,
    problem_description_en:
      typeof result.problem_description_en === "string" ? result.problem_description_en : undefined,
    swing_phase_evaluations: Array.isArray(result.swing_phase_evaluations)
      ? result.swing_phase_evaluations
      : undefined,
    training: result.training && typeof result.training === "object" ? result.training : undefined,
    phase_keyframes:
      result.phase_keyframes && typeof result.phase_keyframes === "object"
        ? result.phase_keyframes
        : undefined,
    phase_source: result.phase_source,
    phase_validation:
      result.phase_validation && typeof result.phase_validation === "object"
        ? result.phase_validation
        : undefined,
    keyframe_validation:
      result.keyframe_validation && typeof result.keyframe_validation === "object"
        ? result.keyframe_validation
        : undefined,
    analysis_reliability:
      result.analysis_reliability && typeof result.analysis_reliability === "object"
        ? result.analysis_reliability
        : undefined,
    phase_debug,
    quality_warning: typeof result.quality_warning === "string" ? result.quality_warning : undefined,
    keyframe_warning: typeof result.keyframe_warning === "string" ? result.keyframe_warning : undefined,
    hand_warning: typeof result.hand_warning === "string" ? result.hand_warning : undefined,
    club_warning: typeof result.club_warning === "string" ? result.club_warning : undefined,
    hand_assumed: typeof result.hand_assumed === "boolean" ? result.hand_assumed : undefined,
    club_assumed: typeof result.club_assumed === "boolean" ? result.club_assumed : undefined,
    video_meta: slimVideoMeta(result.video_meta),
  };
}

function truncateCompactTextFields(compact: Record<string, unknown>, maxLen: number): Record<string, unknown> {
  const c = { ...compact };
  for (const key of ["summary", "summary_zh", "quality_warning", "keyframe_warning", "hand_warning", "club_warning"]) {
    const v = c[key];
    if (typeof v === "string" && v.length > maxLen) c[key] = `${v.slice(0, maxLen)}…`;
  }
  return c;
}

function shrinkCompactForCap(compact: Record<string, unknown>): Record<string, unknown> {
  let c: Record<string, unknown> = { ...compact };
  c = truncateCompactTextFields(c, 4000);
  let j = JSON.stringify(c);
  if (j.length <= MAX_RESULT_JSON_LEN) return c;

  c = { ...c, training: undefined, swing_phase_evaluations: undefined, additional_issues: undefined };
  j = JSON.stringify(c);
  if (j.length <= MAX_RESULT_JSON_LEN) return c;

  const kf = Array.isArray(c.keyframes) ? (c.keyframes as Array<Record<string, unknown>>) : [];
  for (let n = kf.length; n >= 1; n--) {
    c = { ...c, keyframes: kf.slice(0, n) };
    j = JSON.stringify(c);
    if (j.length <= MAX_RESULT_JSON_LEN) return c;
  }

  const pf0 = Array.isArray(c.pose_frames) ? (c.pose_frames as unknown[]) : [];
  if (pf0.length > 0) {
    for (let max = 48; max >= 12; max -= 6) {
      c = { ...c, pose_frames: subsamplePoseFramesEven(pf0, max) };
      j = JSON.stringify(c);
      if (j.length <= MAX_RESULT_JSON_LEN) return c;
    }
    c = { ...c, pose_frames: subsamplePoseFramesEven(pf0, 8) };
    j = JSON.stringify(c);
    if (j.length <= MAX_RESULT_JSON_LEN) return c;
  }

  return { ...c, keyframes: [], pose_frames: [] };
}

function compactResultForStorage(input: unknown): string {
  const fallback = "{}";
  try {
    const parsed = typeof input === "string" ? JSON.parse(input) : input;
    const result = parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : {};

    const fullJson = JSON.stringify(result);
    if (fullJson.length <= MAX_RESULT_JSON_LEN) return fullJson;

    const stages: CompactStage[] = [
      { keyframeImages: true, keyframePoseSnap: true, trimPredictionTrajectory: true, trimPhaseDebug: false },
      { keyframeImages: true, keyframePoseSnap: false, trimPredictionTrajectory: true, trimPhaseDebug: false },
      { keyframeImages: true, keyframePoseSnap: false, trimPredictionTrajectory: true, trimPhaseDebug: true },
      { keyframeImages: false, keyframePoseSnap: false, trimPredictionTrajectory: true, trimPhaseDebug: true },
    ];

    for (const stage of stages) {
      const compact = buildStorageCompact(result, stage);
      const compactJson = JSON.stringify(compact);
      if (compactJson.length <= MAX_RESULT_JSON_LEN) return compactJson;
    }

    const lastCompact = buildStorageCompact(result, stages[stages.length - 1]);
    const shrunk = shrinkCompactForCap(lastCompact);
    return JSON.stringify(shrunk);
  } catch {
    return fallback;
  }
}

/**
 * Ensure the analyses table exists with the correct schema.
 *
 * The original d1_schema.sql created:
 *   type TEXT CHECK(type IN ('lite','pro'))  — blocks 'plus' inserts
 *   no total_score column
 *
 * Safe migration order:
 * 1. Recover from analyses_v1_backup if a previous partial migration left it.
 * 2. Create the table if it doesn't exist at all.
 * 3. Add missing columns (idempotent ALTER TABLE).
 * 4. If the live table still has the old CHECK constraint, migrate it using
 *    D1's db.batch() — which is fully atomic (entire batch rolls back on
 *    any failure, so analyses is never left empty or renamed without a
 *    replacement).
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function ensureSchema(db: any) {
  // ── 1. Recover from a previous failed migration ───────────────────────────
  try {
    const backupCheck = await db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='analyses_v1_backup'")
      .first();
    if (backupCheck?.name) {
      await db.prepare(
        "CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, user_id TEXT, video_url TEXT NOT NULL DEFAULT '', type TEXT, result_json TEXT, total_score INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
      ).run();
      await db.prepare(
        "INSERT OR IGNORE INTO analyses (id, user_id, video_url, type, result_json, total_score, created_at) SELECT id, user_id, COALESCE(video_url,''), type, result_json, COALESCE(total_score,0), created_at FROM analyses_v1_backup"
      ).run();
      await db.prepare("DROP TABLE analyses_v1_backup").run();
    }
  } catch { /* no backup to recover — continue */ }

  // ── 2. Create table if missing ────────────────────────────────────────────
  try {
    await db.prepare(
      "CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, user_id TEXT, video_url TEXT NOT NULL DEFAULT '', type TEXT, result_json TEXT, total_score INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
    ).run();
  } catch { /* table already exists */ }

  // ── 3. Add missing columns (idempotent) ───────────────────────────────────
  try {
    await db.prepare("ALTER TABLE analyses ADD COLUMN total_score INTEGER DEFAULT 0").run();
  } catch { /* column already exists */ }
  try {
    await db.prepare("ALTER TABLE analyses ADD COLUMN result_r2_key TEXT DEFAULT ''").run();
  } catch { /* column already exists */ }
  try {
    await db.prepare("ALTER TABLE analyses ADD COLUMN video_r2_key TEXT DEFAULT ''").run();
  } catch { /* column already exists */ }

  // ── 4. Fix old CHECK constraint using atomic batch ────────────────────────
  // If the table was created from d1_schema.sql it has CHECK(type IN ('lite','pro'))
  // which blocks 'plus' inserts.  db.batch() is transactional in D1: if any
  // step fails the entire batch rolls back, so analyses is never left empty.
  try {
    const row = await db
      .prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name='analyses'")
      .first();
    const ddl: string = (row?.sql as string) || "";
    if (ddl.includes("CHECK") && !ddl.toLowerCase().includes("'plus'")) {
      await db.batch([
        db.prepare("ALTER TABLE analyses RENAME TO analyses_v1_backup"),
        db.prepare(
          "CREATE TABLE analyses (id TEXT PRIMARY KEY, user_id TEXT, video_url TEXT NOT NULL DEFAULT '', type TEXT, result_json TEXT, total_score INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        ),
        db.prepare(
          "INSERT OR IGNORE INTO analyses (id, user_id, video_url, type, result_json, total_score, created_at) SELECT id, user_id, COALESCE(video_url,''), type, result_json, COALESCE(total_score,0), created_at FROM analyses_v1_backup"
        ),
        db.prepare("DROP TABLE analyses_v1_backup"),
      ]);
    }
  } catch { /* sqlite_master not accessible or batch failed — data safe */ }
}

async function saveJsonToR2(
  r2: { put: (key: string, value: string, options?: { httpMetadata?: { contentType?: string } }) => Promise<unknown> } | null,
  key: string,
  json: string
): Promise<boolean> {
  if (!r2) return false;
  try {
    await r2.put(key, json, { httpMetadata: { contentType: "application/json" } });
    return true;
  } catch {
    return false;
  }
}

export async function GET(request: NextRequest) {
  const userId = await getUserId(request);
  if (!userId) {
    return NextResponse.json({ detail: "未登录" }, { status: 401 });
  }

  const db = getDB();
  if (!db) {
    return NextResponse.json({ detail: "数据库不可用" }, { status: 503 });
  }

  await ensureSchema(db);

  const retentionDays = resolveHistoryRetentionDays(cfBindingEnv());
  await purgeExpiredHistoryForUser(db, getR2(), userId, historyRetentionCutoffIso(retentionDays));

  const { searchParams } = new URL(request.url);
  const limit = Math.min(parseInt(searchParams.get("limit") || "100"), 200);

  try {
    const analyses = await db
      .prepare(
        "SELECT id, type, video_url, video_r2_key, result_r2_key, total_score, result_json, created_at FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
      )
      .bind(userId, limit)
      .all();

    const trend = await db
      .prepare(
        "SELECT id, type, total_score, created_at FROM analyses WHERE user_id = ? ORDER BY created_at ASC LIMIT 30"
      )
      .bind(userId)
      .all();

    // Also include Shot Lab completed jobs so all record types are visible
    // in a single history view — lab records have type='lab' with no score/result_json.
    let labRecords: unknown[] = [];
    try {
      const labResult = await db
        .prepare(
          "SELECT id, 'lab' AS type, '' AS video_url, 0 AS total_score, '' AS result_json, created_at FROM lab_jobs WHERE user_id = ? AND status = 'completed' ORDER BY created_at DESC LIMIT 50"
        )
        .bind(userId)
        .all();
      labRecords = labResult.results || [];
    } catch { /* lab_jobs table may not exist yet */ }

    return NextResponse.json({
      analyses: analyses.results || [],
      lab_records: labRecords,
      trend: trend.results || [],
    });
  } catch (e) {
    return NextResponse.json(
      { detail: `查询失败: ${e instanceof Error ? e.message : "未知错误"}` },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  const userId = await getUserId(request);
  if (!userId) {
    return NextResponse.json({ detail: "未登录" }, { status: 401 });
  }

  const db = getDB();
  if (!db) {
    return NextResponse.json({ detail: "数据库不可用" }, { status: 503 });
  }

  await ensureSchema(db);

  const retentionDays = resolveHistoryRetentionDays(cfBindingEnv());
  await purgeExpiredHistoryForUser(db, getR2(), userId, historyRetentionCutoffIso(retentionDays));

  try {
    const body = await request.json();

    // Batch sync: accept an array of records
    if (Array.isArray(body.records)) {
      let synced = 0;
      let skipped = 0;
      const r2 = getR2();
      for (const rec of body.records.slice(0, 100)) {
        const id = rec.id || rec.analysis_id;
        const type = rec.type;
        const totalScore = normalizedTotalScoreForStorage(rec.total_score);
        const createdAt = rec.created_at || new Date().toISOString();

        if (!id || !type) { skipped++; continue; }

        const validTypes = ["lite", "pro", "plus", "lab"];
        if (!validTypes.includes(type)) { skipped++; continue; }

        const { video_url: vUrl, video_r2_key: vKey } = extractVideoFieldsFromSyncRecord(
          rec as Record<string, unknown>,
        );

        // Parse the raw result — it may contain full keyframes + pose data
        const rawJson = typeof rec.result_json === "string"
          ? rec.result_json
          : JSON.stringify(rec.result ?? rec.result_json ?? {});
        let resultJson = rawJson;
        let resultR2Key = "";

        if (rawJson.length > MAX_RESULT_JSON_LEN) {
          resultJson = compactResultForStorage(rawJson);
          const key = `results/${userId}/${id}.json`;
          const saved = await saveJsonToR2(r2, key, rawJson);
          if (saved) resultR2Key = key;
        }

        try {
          await db
            .prepare(
              "INSERT OR IGNORE INTO analyses (id, user_id, video_url, video_r2_key, result_r2_key, type, result_json, total_score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(id, userId, vUrl, vKey, resultR2Key, type, resultJson, totalScore, createdAt)
            .run();
          // Backfill video fields on existing rows (e.g. first sync used empty placeholders).
          if (vKey) {
            await db
              .prepare(
                "UPDATE analyses SET video_r2_key = ? WHERE id = ? AND user_id = ? AND (TRIM(COALESCE(video_r2_key, '')) = '')",
              )
              .bind(vKey, id, userId)
              .run();
          }
          if (vUrl) {
            await db
              .prepare(
                "UPDATE analyses SET video_url = ? WHERE id = ? AND user_id = ? AND (TRIM(COALESCE(video_url, '')) = '')",
              )
              .bind(vUrl, id, userId)
              .run();
          }
          synced++;
        } catch {
          skipped++;
        }
      }
      return NextResponse.json({ success: true, synced, skipped });
    }

    // Single record save
    const { analysis_id, type, total_score: total_score_raw, result, video_url, video_r2_key } = body;
    const total_score = normalizedTotalScoreForStorage(total_score_raw);

    if (!analysis_id || !type || !result) {
      return NextResponse.json({ detail: "缺少必要字段" }, { status: 400 });
    }

    const validTypes = ["lite", "pro", "plus", "lab"];
    if (!validTypes.includes(type)) {
      return NextResponse.json({ detail: "无效的分析类型" }, { status: 400 });
    }

    const rawResultJson = JSON.stringify({ ...(result || {}) });
    let resultJson = rawResultJson;
    let resultR2Key = "";
    if (rawResultJson.length > MAX_RESULT_JSON_LEN) {
      resultJson = compactResultForStorage(result);
      const r2 = getR2();
      const key = `results/${userId}/${analysis_id}.json`;
      const saved = await saveJsonToR2(r2, key, rawResultJson);
      if (saved) resultR2Key = key;
    }

    try {
      await db
        .prepare(
          "INSERT OR REPLACE INTO analyses (id, user_id, video_url, video_r2_key, result_r2_key, type, result_json, total_score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(
          analysis_id,
          userId,
          typeof video_url === "string" ? video_url : "",
          typeof video_r2_key === "string" ? video_r2_key : "",
          resultR2Key,
          type,
          resultJson,
          total_score,
          new Date().toISOString()
        )
        .run();
    } catch (insertErr) {
      // If the table has an old CHECK constraint that rejects the type,
      // the error will surface here. We return success=false so the client
      // knows to keep the local record for retry.
      return NextResponse.json(
        { success: false, detail: `保存失败: ${insertErr instanceof Error ? insertErr.message : "约束错误"}` },
        { status: 200 }
      );
    }

    return NextResponse.json({ success: true, id: analysis_id });
  } catch (e) {
    return NextResponse.json(
      { detail: `保存失败: ${e instanceof Error ? e.message : "未知错误"}` },
      { status: 500 }
    );
  }
}
