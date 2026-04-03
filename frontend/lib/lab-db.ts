/**
 * D1 helpers for Shot Lab tables.
 * Follows the existing ensureSchema / auto-create pattern used by
 * api/history and api/plus/usage routes.
 *
 * Tables: lab_jobs, lab_usage_daily, lab_quota_log, lab_user_entitlements
 * All tables are independent from the legacy `analyses` table.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type DB = any;

export function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

/**
 * Create Shot Lab tables if they don't exist.
 * Uses prepare().run() (D1's recommended path) instead of exec() to avoid
 * silent failures on cold-start / first-deploy where exec() can throw and
 * leave tables uncreated.
 * Each statement is independent so one failure doesn't prevent the others.
 */
export async function ensureLabSchema(db: DB): Promise<void> {
  const stmts: string[] = [
    `CREATE TABLE IF NOT EXISTS lab_jobs (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending',
      tier TEXT NOT NULL DEFAULT 'free',
      video_r2_key TEXT DEFAULT '',
      result_json TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`,
    `CREATE TABLE IF NOT EXISTS lab_usage_daily (
      user_id TEXT NOT NULL,
      usage_date TEXT NOT NULL,
      count INTEGER DEFAULT 0,
      PRIMARY KEY (user_id, usage_date)
    )`,
    `CREATE TABLE IF NOT EXISTS lab_quota_log (
      job_id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      usage_date TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`,
    "CREATE INDEX IF NOT EXISTS idx_lab_jobs_user ON lab_jobs(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_lab_jobs_ts ON lab_jobs(created_at)",
    `CREATE TABLE IF NOT EXISTS lab_user_entitlements (
      user_id TEXT PRIMARY KEY,
      plan TEXT NOT NULL DEFAULT 'free',
      subscription_status TEXT NOT NULL DEFAULT 'active',
      pro_expires_at DATETIME,
      entitlements_version INTEGER DEFAULT 1,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`,
  ];

  for (const sql of stmts) {
    try {
      await db.prepare(sql).run();
    } catch (e) {
      // Log but continue — "already exists" errors are expected on non-first runs;
      // index creation errors are non-fatal. We'll surface table errors later.
      console.warn("[lab-db] ensureLabSchema:", e instanceof Error ? e.message : e);
    }
  }
}

// ── Usage / quota ──

export async function getLabUsageToday(db: DB, userId: string): Promise<number> {
  const today = todayUTC();
  const row = await db
    .prepare("SELECT count FROM lab_usage_daily WHERE user_id = ? AND usage_date = ?")
    .bind(userId, today)
    .first();
  return (row?.count as number) ?? 0;
}

/**
 * Idempotently increment daily lab usage for a given job.
 * If job_id was already counted, this is a no-op (prevents double-charge on retry).
 * Returns the updated daily count.
 */
export async function incrementLabUsage(db: DB, userId: string, jobId: string): Promise<number> {
  const today = todayUTC();

  const existing = await db
    .prepare("SELECT job_id FROM lab_quota_log WHERE job_id = ?")
    .bind(jobId)
    .first();

  if (existing) {
    return getLabUsageToday(db, userId);
  }

  await db
    .prepare("INSERT OR IGNORE INTO lab_quota_log (job_id, user_id, usage_date) VALUES (?, ?, ?)")
    .bind(jobId, userId, today)
    .run();

  await db
    .prepare(
      "INSERT INTO lab_usage_daily (user_id, usage_date, count) VALUES (?, ?, 1) ON CONFLICT(user_id, usage_date) DO UPDATE SET count = count + 1"
    )
    .bind(userId, today)
    .run();

  return getLabUsageToday(db, userId);
}

// ── Job CRUD ──

export async function createLabJob(
  db: DB,
  job: { id: string; user_id: string; tier: string; video_r2_key?: string }
): Promise<void> {
  const now = new Date().toISOString();
  await db
    .prepare(
      "INSERT INTO lab_jobs (id, user_id, status, tier, video_r2_key, created_at, updated_at) VALUES (?, ?, 'processing', ?, ?, ?, ?)"
    )
    .bind(job.id, job.user_id, job.tier, job.video_r2_key || "", now, now)
    .run();
}

/** Same job_id retry (failed / stuck processing): do not INSERT again — avoids D1 UNIQUE constraint error. */
export async function markLabJobRetryProcessing(
  db: DB,
  jobId: string,
  tier: string
): Promise<void> {
  const now = new Date().toISOString();
  await db
    .prepare(
      "UPDATE lab_jobs SET status = 'processing', tier = ?, updated_at = ? WHERE id = ?"
    )
    .bind(tier, now, jobId)
    .run();
}

export async function updateLabJobResult(
  db: DB,
  jobId: string,
  status: string,
  resultJson: string
): Promise<void> {
  const now = new Date().toISOString();
  await db
    .prepare("UPDATE lab_jobs SET status = ?, result_json = ?, updated_at = ? WHERE id = ?")
    .bind(status, resultJson, now, jobId)
    .run();
}

export async function getLabJob(db: DB, jobId: string): Promise<Record<string, unknown> | null> {
  return db
    .prepare("SELECT * FROM lab_jobs WHERE id = ?")
    .bind(jobId)
    .first();
}

export async function getLabHistory(
  db: DB,
  userId: string,
  opts: { maxItems: number; maxAgeDays?: number }
): Promise<Record<string, unknown>[]> {
  let query = "SELECT id, status, tier, created_at, updated_at FROM lab_jobs WHERE user_id = ?";
  const binds: unknown[] = [userId];

  if (opts.maxAgeDays) {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - opts.maxAgeDays);
    query += " AND created_at >= ?";
    binds.push(cutoff.toISOString());
  }

  query += " ORDER BY created_at DESC LIMIT ?";
  binds.push(opts.maxItems);

  const stmt = db.prepare(query);
  const result = await stmt.bind(...binds).all();
  return result.results || [];
}

// ── Compare: fetch two jobs for side-by-side ──

export async function getLabJobsForCompare(
  db: DB,
  userId: string,
  jobIds: [string, string]
): Promise<Record<string, unknown>[]> {
  const results: Record<string, unknown>[] = [];
  for (const id of jobIds) {
    const job = await db
      .prepare("SELECT * FROM lab_jobs WHERE id = ? AND user_id = ? AND status = 'completed'")
      .bind(id, userId)
      .first();
    if (job) results.push(job);
  }
  return results;
}

// ── Trend: fetch completed jobs with metrics for charting ──

export async function getLabTrendData(
  db: DB,
  userId: string,
  opts: { maxDays: number; maxPoints: number }
): Promise<Record<string, unknown>[]> {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - opts.maxDays);

  const result = await db
    .prepare(
      `SELECT id, result_json, created_at
       FROM lab_jobs
       WHERE user_id = ? AND status = 'completed' AND created_at >= ?
       ORDER BY created_at ASC
       LIMIT ?`
    )
    .bind(userId, cutoff.toISOString(), opts.maxPoints)
    .all();

  return result.results || [];
}

// ── Export: fetch a single completed job with full result ──

export async function getLabJobForExport(
  db: DB,
  userId: string,
  jobId: string
): Promise<Record<string, unknown> | null> {
  return db
    .prepare("SELECT * FROM lab_jobs WHERE id = ? AND user_id = ? AND status = 'completed'")
    .bind(jobId, userId)
    .first();
}

// ── Update job with summary snippet for history cards ──

export async function updateLabJobSummary(
  db: DB,
  jobId: string,
  snippet: string
): Promise<void> {
  try {
    await db
      .prepare("UPDATE lab_jobs SET summary_snippet = ? WHERE id = ?")
      .bind(snippet.slice(0, 80), jobId)
      .run();
  } catch {
    // v2 column may not exist yet; non-fatal
  }
}
