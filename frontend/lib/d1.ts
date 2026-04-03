/**
 * Cloudflare D1 database helpers.
 * These functions are designed to work with Cloudflare Workers/Pages runtime.
 * When deployed to Cloudflare Pages, the DB binding is available through the platform context.
 */

export interface D1Database {
  prepare(query: string): D1PreparedStatement;
  batch<T>(statements: D1PreparedStatement[]): Promise<D1Result<T>[]>;
  exec(query: string): Promise<D1ExecResult>;
}

export interface D1PreparedStatement {
  bind(...values: unknown[]): D1PreparedStatement;
  first<T>(column?: string): Promise<T | null>;
  run(): Promise<D1Result<unknown>>;
  all<T>(): Promise<D1Result<T>>;
}

export interface D1Result<T> {
  results: T[];
  success: boolean;
  meta: Record<string, unknown>;
}

export interface D1ExecResult {
  count: number;
  duration: number;
}

export interface User {
  id: string;
  email: string;
  username: string;
  password_hash: string;
  is_pro: boolean;
  daily_count: number;
  last_reset: string;
  created_at: string;
}

export interface Analysis {
  id: string;
  user_id: string;
  video_url: string;
  type: "lite" | "pro" | "plus";
  result_json: string;
  total_score: number;
  created_at: string;
}

export interface PlusUsage {
  user_id: string;
  usage_date: string;
  count: number;
}

export async function getUserByEmail(
  db: D1Database,
  email: string
): Promise<User | null> {
  return db
    .prepare("SELECT * FROM users WHERE email = ?")
    .bind(email)
    .first<User>();
}

export async function getUserById(
  db: D1Database,
  id: string
): Promise<User | null> {
  return db
    .prepare("SELECT * FROM users WHERE id = ?")
    .bind(id)
    .first<User>();
}

export async function createUser(
  db: D1Database,
  user: Omit<User, "created_at">
): Promise<void> {
  await db
    .prepare(
      "INSERT INTO users (id, email, password_hash, is_pro, daily_count, last_reset) VALUES (?, ?, ?, ?, ?, ?)"
    )
    .bind(
      user.id,
      user.email,
      user.password_hash,
      user.is_pro ? 1 : 0,
      user.daily_count,
      user.last_reset
    )
    .run();
}

export async function updateUserDailyCount(
  db: D1Database,
  userId: string,
  count: number,
  lastReset: string
): Promise<void> {
  await db
    .prepare(
      "UPDATE users SET daily_count = ?, last_reset = ? WHERE id = ?"
    )
    .bind(count, lastReset, userId)
    .run();
}

export async function upgradeUserToPro(
  db: D1Database,
  userId: string
): Promise<void> {
  await db
    .prepare("UPDATE users SET is_pro = 1 WHERE id = ?")
    .bind(userId)
    .run();
}

export async function saveAnalysis(
  db: D1Database,
  analysis: {
    id: string;
    user_id: string;
    type: "lite" | "pro" | "plus";
    result_json: string;
    total_score: number;
  }
): Promise<void> {
  await db
    .prepare(
      "INSERT INTO analyses (id, user_id, video_url, type, result_json, total_score) VALUES (?, ?, ?, ?, ?, ?)"
    )
    .bind(
      analysis.id,
      analysis.user_id,
      "",
      analysis.type,
      analysis.result_json,
      analysis.total_score
    )
    .run();
}

export async function getUserAnalyses(
  db: D1Database,
  userId: string,
  limit: number = 50
): Promise<Analysis[]> {
  const result = await db
    .prepare(
      "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
    )
    .bind(userId, limit)
    .all<Analysis>();
  return result.results;
}

export interface AnalysisSummary {
  id: string;
  type: string;
  total_score: number;
  created_at: string;
}

export async function getUserAnalysisTrend(
  db: D1Database,
  userId: string,
  limit: number = 30
): Promise<AnalysisSummary[]> {
  const result = await db
    .prepare(
      "SELECT id, type, total_score, created_at FROM analyses WHERE user_id = ? ORDER BY created_at ASC LIMIT ?"
    )
    .bind(userId, limit)
    .all<AnalysisSummary>();
  return result.results;
}

// ── Plus usage tracking ──

export async function ensurePlusUsageTable(db: D1Database): Promise<void> {
  await db.exec(
    "CREATE TABLE IF NOT EXISTS plus_usage (user_id TEXT NOT NULL, usage_date TEXT NOT NULL, count INTEGER DEFAULT 0, PRIMARY KEY (user_id, usage_date))"
  );
}

export async function getPlusUsageToday(
  db: D1Database,
  userId: string
): Promise<number> {
  const today = new Date().toISOString().slice(0, 10);
  const row = await db
    .prepare("SELECT count FROM plus_usage WHERE user_id = ? AND usage_date = ?")
    .bind(userId, today)
    .first<{ count: number }>();
  return row?.count ?? 0;
}

export async function incrementPlusUsage(
  db: D1Database,
  userId: string
): Promise<number> {
  const today = new Date().toISOString().slice(0, 10);
  await db
    .prepare(
      "INSERT INTO plus_usage (user_id, usage_date, count) VALUES (?, ?, 1) ON CONFLICT(user_id, usage_date) DO UPDATE SET count = count + 1"
    )
    .bind(userId, today)
    .run();
  return getPlusUsageToday(db, userId);
}
