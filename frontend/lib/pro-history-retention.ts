/**
 * 全部分析历史（Lite / Pro / Plus / Lab）：默认保留 30 天，到期从 D1 删除并删除关联 R2。
 * 可选绑定/环境变量：STELLAR_HISTORY_RETENTION_DAYS 或 HISTORY_RETENTION_DAYS（1–3650）。
 *
 * Edge 路由里传入 ``getRequestContext().env``；客户端仅可导入 prune 与常量。
 */

export const DEFAULT_HISTORY_RETENTION_DAYS = 30;

export function resolveHistoryRetentionDays(bindingEnv?: Record<string, unknown>): number {
  const candidates = [
    bindingEnv?.STELLAR_HISTORY_RETENTION_DAYS,
    bindingEnv?.HISTORY_RETENTION_DAYS,
    process.env.STELLAR_HISTORY_RETENTION_DAYS,
    process.env.HISTORY_RETENTION_DAYS,
  ];
  for (const c of candidates) {
    const n = parseInt(String(c ?? "").trim(), 10);
    if (Number.isFinite(n) && n >= 1 && n <= 3650) return n;
  }
  return DEFAULT_HISTORY_RETENTION_DAYS;
}

export function historyRetentionCutoffIso(retentionDays: number): string {
  const ms = Math.max(1, retentionDays) * 24 * 60 * 60 * 1000;
  return new Date(Date.now() - ms).toISOString();
}

type AnalysisRow = { id: string; video_r2_key: string | null; result_r2_key: string | null };

/**
 * 删除该用户 analyses 中早于 cutoff 的全部类型；删 R2、share_tokens、D1。
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function purgeExpiredAnalysesForUser(db: any, r2: any, userId: string, cutoffIso: string): Promise<number> {
  const res = await db
    .prepare(
      "SELECT id, video_r2_key, result_r2_key FROM analyses WHERE user_id = ? AND created_at < ?",
    )
    .bind(userId, cutoffIso)
    .all();
  const rows = (res?.results || []) as AnalysisRow[];
  if (rows.length === 0) return 0;

  for (const row of rows) {
    const vk = String(row.video_r2_key || "").trim();
    const rk = String(row.result_r2_key || "").trim();
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
  }

  for (const row of rows) {
    try {
      await db
        .prepare("DELETE FROM share_tokens WHERE analysis_id = ? AND user_id = ?")
        .bind(row.id, userId)
        .run();
    } catch {
      /* share_tokens 可能不存在 */
    }
  }

  await db.prepare("DELETE FROM analyses WHERE user_id = ? AND created_at < ?").bind(userId, cutoffIso).run();

  return rows.length;
}

type LabRow = { id: string; video_r2_key: string | null };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function purgeExpiredLabJobsForUser(db: any, r2: any, userId: string, cutoffIso: string): Promise<number> {
  let labRows: LabRow[] = [];
  try {
    const res = await db
      .prepare("SELECT id, video_r2_key FROM lab_jobs WHERE user_id = ? AND created_at < ?")
      .bind(userId, cutoffIso)
      .all();
    labRows = (res?.results || []) as LabRow[];
  } catch {
    return 0;
  }
  if (labRows.length === 0) return 0;

  for (const row of labRows) {
    const vk = String(row.video_r2_key || "").trim();
    if (r2 && vk) {
      try {
        await r2.delete(vk);
      } catch {
        /* ignore */
      }
    }
  }

  try {
    await db.prepare("DELETE FROM lab_jobs WHERE user_id = ? AND created_at < ?").bind(userId, cutoffIso).run();
  } catch {
    return 0;
  }

  return labRows.length;
}

/** analyses（Lite/Pro/Plus）+ Shot Lab jobs，同一保留窗口 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function purgeExpiredHistoryForUser(db: any, r2: any, userId: string, cutoffIso: string): Promise<void> {
  await purgeExpiredAnalysesForUser(db, r2, userId, cutoffIso);
  await purgeExpiredLabJobsForUser(db, r2, userId, cutoffIso);
}

const LOCAL_KEY = "stellar_history_local";

/**
 * 浏览器端：移除 localStorage 里超过保留期的记录（所有 type）。
 */
export function pruneLocalStellarHistoryRecords(retentionDays: number = DEFAULT_HISTORY_RETENTION_DAYS): void {
  if (typeof window === "undefined" || !window.localStorage) return;
  const cutoff = Date.now() - Math.max(1, retentionDays) * 24 * 60 * 60 * 1000;
  try {
    const raw = localStorage.getItem(LOCAL_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return;
    const arr = parsed as Array<{ created_at?: string }>;
    const next = arr.filter((r) => {
      const t = Date.parse(String(r.created_at || ""));
      // 无法解析日期的条目保留，避免误删刚写入或旧版无字段的记录
      if (!Number.isFinite(t)) return true;
      return t >= cutoff;
    });
    if (next.length !== arr.length) localStorage.setItem(LOCAL_KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
}
