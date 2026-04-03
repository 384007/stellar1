/**
 * D1 persistence for manual / QR / crypto payment orders (pending until you confirm in admin).
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type DB = any;

export async function ensurePaymentSchema(db: DB): Promise<void> {
  try {
    await db.exec(`
      CREATE TABLE IF NOT EXISTS payment_orders (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        email TEXT DEFAULT '',
        channel TEXT NOT NULL,
        amount_hint TEXT DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending',
        user_memo TEXT NOT NULL,
        tx_hash TEXT DEFAULT '',
        admin_note TEXT DEFAULT '',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);
  } catch { /* exists */ }

  try {
    await db.exec("CREATE INDEX IF NOT EXISTS idx_payment_orders_user ON payment_orders(user_id)");
    await db.exec("CREATE INDEX IF NOT EXISTS idx_payment_orders_status ON payment_orders(status)");
    await db.exec("CREATE INDEX IF NOT EXISTS idx_payment_orders_created ON payment_orders(created_at)");
  } catch { /* ignore */ }
}

export async function createPaymentOrder(
  db: DB,
  row: {
    id: string;
    user_id: string;
    email: string;
    channel: string;
    amount_hint: string;
    user_memo: string;
  }
): Promise<void> {
  const now = new Date().toISOString();
  await db
    .prepare(
      `INSERT INTO payment_orders (id, user_id, email, channel, amount_hint, status, user_memo, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)`
    )
    .bind(row.id, row.user_id, row.email, row.channel, row.amount_hint, row.user_memo, now, now)
    .run();
}

export async function listPaymentOrdersForUser(
  db: DB,
  userId: string,
  limit: number
): Promise<Record<string, unknown>[]> {
  const r = await db
    .prepare(
      "SELECT id, channel, amount_hint, status, user_memo, created_at FROM payment_orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
    )
    .bind(userId, limit)
    .all();
  return r.results || [];
}

export async function listPaymentOrdersAdmin(
  db: DB,
  limit: number
): Promise<Record<string, unknown>[]> {
  const r = await db
    .prepare(
      "SELECT id, user_id, email, channel, amount_hint, status, user_memo, tx_hash, admin_note, created_at, updated_at FROM payment_orders ORDER BY created_at DESC LIMIT ?"
    )
    .bind(limit)
    .all();
  return r.results || [];
}

export async function financeSummary(db: DB): Promise<{
  pending: number;
  confirmed: number;
  by_channel: Record<string, number>;
}> {
  const pending = await db
    .prepare("SELECT COUNT(*) as c FROM payment_orders WHERE status = 'pending'")
    .first();
  const confirmed = await db
    .prepare("SELECT COUNT(*) as c FROM payment_orders WHERE status = 'confirmed'")
    .first();
  const rows = await db.prepare("SELECT channel, status, COUNT(*) as c FROM payment_orders GROUP BY channel, status").all();
  const by_channel: Record<string, number> = {};
  for (const row of rows.results || []) {
    const ch = String((row as { channel: string }).channel);
    const st = String((row as { status: string }).status);
    const c = Number((row as { c: number }).c);
    const k = `${ch}:${st}`;
    by_channel[k] = c;
  }
  return {
    pending: Number((pending as { c?: number })?.c ?? 0),
    confirmed: Number((confirmed as { c?: number })?.c ?? 0),
    by_channel,
  };
}
