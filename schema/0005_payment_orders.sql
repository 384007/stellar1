-- Payment orders (WeChat / Alipay / crypto — manual confirmation workflow)
-- Auto-created by ensurePaymentSchema in frontend/lib/payment-db.ts
-- Optional manual apply: wrangler d1 execute stellar-golf-db --file=./schema/0005_payment_orders.sql

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
);

CREATE INDEX IF NOT EXISTS idx_payment_orders_user ON payment_orders(user_id);
CREATE INDEX IF NOT EXISTS idx_payment_orders_status ON payment_orders(status);
CREATE INDEX IF NOT EXISTS idx_payment_orders_created ON payment_orders(created_at);
