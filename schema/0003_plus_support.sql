-- Plus feature support migration
-- Adds 'plus' type to analyses and creates usage tracking table.
-- Run: wrangler d1 execute stellar-golf-db --file=./schema/0003_plus_support.sql

-- 1. Recreate analyses table to accept 'plus' type (SQLite cannot ALTER CHECK)
CREATE TABLE IF NOT EXISTS analyses_v2 (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES users(id),
  video_url TEXT NOT NULL DEFAULT '',
  type TEXT CHECK(type IN ('lite','pro','plus')),
  result_json TEXT,
  total_score INTEGER DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO analyses_v2
  SELECT id, user_id, COALESCE(video_url,''), type, result_json,
         COALESCE(total_score,0), created_at
  FROM analyses;

DROP TABLE IF EXISTS analyses;
ALTER TABLE analyses_v2 RENAME TO analyses;

CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);

-- 2. Plus daily usage tracking (independent table, no ALTER needed)
CREATE TABLE IF NOT EXISTS plus_usage (
  user_id TEXT NOT NULL,
  usage_date TEXT NOT NULL,
  count INTEGER DEFAULT 0,
  PRIMARY KEY (user_id, usage_date)
);
