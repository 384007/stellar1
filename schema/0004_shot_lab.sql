-- Shot Lab tables (auto-created by ensureLabSchema at runtime, but documented here)
-- Run: wrangler d1 execute stellar-golf-db --file=./schema/0004_shot_lab.sql

-- 1. Lab analysis jobs
CREATE TABLE IF NOT EXISTS lab_jobs (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  tier TEXT NOT NULL DEFAULT 'free',
  video_r2_key TEXT DEFAULT '',
  result_json TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lab_jobs_user ON lab_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_lab_jobs_ts ON lab_jobs(created_at);

-- 2. Daily usage tracking for Shot Lab (independent from plus_usage)
CREATE TABLE IF NOT EXISTS lab_usage_daily (
  user_id TEXT NOT NULL,
  usage_date TEXT NOT NULL,
  count INTEGER DEFAULT 0,
  PRIMARY KEY (user_id, usage_date)
);

-- 3. Idempotent quota log (prevents double-charge on job retry)
CREATE TABLE IF NOT EXISTS lab_quota_log (
  job_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  usage_date TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
