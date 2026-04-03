-- Shot Lab v2: subscription, enhanced jobs, compare/trend support
-- Run: wrangler d1 execute stellar-golf-db --file=./schema/0007_lab_v2.sql

-- 1. User entitlements (isolated from core `users` table)
CREATE TABLE IF NOT EXISTS lab_user_entitlements (
  user_id TEXT PRIMARY KEY,
  plan TEXT NOT NULL DEFAULT 'free',
  subscription_status TEXT NOT NULL DEFAULT 'active',
  pro_expires_at DATETIME,
  entitlements_version INTEGER DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Extend lab_jobs with richer metadata (additive only, no column drops)
--    input_type: upload | capture | screen
--    club_type:  driver | iron | wedge | putter | unknown
--    summary_snippet: first ~80 chars of AI summary for history cards
ALTER TABLE lab_jobs ADD COLUMN input_type TEXT DEFAULT 'upload';
ALTER TABLE lab_jobs ADD COLUMN club_type TEXT DEFAULT 'unknown';
ALTER TABLE lab_jobs ADD COLUMN summary_snippet TEXT DEFAULT '';
