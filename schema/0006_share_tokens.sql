-- Share tokens table — enables public, no-login access to a single analysis record
-- Run: wrangler d1 execute stellar-golf-db --file=./schema/0006_share_tokens.sql

CREATE TABLE IF NOT EXISTS share_tokens (
  token      TEXT PRIMARY KEY,
  analysis_id TEXT NOT NULL,
  user_id    TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_share_tokens_analysis ON share_tokens(analysis_id);
CREATE INDEX IF NOT EXISTS idx_share_tokens_user     ON share_tokens(user_id);
