-- Stellar AI - Cloudflare D1 Database Schema
-- Run this SQL against your D1 database to initialize tables.
-- Command: wrangler d1 execute stellar-golf-db --file=./schema/d1_schema.sql

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  is_pro BOOLEAN DEFAULT FALSE,
  daily_count INTEGER DEFAULT 0,
  last_reset DATE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES users(id),
  video_url TEXT NOT NULL,
  type TEXT CHECK(type IN ('lite','pro')),
  result_json TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS news_cache (
  id TEXT PRIMARY KEY,
  data_json TEXT,
  cached_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_news_cache_cached_at ON news_cache(cached_at);
