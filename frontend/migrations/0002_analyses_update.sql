-- Update analyses table for history feature
-- video_url: default empty string (we don't persist uploaded videos)
-- total_score: extracted from result_json for fast trend queries

ALTER TABLE analyses ADD COLUMN total_score INTEGER DEFAULT 0;

-- Allow video_url to be empty (SQLite doesn't support ALTER COLUMN,
-- but the default NOT NULL was only in the CREATE TABLE, and SQLite
-- accepts empty strings, so new inserts just use '' as default)
