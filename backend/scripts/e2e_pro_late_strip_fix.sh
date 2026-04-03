#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10000}"
API_BASE="http://${HOST}:${PORT}"

if [[ -z "$VIDEO_PATH" ]]; then
  echo "Usage: bash backend/scripts/e2e_pro_late_strip_fix.sh /path/to/video.mp4"
  exit 1
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found: $VIDEO_PATH"
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

HOST="$HOST" PORT="$PORT" bash backend/scripts/run_pro_late_strip_fix.sh >/tmp/pro_late_strip_fix.log 2>&1 &
SERVER_PID=$!

echo "Started patched old pro route server (pid=$SERVER_PID)"

for _ in $(seq 1 30); do
  if curl -fsS "$API_BASE/health" >/dev/null 2>&1; then
    echo "Health OK: $API_BASE/health"
    break
  fi
  sleep 1
done

curl -fsS -X POST "$API_BASE/stellar-pro/analyze" -F "file=@${VIDEO_PATH}"
echo

echo "Server log tail:"
tail -n 80 /tmp/pro_late_strip_fix.log || true
