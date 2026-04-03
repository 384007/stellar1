#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10000}"
API_BASE="http://${HOST}:${PORT}"

if [[ -z "$VIDEO_PATH" ]]; then
  echo "Usage: bash backend/scripts/e2e_stellar_pro_api.sh /path/to/video.mp4"
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

HOST="$HOST" PORT="$PORT" bash backend/scripts/run_stellar_pro_api.sh >/tmp/stellar_pro_api.log 2>&1 &
SERVER_PID=$!

echo "Started main:app (pid=$SERVER_PID)"

for _ in $(seq 1 30); do
  if curl -fsS "$API_BASE/health" >/dev/null 2>&1; then
    echo "Health OK: $API_BASE/health"
    break
  fi
  sleep 1
done

curl -fsS "$API_BASE/health"
echo
bash backend/scripts/smoke_test_stellar_pro_api.sh "$VIDEO_PATH"
echo

echo "==> POST /stellar-pro/analyze (full chain; long timeout)"
curl -fsS --max-time 1200 -X POST "$API_BASE/stellar-pro/analyze" \
  -F "file=@${VIDEO_PATH}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('analyze_ok', 'analysis_id' in d, 'n_keyframes=', len(d.get('keyframes') or []), 'has_contact=', bool(d.get('contact_sheet_url')))"

echo
echo "Server log tail:"
tail -n 60 /tmp/stellar_pro_api.log || true
