#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:10000}"
VIDEO_PATH="${1:-}"
ROUGH_IMPACT_TIME_S="${ROUGH_IMPACT_TIME_S:-}"

if [[ -z "$VIDEO_PATH" ]]; then
  echo "Usage: bash backend/scripts/smoke_test_stellar_pro_api.sh /path/to/video.mp4"
  exit 1
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found: $VIDEO_PATH"
  exit 1
fi

extra=()
if [[ -n "$ROUGH_IMPACT_TIME_S" ]]; then
  extra+=( -F "rough_impact_time_s=${ROUGH_IMPACT_TIME_S}" )
fi

curl -sS --max-time 1200 -X POST "$API_BASE/stellar-pro/analyze" \
  -F "file=@${VIDEO_PATH}" \
  "${extra[@]}"

echo
