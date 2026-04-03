#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-}"

if [[ -z "$VIDEO_PATH" ]]; then
  echo "Usage: bash backend/scripts/merge_test_stellar_pro_api.sh /path/to/video.mp4"
  exit 1
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found: $VIDEO_PATH"
  exit 1
fi

echo "==> Verify stellar_pro stack"
bash backend/scripts/verify_stellar_pro_stack.sh

echo
echo "==> Run e2e stellar_pro test"
bash backend/scripts/e2e_stellar_pro_api.sh "$VIDEO_PATH"
