#!/usr/bin/env bash
# Upload packed MediaPipe folder to R2 (YOU run this locally after `wrangler login`).
# Does not run in CI from this repo unless you explicitly invoke it.
#
#   cd frontend && npm install && node ../tools/mediapipe-pack-for-r2.mjs
#   export R2_BUCKET=stellar-golf-media   # optional, default below
#   bash tools/r2-upload-mediapipe.sh
#
# Requires: npx wrangler (or global wrangler) authenticated to your account.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/frontend/build/mediapipe-r2/0.10.33"
BUCKET="${R2_BUCKET:-stellar-golf-media}"
PREFIX="static/mediapipe/tasks-vision/0.10.33"

if [[ ! -d "$SRC" ]]; then
  echo "Missing $SRC — run first: cd frontend && node ../tools/mediapipe-pack-for-r2.mjs"
  exit 1
fi

content_type() {
  case "${1##*.}" in
    mjs) echo "application/javascript" ;;
    js)  echo "application/javascript" ;;
    wasm) echo "application/wasm" ;;
    task) echo "application/octet-stream" ;;
    *) echo "application/octet-stream" ;;
  esac
}

echo "Uploading to r2://$BUCKET/$PREFIX/ from $SRC"
while IFS= read -r -d '' f; do
  rel="${f#"$SRC"/}"
  key="$PREFIX/$rel"
  ct="$(content_type "$f")"
  echo "  $key ($ct)"
  npx wrangler r2 object put "$BUCKET/$key" --file="$f" --content-type="$ct"
done < <(find "$SRC" -type f -print0)

echo
echo "Done. Next:"
echo "  1) R2 → bucket → Public access / CORS for your Pages domain"
echo "  2) Pages → NEXT_PUBLIC_MEDIAPIPE_CDN_BASE=https://<public-host>/$PREFIX"
echo "  3) Redeploy Pages"
