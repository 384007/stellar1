#!/usr/bin/env bash
# Bulk rerequest failed Cloudflare (Pages) check *runs* on recent commits (not check_suite — suite rerequest often 422 for third-party apps).
# Requires: `gh auth login` with checks:write / repo scope.
set -euo pipefail

OWNER="dytsui"
REPO="stellar1"
DEPTH="${DEPTH:-80}"
BASE_BRANCH="${BASE_BRANCH:-main}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI: https://cli.github.com/ then run: gh auth login"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Run: gh auth login"
  exit 1
fi

export GH_PAGER=cat
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

for sha in $(git log "${BASE_BRANCH}" --format=%H "-${DEPTH}"); do
  json=$(gh api "repos/${OWNER}/${REPO}/commits/${sha}/check-runs?per_page=100" 2>/dev/null || true)
  [[ -z "$json" ]] && continue
  echo "$json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for r in d.get('check_runs') or []:
    if r.get('conclusion') != 'failure':
        continue
    app = (r.get('app') or {}).get('name') or ''
    if 'cloudflare' not in app.lower():
        continue
    rid = r.get('id')
    if rid is not None:
        print(rid)
" 2>/dev/null >>"$TMP" || true
done

sort -u "$TMP" | while read -r rid; do
  [[ -z "$rid" ]] && continue
  echo "Rerequest check_run=$rid"
  if gh api --silent -X POST "repos/${OWNER}/${REPO}/check-runs/${rid}/rerequest"; then
    echo "  ok"
  else
    echo "  failed (run may be too old or token cannot rerequest this app)"
  fi
  sleep 0.4
done

echo "Done."
