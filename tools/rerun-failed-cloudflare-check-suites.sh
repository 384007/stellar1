#!/usr/bin/env bash
# Bulk rerequest GitHub Check Suites that failed for Cloudflare (Pages) on recent main commits.
# Requires: `gh auth login` with a token that can rerequest check suites (repo scope).
# Does not touch GitHub Actions workflows — only Check Suites (e.g. Cloudflare Pages app).
set -euo pipefail

OWNER="dytsui"
REPO="stellar1"
DEPTH="${DEPTH:-80}"
# Scan this branch's history (default: main). Override: BASE_BRANCH=main ./script.sh
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
    cs = r.get('check_suite') or {}
    sid = cs.get('id')
    if sid:
        print(sid)
" 2>/dev/null >>"$TMP" || true
done

sort -u "$TMP" | while read -r sid; do
  [[ -z "$sid" ]] && continue
  echo "Rerequest check_suite=$sid"
  if gh api --silent -X POST "repos/${OWNER}/${REPO}/check-suites/${sid}/rerequest"; then
    echo "  ok"
  else
    echo "  failed (suite may be too old or app may ignore rerequest)"
  fi
  sleep 0.4
done

echo "Done."
