#!/usr/bin/env bash
# Deploy **lite-only** Modal app (1 CPU / 4G / 900s). Same secrets as main (`custom-secret`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.modal.env.local" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT/.modal.env.local"
  set +a
fi

if [[ -z "${MODAL_TOKEN_ID:-}" || -z "${MODAL_TOKEN_SECRET:-}" ]]; then
  echo "Missing MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (see tools/deploy_modal.sh)"
  exit 1
fi

python3 -m pip install -q -U modal
export STELLAR_GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
export STELLAR_GIT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
export STELLAR_BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

exec python3 -m modal deploy "$ROOT/modal_app_lite.py"
