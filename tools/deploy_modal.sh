#!/usr/bin/env bash
# Deploy stellar-ai to Modal from your machine (no GitHub Actions).
# Prerequisites:
#   1) Create repo-root .modal.env.local (gitignored) with:
#        MODAL_TOKEN_ID=...
#        MODAL_TOKEN_SECRET=...
#      (from https://modal.com/settings → API tokens)
#   OR run `modal token new` once so ~/.modal is populated — then this script still works
#      if those env vars are already exported in your shell.
#   2) Python 3 with pip.

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
  echo "Missing MODAL_TOKEN_ID / MODAL_TOKEN_SECRET."
  echo "  Put them in $ROOT/.modal.env.local (see tools/deploy_modal.sh header), or run: modal token new"
  exit 1
fi

python3 -m pip install -q -U modal

# Baked into the Modal image and printed at worker startup (`modal_app.fastapi_app`).
# Modal Pro: STELLAR_MODAL_PRO_V3_ONLY=1 + STELLAR_RUNTIME=modal — POST /pro-v3/analyze only; no /pro-v2; no /stellar-pro/analyze.
export STELLAR_GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
export STELLAR_GIT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
export STELLAR_BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

exec python3 -m modal deploy "$ROOT/modal_app.py"
