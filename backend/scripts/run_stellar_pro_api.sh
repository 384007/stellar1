#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10000}"

exec uvicorn main:app --host "$HOST" --port "$PORT"
