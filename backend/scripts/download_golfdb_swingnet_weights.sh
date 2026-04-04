#!/usr/bin/env bash
# Download official SwingNet weights (wmcnally/golfdb) into backend/models/.
#
# Upstream README (Evaluate section):
#   https://github.com/wmcnally/golfdb
# Google Drive file id from that README — license: CC BY-NC 4.0 (non-commercial).
#
# Requires: pip install gdown   OR   python3 -m pip install gdown
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/models/swingnet_1800.pth.tar"
FILE_ID="1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW"
URL="https://drive.google.com/uc?id=${FILE_ID}"

mkdir -p "${ROOT}/models"

if [[ -f "$DEST" ]]; then
  echo "Already present: $DEST"
  exit 0
fi

# Prefer backend/.venv/bin/gdown (repo venv) so PATH does not need activation.
GDOWN_BIN=""
if [[ -x "${ROOT}/.venv/bin/gdown" ]]; then
  GDOWN_BIN="${ROOT}/.venv/bin/gdown"
elif command -v gdown >/dev/null 2>&1; then
  GDOWN_BIN="gdown"
fi

if [[ -n "$GDOWN_BIN" ]]; then
  "$GDOWN_BIN" "$URL" -O "$DEST"
elif [[ -x "${ROOT}/.venv/bin/python3" ]] && "${ROOT}/.venv/bin/python3" -c "import gdown" 2>/dev/null; then
  "${ROOT}/.venv/bin/python3" -m gdown "$URL" -O "$DEST"
elif python3 -c "import gdown" 2>/dev/null; then
  python3 -m gdown "$URL" -O "$DEST"
else
  echo "Install gdown into backend/.venv:"
  echo "  cd \"${ROOT}\" && python3 -m venv .venv && .venv/bin/pip install gdown"
  echo "Or: pip install gdown"
  echo "Manual download: https://drive.google.com/file/d/${FILE_ID}/view"
  exit 1
fi

echo "OK: $DEST"
ls -la "$DEST"
