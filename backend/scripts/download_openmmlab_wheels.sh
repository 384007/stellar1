#!/usr/bin/env bash
# Download official OpenMMLab mmcv CPU wheels via curl (large files — not committed; see .gitignore).
#
# Usage (from repo root or backend/):
#   chmod +x backend/scripts/download_openmmlab_wheels.sh
#   ./backend/scripts/download_openmmlab_wheels.sh [output_dir]
#
# Default output: backend/.openmmlab-wheels/
#
# Then install locally (pick ONE mmcv file for your OS + Python 3.11):
#   pip install backend/.openmmlab-wheels/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl   # Linux x86_64
#   pip install backend/.openmmlab-wheels/mmcv-2.1.0-cp311-cp311-macosx_10_9_universal2.whl # macOS
#   pip install "mmengine>=0.10.3" "mmpose>=1.3.0" "opencv-python==4.10.0.84"
#
# Env:
#   TORCH_VER=torch2.1.0   (OpenMMLab index path segment; must match your torch major.minor)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${1:-$BACKEND_DIR/.openmmlab-wheels}"
TORCH_VER="${TORCH_VER:-torch2.1.0}"
BASE="https://download.openmmlab.com/mmcv/dist/cpu/${TORCH_VER}"

mkdir -p "$OUT"

fetch() {
  local fname="$1"
  local url="$2"
  echo "Downloading $fname"
  curl -fSL --retry 3 -o "$OUT/$fname" "$url"
}

fetch "mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl" "${BASE}/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl"
fetch "mmcv-2.1.0-cp311-cp311-macosx_10_9_universal2.whl" "${BASE}/mmcv-2.1.0-cp311-cp311-macosx_10_9_universal2.whl"

echo ""
echo "Saved under: $OUT"
ls -lh "$OUT"
