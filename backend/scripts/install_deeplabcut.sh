#!/usr/bin/env bash
# Install DeepLabCut + TensorFlow for STELLAR_RESEARCH_BACKEND=deeplabcut (optional).
#
# macOS: pre-install binary llvmlite+numba so deeplabcut does not trigger a llvmlite source build (needs cmake).
#
# Usage (from backend/):
#   chmod +x scripts/install_deeplabcut.sh
#   PYTHON=.venv/bin/python ./scripts/install_deeplabcut.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "pip missing in $PY" >&2
  exit 1
fi

echo "==> Pin llvmlite + numba (avoid cmake build on macOS x86_64)"
"$PY" -m pip install "llvmlite==0.42.0" "numba==0.59.1"

echo "==> Install deeplabcut + tensorflow (large download)"
"$PY" -m pip install -r requirements-deeplabcut.txt

echo "==> Upgrade ml-dtypes (jax/Vertex need float8_e3m4; TF wheel pins 0.3.x)"
"$PY" -m pip install --no-deps --force-reinstall 'ml-dtypes>=0.5.0,<0.6'

echo "==> Bootstrap workspace (minimal project + .stellar_dlc_config)"
"$PY" scripts/bootstrap_deeplabcut_workspace.py

echo "OK: deeplabcut stack installed. Train a model before STELLAR_RESEARCH_BACKEND=deeplabcut."
