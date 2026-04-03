#!/usr/bin/env bash
# Install MMPose into the current Python environment (after mmcv from install_mmaction2.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

echo "Installing MMPose from requirements-mmpose.txt ..."
"$PY" -m pip install -r requirements-mmpose.txt

echo ""
echo "Verify: $PY -c \"import mmpose; print('mmpose ok')\""
echo "Set STELLAR_RTMPOSE_CONFIG / STELLAR_RTMPOSE_CHECKPOINT when enabling RTMPose."
