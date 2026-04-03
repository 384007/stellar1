#!/usr/bin/env bash
# Install MMAction2 stack into the current Python environment (see backend/requirements-mmaction2.txt).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

ver="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$ver" in
  3.10|3.11|3.12) ;;
  *)
    echo "Note: PyTorch/MMCV wheels are most reliable on Python 3.10–3.12. This interpreter is ${ver}." >&2
    echo "      On Intel macOS + Python 3.13, PyTorch may have no wheel — use 3.11 venv or Linux/Docker." >&2
    ;;
esac

echo "Installing pip packages from requirements-mmaction2.txt ..."
"$PY" -m pip install -r requirements-mmaction2.txt

echo "Installing mmcv via OpenMIM (matches your torch build) ..."
PREFIX="$("$PY" -c "import sys; print(sys.prefix)")"
if [[ -x "$PREFIX/bin/mim" ]]; then
  "$PREFIX/bin/mim" install "mmcv>=2.0.0,<2.3.0"
elif command -v mim >/dev/null 2>&1; then
  mim install "mmcv>=2.0.0,<2.3.0"
else
  echo "ERROR: could not find \`mim\`. Use the same Python you use for pip, e.g.:" >&2
  echo "  $PY -m pip install -U openmim && $PY -m pip install -r requirements-mmaction2.txt" >&2
  echo "  then add that environment's bin/ to PATH and run: mim install \"mmcv>=2.0.0,<2.3.0\"" >&2
  exit 1
fi

echo ""
echo "Verify: $PY -c \"import mmaction; print('mmaction ok')\""
echo "Download a default Kinetics TSN checkpoint (example):"
echo "  cd \"$ROOT\" && $PY -m mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest ./mmaction_models"
