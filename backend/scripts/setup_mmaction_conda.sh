#!/usr/bin/env bash
# Recreates the local MMAction2 environment under backend/.conda-mmaction (not committed; ~2GB+).
# Use when system Python is 3.13 or when pip cannot install torch/mmcv wheels.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONDA_ROOT="$ROOT/.conda-mmaction"
CONDA="$CONDA_ROOT/bin/conda"
ENV_NAME="stellar-mmaction"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
PIP="$CONDA_ROOT/envs/$ENV_NAME/bin/pip"

if [[ ! -x "$CONDA" ]]; then
  echo "Installing Miniconda into $CONDA_ROOT ..."
  ARCH="$(uname -m)"
  case "$ARCH" in
    arm64) URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
    x86_64) URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
    *) echo "Unsupported arch: $ARCH"; exit 1 ;;
  esac
  curl -fsSL -o /tmp/miniconda-mmaction.sh "$URL"
  bash /tmp/miniconda-mmaction.sh -b -p "$CONDA_ROOT"
  rm -f /tmp/miniconda-mmaction.sh
fi

if [[ ! -d "$CONDA_ROOT/envs/$ENV_NAME" ]]; then
  "$CONDA" create -y -n "$ENV_NAME" python=3.11 pip
fi

echo "Installing PyTorch 2.1.x + numpy<2 (matches prebuilt mmcv for macOS) ..."
# setuptools>=82 drops pkg_resources; openmim/mim still imports it — cap below 82.
"$PIP" install "numpy>=1.23,<2" "setuptools>=69,<82" torch==2.1.2 torchvision==0.16.2

echo "decord (video decoding) from conda-forge ..."
"$CONDA" install -y -n "$ENV_NAME" -c conda-forge decord

"$PIP" install opencv-python-headless==4.10.0.84
"$PIP" install --no-deps "mmengine>=0.10.3"
"$PIP" install addict matplotlib pyyaml termcolor yapf
"$PIP" install "openmim>=0.3.9"

"$PIP" install einops scipy importlib_metadata
"$PIP" install --no-deps mmaction2==1.2.0
bash "$(cd "$(dirname "$0")" && pwd)/ensure_mmaction2_runtime.sh" "$PY"

echo "mmcv: use official macOS universal2 wheel (torch 2.1 CPU index); --no-deps avoids opencv source build ..."
"$PIP" install mmcv==2.1.0 --no-deps -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html

echo ""
echo "Verify:"
"$PY" -c "import torch, mmcv, mmaction; from mmaction.apis import init_recognizer; print('torch', torch.__version__, 'mmcv', mmcv.__version__, 'mmaction ok')"
echo ""
echo "Run backend with this interpreter (example):"
echo "  export PYTHONPATH=\"$ROOT\""
echo "  $PY -m uvicorn main:app --reload"
echo "(Adjust main:app to your FastAPI entrypoint.)"
