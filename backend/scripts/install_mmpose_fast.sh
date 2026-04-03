#!/usr/bin/env bash
# Fast MMPose stack: prebuilt mmcv wheel only — no openmim / mim (avoids opencv source builds).
#
# Usage (from backend/):
#   chmod +x scripts/install_mmpose_fast.sh
#   ./scripts/install_mmpose_fast.sh
#
# Env:
#   PYTHON=python3   interpreter (must match venv)
#   UV=1             use `uv pip install` — often faster resolve/download than pip
#   SKIP_IF_PRESENT=0  force reinstall even when imports already work (default: 1 = skip satisfied steps)
#
# Slow part: mmpose pulls OpenCV and other large wheels — first run can take several minutes;
#   this script prints timestamps each step and enables pip progress bars (no silent -q).
#
# PyTorch / Python:
#   - macOS: OpenMMLab only ships mmcv CPU wheels for Python 3.9–3.11 (not 3.12+). Use a 3.11 venv.
#   - Linux + Python 3.12: use PyTorch >= 2.4 so we can pull mmcv 2.2.0 (torch2.4.0 index); 3.12 has no mmcv 2.1 wheel.
#   - Linux + Python 3.11: torch 2.1.x + mmcv 2.1.0 (default URLs below).
set -euo pipefail
export PYTHONUNBUFFERED=1
# Pip 23+ progress bar on stderr so you see activity during big downloads.
export PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-on}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
SKIP_IF_PRESENT="${SKIP_IF_PRESENT:-1}"

have_uv() { command -v uv >/dev/null 2>&1 && test "${UV:-}" = "1"; }

pip_leg() {
  if have_uv; then
    uv pip install --python "$PY" "$@"
  else
    # Never -q: slow steps need visible download/extract progress.
    "$PY" -m pip install "$@"
  fi
}

step() {
  printf '\n==> %s %s\n' "$(date '+%H:%M:%S')" "$*"
}

py_import_ok() {
  "$PY" -c "import $1" 2>/dev/null
}

mmpose_inferencer_ok() {
  "$PY" -c "from mmpose.apis import MMPoseInferencer" 2>/dev/null
}

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "pip missing; try: uv pip install pip --python \"$PY\"  OR  python -m ensurepip --upgrade" >&2
  exit 1
fi

cp_tag="$("$PY" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
py_mm="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
torch_ver="$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || true)"
case "$torch_ver" in
  2.1.*) mmcv_torch="torch2.1.0" ;;
  2.2.*) mmcv_torch="torch2.2.0" ;;
  *)
    echo "WARN: torch=${torch_ver:-missing}; mmcv URL uses torch2.1.0 index (on mac/Linux3.11 match with torch 2.1.2 / 2.2.x)." >&2
    mmcv_torch="torch2.1.0"
    ;;
esac

os="$(uname -s)"
mmcv_url=""
mmcv_file_ver="2.1.0"

darwin_too_new_python() {
  cat >&2 <<EOF
ERROR: This interpreter is Python ${py_mm} on macOS. OpenMMLab does not publish mmcv CPU wheels for cp312/cp313 on macOS.

Fix (pick one):
  1) Homebrew 3.11 + venv (Apple Silicon / Intel):
       brew install python@3.11
       /opt/homebrew/bin/python3.11 -m venv .venv-mmpose   # Intel: /usr/local/opt/python@3.11/bin/python3.11
       source .venv-mmpose/bin/activate
       pip install -U pip
       pip install 'torch==2.1.2' 'torchvision==0.16.2'
       cd backend && PYTHON=python ./scripts/install_mmpose_fast.sh

  2) pyenv:
       pyenv install 3.11.9 && pyenv local 3.11.9
       python -m venv .venv && source .venv/bin/activate
       pip install 'torch==2.1.2' 'torchvision==0.16.2'
       cd backend && PYTHON=python ./scripts/install_mmpose_fast.sh

  3) Conda / Docker with Python 3.11 and torch 2.1.x (see backend/requirements-mmpose.txt).
EOF
}

if test "$os" = "Darwin"; then
  # Official CPU index: cp311 → universal2 (Intel + Apple Silicon); cp310/cp39 → x86_64 wheels.
  case "$cp_tag" in
    cp311)
      mmcv_url="https://download.openmmlab.com/mmcv/dist/cpu/${mmcv_torch}/mmcv-2.1.0-cp311-cp311-macosx_10_9_universal2.whl"
      ;;
    cp310)
      mmcv_url="https://download.openmmlab.com/mmcv/dist/cpu/${mmcv_torch}/mmcv-2.1.0-cp310-cp310-macosx_11_0_x86_64.whl"
      ;;
    cp39)
      mmcv_url="https://download.openmmlab.com/mmcv/dist/cpu/${mmcv_torch}/mmcv-2.1.0-cp39-cp39-macosx_11_0_x86_64.whl"
      ;;
    *)
      darwin_too_new_python
      exit 1
      ;;
  esac
elif test "$os" = "Linux"; then
  case "$cp_tag" in
    cp312)
      # No mmcv-2.1.0 cp312 manylinux wheel; 2.2.0 exists under torch2.4.0 only.
      case "$torch_ver" in
        2.4.*|2.5.*|2.6.*|2.7.*)
          mmcv_torch="torch2.4.0"
          mmcv_file_ver="2.2.0"
          mmcv_url="https://download.openmmlab.com/mmcv/dist/cpu/${mmcv_torch}/mmcv-${mmcv_file_ver}-cp312-cp312-manylinux1_x86_64.whl"
          ;;
        *)
          cat >&2 <<EOF
ERROR: Python 3.12 on Linux has no mmcv 2.1.0 wheel. Either:
  - Install PyTorch >= 2.4 then re-run this script (uses mmcv 2.2.0 / torch2.4.0 index), or
  - Use Python 3.11 venv + torch 2.1.2 + this script (mmcv 2.1.0).
EOF
          exit 1
          ;;
      esac
      ;;
    cp313|cp314)
      cat >&2 <<EOF
ERROR: No official OpenMMLab mmcv CPU wheel found for ${cp_tag} on Linux. Use Python 3.11 or 3.12 (3.12 needs torch>=2.4), or mim/conda.
EOF
      exit 1
      ;;
    *)
      mmcv_url="https://download.openmmlab.com/mmcv/dist/cpu/${mmcv_torch}/mmcv-${mmcv_file_ver}-${cp_tag}-${cp_tag}-manylinux1_x86_64.whl"
      ;;
  esac
else
  echo "Unsupported OS: $os — use Linux container or mim install mmcv." >&2
  exit 1
fi

if test -z "${torch_ver}" && test "$os" = "Darwin"; then
  echo "TIP: Install PyTorch before mmpose (example): pip install 'torch==2.1.2' 'torchvision==0.16.2'" >&2
fi

step "Fast install: mmengine + mmcv (wheel) + mmpose"
echo "    python=$("$PY" -c 'import sys; print(sys.executable)')"
echo "    cp_tag=$cp_tag  torch=${torch_ver:-missing}  mmcv_index=$mmcv_torch  mmcv=${mmcv_file_ver}"
echo "    mmcv_url=$mmcv_url"
if test "${UV:-}" != "1" && command -v uv >/dev/null 2>&1; then
  echo "    tip: UV=1 ./scripts/install_mmpose_fast.sh  often speeds up installs"
fi

if test "$SKIP_IF_PRESENT" = "1" && py_import_ok mmengine && py_import_ok mmcv && py_import_ok mmpose && mmpose_inferencer_ok; then
  step "mmengine, mmcv, mmpose + MMPoseInferencer (RTMPose API) already OK — nothing to do (SKIP_IF_PRESENT=1)."
  "$PY" -c "import mmcv, mmengine, mmpose; print('OK mmcv', mmcv.__version__, 'mmpose', mmpose.__version__)"
  exit 0
fi

if test "$SKIP_IF_PRESENT" = "1" && py_import_ok mmengine; then
  step "Skip mmengine (already installed)"
else
  step "Install mmengine (small, should be quick)…"
  pip_leg "mmengine>=0.10.3"
fi

if test "$SKIP_IF_PRESENT" = "1" && py_import_ok mmcv; then
  step "Skip mmcv (already installed)"
else
  step "Install mmcv prebuilt wheel (small download, quick)…"
  if ! pip_leg "$mmcv_url"; then
    echo "ERROR: mmcv wheel failed. Try torch 2.1.2+torchvision 0.16.2, or another mmcv_torch on:" >&2
    echo "  https://download.openmmlab.com/mmcv/dist/cpu/${mmcv_torch}/index.html" >&2
    exit 1
  fi
fi

if test "$SKIP_IF_PRESENT" = "1" && py_import_ok mmpose; then
  step "Skip mmpose (already installed)"
else
  step "Install mmpose + dependencies (OpenCV etc.) — often the slowest step, 2–8+ min on first run…"
  # mmpose → chumpy (legacy sdist): isolated PEP517 build can fail (no pip in build env / old setuptools).
  if ! py_import_ok chumpy; then
    step "Upgrade pip/setuptools/wheel (helps chumpy build)…"
    pip_leg -U "pip>=24" "setuptools>=69" wheel || true
    step "Pre-install chumpy with --no-build-isolation…"
    pip_leg --no-build-isolation "chumpy>=0.69"
  fi
  pip_leg "mmpose>=1.3.0"
fi

# RTMPose path in this repo uses MMPoseInferencer → needs mmdet; mmengine then needs setuptools<81 (pkg_resources).
if ! mmpose_inferencer_ok; then
  if ! py_import_ok mmdet; then
    step "Install mmdet (MMDetection — required for MMPoseInferencer / RTMPose)…"
    pip_leg "mmdet>=3.0,<3.3"
  else
    step "mmdet present; fixing inferencer import if needed…"
  fi
  step "Pin setuptools to <81 (mmengine + MMPoseInferencer use pkg_resources; setuptools 82+ breaks)…"
  pip_leg "setuptools>=70,<81" || true
fi

step "Verify imports…"
"$PY" -c "import mmcv, mmengine, mmpose; print('OK mmcv', mmcv.__version__, 'mmengine', mmengine.__version__, 'mmpose', mmpose.__version__)"
"$PY" -c "from mmpose.apis import MMPoseInferencer; print('OK MMPoseInferencer (use with STELLAR_RTMPOSE_CONFIG + STELLAR_RTMPOSE_CHECKPOINT)')"
