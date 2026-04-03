#!/usr/bin/env bash
# Create backend/.venv with Python 3.11 + MediaPipe without building opencv-contrib from source.
# Requires: curl (installs uv to repo .uv/ if missing).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UV="${ROOT}/.uv/uv"
if [[ ! -x "$UV" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${ROOT}/.uv" sh
fi
cd "${ROOT}/backend"
rm -rf .venv
"$UV" venv --python 3.11 .venv
"$UV" pip install opencv-python-headless==4.10.0.84 'numpy<2' Pillow==11.1.0 --python .venv/bin/python
"$UV" pip install mediapipe==0.10.21 --python .venv/bin/python --no-deps
"$UV" pip install matplotlib absl-py attrs flatbuffers jax jaxlib sentencepiece --python .venv/bin/python
"$UV" pip install fastapi==0.115.6 'uvicorn[standard]==0.34.0' python-multipart==0.0.20 pyjwt==2.10.1 httpx==0.28.1 \
  google-generativeai==0.8.4 'google-cloud-aiplatform>=1.73.0' python-dotenv==1.0.1 'openai>=1.54.0' --python .venv/bin/python
echo "Done. Activate: source ${ROOT}/backend/.venv/bin/activate"
