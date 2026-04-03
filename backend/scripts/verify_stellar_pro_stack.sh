#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Check ffmpeg"
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg missing"; exit 1; }
ffmpeg -version | head -n 1

echo "==> Check ffprobe"
command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe missing"; exit 1; }
ffprobe -version | head -n 1

echo "==> Check Python imports"
python - <<'PY'
from services.ffmpeg_preprocess_service import ffmpeg_available
from services.stellar_pro_api_service import create_stellar_pro_api_service
from services.stellar_pro_pipeline import run_stellar_pro_pipeline_async
from services.stellar_pro_role_log import STELLAR_PRO_ROLE_ORDER

assert ffmpeg_available() is True or ffmpeg_available() is False
svc = create_stellar_pro_api_service()
print('service=', svc.__class__.__name__)
print('pipeline=', run_stellar_pro_pipeline_async.__name__)
print('roles=', len(STELLAR_PRO_ROLE_ORDER))
print('first_role=', STELLAR_PRO_ROLE_ORDER[0])
print('last_role=', STELLAR_PRO_ROLE_ORDER[-1])
PY

echo "==> Verify main entrypoint import"
python - <<'PY'
import main
print('main_app=', type(main.app).__name__)
PY

echo "==> Stellar Pro stack verify OK"
