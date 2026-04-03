#!/usr/bin/env bash
# Baked into Modal image after tools/modal_install_mmaction2.sh.
# Adds: MMDetection (MMPoseInferencer / RTMPose), DeepLabCut + TensorFlow stack (manylinux cp311).
#
# Pins llvmlite+numba before deeplabcut to avoid llvmlite source builds. Re-pins numpy at end.
set -euxo pipefail
export PIP_NO_CACHE_DIR=1

pip_leg() {
  python -m pip install --use-deprecated=legacy-resolver "$@"
}

# ── RTMPose API (MMPoseInferencer imports mmdet) ─────────────────────────────
pip_leg 'mmdet>=3.0,<3.3'

# ── DeepLabCut 2.3.x (TensorFlow) ───────────────────────────────────────────
pip_leg --no-deps 'llvmlite==0.42.0'
pip_leg --no-deps 'numba==0.59.1'
pip_leg 'tensorpack>=0.11' 'tf-slim>=1.1.0'
pip_leg 'tensorflow==2.15.1'
pip_leg 'deeplabcut==2.3.11'

# TF 2.15 pulls ml-dtypes~=0.3; jax (from google-cloud stack in requirements-modal) needs
# ml-dtypes>=0.5 for float8_e3m4 — upgrade without letting pip downgrade TF deps tree.
pip_leg --no-deps --force-reinstall 'ml-dtypes>=0.5.0,<0.6'

# Torch / mmcv / TF all expect numpy 1.26.x
pip_leg --no-deps --force-reinstall 'numpy==1.26.4'

python -c "
from mmpose.apis import MMPoseInferencer
import deeplabcut as dlc
print('[build] Modal ML extras: MMPoseInferencer + deeplabcut OK', dlc.__version__)
"
