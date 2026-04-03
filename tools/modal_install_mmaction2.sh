#!/usr/bin/env bash
# Baked into Modal image (see modal_app.py). Installs torch+torchvision (CPU), then MMAction2 deps.
# RTMPose inferencer (mmdet) + DeepLabCut + TensorFlow: see tools/modal_install_ml_extras.sh (runs after this script in modal_app.py).
#
# Precondition: base image used backend/requirements-modal.txt (no torch / ultralytics) so pip never sees
# torch 2.5 vs mmcv 2.4 conflict. Ultralytics is installed with --no-deps after this script in modal_app.py.
set -euxo pipefail
export PIP_NO_CACHE_DIR=1

test -d /root/mmaction2_localizers_drn/drn_utils
test -f /opt/mmaction_configs/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py

PYTORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
PYPI_SIMPLE="https://pypi.org/simple"
# mmaction2==1.2.0 asserts mmcv<2.2.0; only cp311 CPU wheel in OpenMMLab index is under torch2.1.0/.
MMCV_CP311_MANYLINUX_URL="https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl"
DECORD_MANYLINUX_URL="https://files.pythonhosted.org/packages/11/79/936af42edf90a7bd4e41a6cac89c913d4b47fa48a26b042d5129a9242ee3/decord-0.6.0-py3-none-manylinux2010_x86_64.whl"

pip_leg() {
  python -m pip install --use-deprecated=legacy-resolver "$@"
}

# Pin numpy before torch so pip does not pull numpy 2.x while satisfying torch.
pip_leg --no-deps --upgrade 'numpy==1.26.4'

# Match mmcv torch2.1.0 wheel exactly (2.1.0+cpu). Do NOT use --only torch index: torch needs PyPI deps
# for a working _C extension; --no-deps + CPU-only index yields broken/incomplete installs (NameError: _C).
python -m pip install --no-cache-dir --use-deprecated=legacy-resolver \
  --force-reinstall \
  --index-url "${PYTORCH_CPU_INDEX}" \
  --extra-index-url "${PYPI_SIMPLE}" \
  'torch==2.1.0+cpu' 'torchvision==0.16.0+cpu'

# PyPI deps pulled with torch often upgrade numpy to 2.x; torch 2.1+mmcv wheels expect numpy 1.x C API.
pip_leg --no-deps --force-reinstall 'numpy==1.26.4'

# Needed by setuptools 81+ and tools like yapf when installed with --no-deps elsewhere.
pip_leg 'platformdirs>=4.2,<5'
# mmengine / mmaction2 import paths expect rich; omitted when mmengine is installed --no-deps.
pip_leg 'rich>=13,<14'

pip_leg --no-deps \
  filelock typing-extensions sympy networkx jinja2 fsspec

pip_leg --no-deps 'setuptools==81.0.0'

pip_leg --no-deps importlib_metadata einops addict pyyaml termcolor yapf packaging
pip_leg --no-deps scipy
pip_leg --no-deps "${DECORD_MANYLINUX_URL}"
pip_leg --no-deps contourpy cycler fonttools kiwisolver pyparsing python-dateutil matplotlib

pip_leg --no-deps 'mmengine==0.10.5'
pip_leg --no-deps "${MMCV_CP311_MANYLINUX_URL}"
pip_leg --no-deps 'mmaction2==1.2.0'

# MMPose / RTMPose: PyPI pulls opencv-python + deps as wheels on manylinux (may replace opencv-python-headless from requirements-modal).
# Pose checkpoints stay off-image: STELLAR_RTMPOSE_* or /models at runtime.
pip_leg 'mmpose==1.3.2'
pip_leg --no-deps --force-reinstall 'numpy==1.26.4'

SITE="$(python -c "import site; ps=[p for p in site.getsitepackages() if 'site-packages' in p]; assert ps, site.getsitepackages(); print(ps[0])")"
mkdir -p "${SITE}/mmaction/models/localizers"
rm -rf "${SITE}/mmaction/models/localizers/drn"
cp -R /root/mmaction2_localizers_drn "${SITE}/mmaction/models/localizers/drn"

mkdir -p /opt/mmaction_models
TSN_URL="https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
curl -fsSL -o /opt/mmaction_models/tsn_kinetics400.pth "${TSN_URL}"
test -s /opt/mmaction_models/tsn_kinetics400.pth
python -c "import os; assert os.path.getsize('/opt/mmaction_models/tsn_kinetics400.pth') > 50_000_000"

python -c "import torch; import mmcv; import mmaction; import mmpose; print('[build] MMAction2+MMPose imports OK', torch.__version__, mmcv.__version__, mmpose.__version__)"
