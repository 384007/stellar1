#!/usr/bin/env bash
# PyPI mmaction2==1.2.0 wheel omits localizers/drn; mim needs setuptools<82 for pkg_resources.
# Run after: pip install mmaction2==1.2.0 --no-deps  (e.g. from setup_mmaction_conda.sh)
set -euo pipefail
PY="${1:-python3}"
SITE="$("$PY" -c "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])")"
DRN="$SITE/mmaction/models/localizers/drn"
if [[ -d "$DRN" ]] && [[ -f "$DRN/drn.py" ]]; then
  echo "mmaction localizers/drn already present."
  exit 0
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --depth 1 --branch main https://github.com/open-mmlab/mmaction2.git "$TMP/mmaction2"
mkdir -p "$(dirname "$DRN")"
cp -R "$TMP/mmaction2/mmaction/models/localizers/drn" "$(dirname "$DRN")/"
echo "Patched drn into $DRN"
