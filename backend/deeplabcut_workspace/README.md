# DeepLabCut workspace (local only)

This directory is **gitignored** except this file. Use it for a DLC project + inference outputs.

## One-time bootstrap

From `backend/` with the same venv that has `deeplabcut` + `tensorflow`:

```bash
python scripts/bootstrap_deeplabcut_workspace.py
```

This creates a minimal placeholder video, runs `deeplabcut.create_new_project`, and writes `.stellar_dlc_config` (path to `config.yaml`). The backend resolves `STELLAR_DLC_PROJECT_CONFIG` from that file when the env var is unset.

**Important:** `analyze_videos` needs a **trained snapshot** under `dlc-models/`. Until you label + train (or copy a trained project here), keep `STELLAR_RESEARCH_BACKEND=disabled` so Plus analysis does not call DLC on every request.

## Env (see `backend/.env.example`)

- `STELLAR_RESEARCH_BACKEND=deeplabcut` — enable the research refine hook
- `STELLAR_DLC_PROJECT_CONFIG` — optional override; absolute path to `config.yaml`
- `STELLAR_DLC_OUTPUT_DIR` — optional; default `deeplabcut_workspace/outputs`

Install stack: `backend/scripts/install_deeplabcut.sh` or `requirements-deeplabcut.txt`.

**Modal:** the image runs `tools/modal_install_ml_extras.sh` and bakes a project under `/opt/deeplabcut_workspace` (same marker file `.stellar_dlc_config`). The backend resolves that path automatically; training is still required before enabling `STELLAR_RESEARCH_BACKEND=deeplabcut`.
