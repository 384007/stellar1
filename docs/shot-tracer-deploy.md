# Shot Tracer deployment checklist

This checklist covers the `stellar_shot_tracer_v1` feature added by PR #56.

## What runs where

- Cloudflare Pages serves the Next.js UI and the Edge proxy route:
  - `frontend/app/shot-tracer/page.tsx`
  - `frontend/app/api/shot-tracer/reconstruct/route.ts`
- Modal runs the heavy FastAPI reconstruction endpoint:
  - `POST /shot-tracer/reconstruct`
  - `backend/routers/shot_tracer.py`
  - `backend/services/shot_tracer_reconstruct_service.py`

The Cloudflare route must stay `export const runtime = "edge";` because Cloudflare Pages cannot deploy Node runtime routes.

## Required Modal deployment

Deploy the backend after merging or checking out the Shot Tracer branch:

```bash
modal deploy modal_app.py
```

Then verify the deployed backend:

```bash
curl -s "https://YOUR-MODAL-APP-URL/health"
```

The response should include:

```json
{
  "shot_tracer": {
    "route": "POST /shot-tracer/reconstruct",
    "enabled": true,
    "fallback_available": true,
    "modal_ready": true
  }
}
```

## Modal dependencies

`backend/requirements-modal.txt` already includes the required fallback stack:

- `fastapi`
- `python-multipart`
- `httpx`
- `mediapipe`
- `opencv-python-headless`
- `imageio-ffmpeg`
- `numpy`
- `Pillow`

`modal_app.py` installs OS packages needed by OpenCV/MediaPipe and also installs `ultralytics` for optional YOLO adapters. The service is safe when `ultralytics` or YOLO weights are missing because YOLO is imported lazily and falls back to MediaPipe/OpenCV.

## Required Cloudflare Pages environment

Set the same backend base URL variable already used by `modalAnalysisBase(...)` in your existing frontend chain helpers. The value must be the Modal FastAPI origin, without a trailing slash.

Example value:

```env
MODAL_ANALYSIS_BASE=https://YOUR-MODAL-APP-URL
```

If your existing project uses a different variable name for Modal/backend base URL, keep using that name. The important requirement is that `modalAnalysisBase(getCfEnv, request)` resolves to the Modal origin, otherwise the proxy returns:

```json
{ "detail": "分析上游未配置" }
```

Also keep shared secrets aligned between Cloudflare Pages and Modal where the existing app requires them:

```env
JWT_SECRET=the-same-value-as-modal
FRONTEND_URL=https://YOUR-CLOUDFLARE-PAGES-DOMAIN
```

## Optional Shot Tracer model/provider environment

These are backend-only variables. Configure them in the Modal `custom-secret` only. Do not expose them as `NEXT_PUBLIC_*` values.

### YOLO club and ball weights

```env
STELLAR_YOLO_CLUB_WEIGHTS=/models/golf-club/best.pt
STELLAR_YOLO_BALL_WEIGHTS=/models/golf-ball/best.pt
```

Upload the model files to the Modal volume mounted at `/models`:

```bash
modal volume put stellar-models ./best-club.pt /golf-club/best.pt
modal volume put stellar-models ./best-ball.pt /golf-ball/best.pt
```

If these are not configured, Shot Tracer still runs with MediaPipe pose proxy and OpenCV motion fallback, but club-head and ball tracking quality is lower.

### Roboflow adapters

```env
STELLAR_ROBOFLOW_API_KEY=
STELLAR_ROBOFLOW_CLUB_MODEL=
STELLAR_ROBOFLOW_BALL_MODEL=
```

### TrackNet ball tracker

```env
STELLAR_TRACKNET_API_URL=
STELLAR_TRACKNET_API_KEY=
```

### 3D scene providers

```env
STELLAR_POSTSHOT_API_URL=
STELLAR_POSTSHOT_API_KEY=
STELLAR_TRELLIS_API_URL=
STELLAR_TRELLIS_API_KEY=
```

These are optional. Without them, `mode=3d_scene` will not create external 3D assets, but the standard video-based 2D/3D path response still works.

## Test commands

### Backend syntax check

Use `python3` on macOS if `python` is not installed:

```bash
python3 -m py_compile \
  backend/services/shot_tracer_reconstruct_service.py \
  backend/routers/shot_tracer.py \
  backend/main.py
```

### Frontend type check

```bash
cd frontend
npx tsc --noEmit
```

### Direct Modal API test

```bash
curl -X POST "https://YOUR-MODAL-APP-URL/shot-tracer/reconstruct" \
  -F "file=@/path/to/swing.mp4" \
  -F "mode=single_video"
```

Expected top-level fields:

```json
{
  "status": "ok",
  "engine": "stellar_shot_tracer_v1",
  "video": {},
  "phases": {},
  "paths": {},
  "metrics": {},
  "providers": {},
  "limitations": []
}
```

### Cloudflare proxy test

```bash
curl -X POST "https://YOUR-CLOUDFLARE-PAGES-DOMAIN/api/shot-tracer/reconstruct" \
  -F "file=@/path/to/swing.mp4" \
  -F "mode=single_video"
```

If this fails with `分析上游未配置`, the Cloudflare backend base URL environment variable is missing or not recognized by `modalAnalysisBase(...)`.

## Minimum production setup

For a working demo:

1. Deploy Modal with `modal deploy modal_app.py`.
2. Configure Cloudflare Pages backend base URL so the Edge proxy can reach Modal.
3. Confirm `/health` reports `shot_tracer.enabled=true`.
4. Test direct Modal upload.
5. Test Cloudflare `/api/shot-tracer/reconstruct` upload.

For higher-quality tracking:

1. Upload club and ball YOLO weights to the Modal `stellar-models` volume.
2. Set `STELLAR_YOLO_CLUB_WEIGHTS` and `STELLAR_YOLO_BALL_WEIGHTS` in the Modal `custom-secret`.
3. Redeploy Modal.
4. Re-test and confirm `providers.club_detector` / `providers.ball_tracker` use YOLO or another configured provider.
