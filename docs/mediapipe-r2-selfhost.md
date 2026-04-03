# Self-host MediaPipe (R2 + CDN) for China / blocked networks

**Click-by-click (Cloudflare Dashboard, Chinese):** [mediapipe-r2-dashboard-steps.zh.md](./mediapipe-r2-dashboard-steps.zh.md)

The live skeleton (`ScreenModeCapture`) loads:

- `vision_bundle.mjs` + WASM from npm CDNs (often blocked in CN)
- `.task` models from `storage.googleapis.com` (blocked in CN)

Set **`NEXT_PUBLIC_MEDIAPIPE_CDN_BASE`** to a **public URL** whose path matches the folder you upload to R2.

**Default (recommended for CN):** when this variable is set, the client uses **only** your R2 URLs (bundle, wasm, models). It does **not** chain jsdelivr / unpkg / Google — those retries used multi-second timeouts each and felt like endless「加载中」 on blocked networks.

To restore the old “self first, then foreign CDNs” behavior, set **`NEXT_PUBLIC_MEDIAPIPE_ALLOW_FOREIGN_FALLBACK=1`**.

## 1. Pack files locally

```bash
cd frontend && npm install
node ../tools/mediapipe-pack-for-r2.mjs
```

This writes:

```
frontend/build/mediapipe-r2/<version>/
  vision_bundle.mjs
  wasm/          # all .js + .wasm from the npm package
  models/
    pose_landmarker_full.task
    pose_landmarker_lite.task
```

Version must match `frontend/lib/mediapipe-assets.ts` (`MEDIAPIPE_TASKS_VISION_VERSION`) and `package.json` dependency.

## 2. Upload to R2

Pick a **key prefix** (example uses bucket `stellar-golf-media` from `frontend/wrangler.toml`):

```
static/mediapipe/tasks-vision/0.10.33/vision_bundle.mjs
static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_internal.wasm
…
static/mediapipe/tasks-vision/0.10.33/models/pose_landmarker_full.task
static/mediapipe/tasks-vision/0.10.33/models/pose_landmarker_lite.task
```

**Wrangler — one script** (after `npx wrangler login`, from repo root):

```bash
cd frontend && npm install && node ../tools/mediapipe-pack-for-r2.mjs
cd ..
bash tools/r2-upload-mediapipe.sh
# Optional: R2_BUCKET=my-bucket bash tools/r2-upload-mediapipe.sh
```

Or upload manually:

```bash
cd frontend/build/mediapipe-r2/0.10.33
wrangler r2 object put stellar-golf-media/static/mediapipe/tasks-vision/0.10.33/vision_bundle.mjs --file=vision_bundle.mjs --content-type=application/javascript
# … repeat for wasm/* and models/*
```

Or use **Cloudflare Dashboard** → R2 → Upload folder (preserve paths).

Or **aws s3 sync** with R2 S3 API (see Cloudflare R2 docs).

### Content types (recommended)

| Object | `Content-Type` |
|--------|----------------|
| `vision_bundle.mjs` | `application/javascript` or `text/javascript` |
| `wasm/*.wasm` | `application/wasm` |
| `wasm/*.js` | `application/javascript` |
| `models/*.task` | `application/octet-stream` |

## 3. Public URL + CORS

1. R2 bucket → **Settings** → allow **public access** (R2.dev subdomain or **Custom Domain**).
2. **CORS** on the bucket: allow `GET` from your Pages domain(s), e.g.:

```json
[
  {
    "AllowedOrigins": ["https://your-app.pages.dev", "https://your-domain.com"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 86400
  }
]
```

Without CORS, the browser will block WASM / module loads.

## 4. App configuration

**Cloudflare Pages** → **Settings** → **Environment variables**:

```
NEXT_PUBLIC_MEDIAPIPE_CDN_BASE=https://pub-xxxxx.r2.dev/static/mediapipe/tasks-vision/0.10.33
```

- No trailing slash.
- Must be the URL that maps to the folder containing `vision_bundle.mjs` and `wasm/`.

Redeploy the site so Next.js inlines the variable.

## 5. Verify

Open DevTools → Network: first requests for `vision_bundle.mjs` and `wasm` should hit your R2 host. Skeleton should leave “加载中” without relying on jsdelivr or Google.

## Rollback

Remove `NEXT_PUBLIC_MEDIAPIPE_CDN_BASE` and redeploy; the app uses public CDNs only (may fail in CN).
