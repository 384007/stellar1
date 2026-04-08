/**
 * Browser ``fetch`` and Edge proxy upstream wait for ``POST /analyze/lite`` (one hour).
 * Keep standalone Lite Modal ``timeout=`` (``modal_app_lite.py``) at least this high so the worker is not
 * killed while the client still waits.
 *
 * **Cloudflare Pages / Workers:** the *browser → Pages* leg for ``/api/lite/analyze-proxy`` may still hit CF’s
 * own wall clock (~100s typical) → **524** regardless of this value (CN path).
 *
 * **Other reverse proxies** may enforce their own caps (e.g. ~3–4m); raise origin read timeout there if needed.
 */
export const LITE_ANALYZE_FETCH_TIMEOUT_MS = 3_600_000;
