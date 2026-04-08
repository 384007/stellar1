/**
 * Client / Edge subrequest abort budget. Must be >= Modal worker wall time where applicable.
 *
 * **Cloudflare Pages / Workers:** the *incoming* request to ``/api/lite/analyze-proxy`` is still capped by
 * Cloudflare’s edge timeout (~100s typical) → HTTP **524** before this value matters for CN users.
 *
 * **Direct Modal / custom domains:** some proxies cut idle or total wait around **3–4 minutes** even when this
 * constant is 15m — shorten server work (see ``STELLAR_SWINGNET_LITE_MAX_FRAMES`` / ``STELLAR_LITE_B_SKIP_RECOVERY`` on Modal).
 */
export const LITE_ANALYZE_FETCH_TIMEOUT_MS = 900_000;
