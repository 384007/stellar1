/**
 * Gemini API proxy + multi-key support for China mainland access.
 *
 * CF Edge Workers in CN PoPs can't reach generativelanguage.googleapis.com
 * reliably. This module provides:
 *   1. Ordered list of hosts: direct Google → ALI proxy → JD proxy
 *   2. Multi-key failover: GEMINI_API_KEY → GEMINI_API_KEY_2
 *   3. URL rewriting for upload URIs returned by Google
 *
 * Configure via CF Pages Secrets / env vars:
 *   GEMINI_API_KEY        — primary Gemini key
 *   GEMINI_API_KEY_2      — secondary Gemini key (quota failover)
 *   GEMINI_PROXY_ALI      — Alibaba Cloud reverse proxy host
 *   GEMINI_PROXY_JD       — JD Cloud reverse proxy host
 *   QWEN_API_KEY          — Qwen fallback key (dashscope.aliyuncs.com)
 *
 * Modal/Render (Python `gemini_service`): set the same GEMINI_PROXY_* secrets on Modal/Render
 * so Pro v2 / Plus AI tries those hosts before generativelanguage.googleapis.com (browser→worker
 * often has no CF-IPCountry header).
 */

export const GEMINI_DIRECT = "https://generativelanguage.googleapis.com";

/**
 * Return ordered Gemini API hosts based on region.
 * - CN: proxies first (direct Google is blocked by GFW), Google last as hail-mary
 * - non-CN: Google direct only (proxies add latency, not needed)
 *
 * Pro v2 browser uploads send `X-Stellar-Network-Hint: cn` when precheck marks CN; set the same
 * `GEMINI_PROXY_*` secrets on Modal/Render so the report step can reach Gemini from the worker.
 */
export function getGeminiHosts(getCfEnv: (key: string) => string, isCN = false): string[] {
  if (!isCN) return [GEMINI_DIRECT];

  const proxies: string[] = [];
  const ali = getCfEnv("GEMINI_PROXY_ALI");
  if (ali) proxies.push(ali.replace(/\/+$/, ""));
  const jd = getCfEnv("GEMINI_PROXY_JD");
  if (jd) proxies.push(jd.replace(/\/+$/, ""));
  return proxies.length > 0 ? [...proxies, GEMINI_DIRECT] : [GEMINI_DIRECT];
}

export function getGeminiKeys(getCfEnv: (key: string) => string): string[] {
  const keys: string[] = [];
  const k1 = getCfEnv("GEMINI_API_KEY");
  if (k1) keys.push(k1);
  const k2 = getCfEnv("GEMINI_API_KEY_2");
  if (k2) keys.push(k2);
  return keys;
}

/**
 * Rewrite an absolute Google URL (e.g. upload URI in x-goog-upload-url header)
 * so it routes through the current proxy host instead of direct Google.
 */
export function rewriteGoogleUrl(url: string, proxyHost: string): string {
  if (proxyHost === GEMINI_DIRECT) return url;
  return url.replace(GEMINI_DIRECT, proxyHost);
}
