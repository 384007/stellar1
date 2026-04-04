/**
 * Pro v2 upload targets: primary + optional fallbacks for CN / unstable backends.
 *
 * **All regions (including China):** the client always attempts **Modal (GPU) first**. It does not
 * put Render ahead of Modal when `network_hint` is `cn`. China only adds `X-Stellar-Network-Hint: cn`
 * (workers can tune Gemini/copy) and slightly more aggressive **Render** connect retries **when**
 * `NEXT_PUBLIC_PRO_V2_RENDER_FALLBACK=true` — that flag is off by default (Modal-only + Modal HTTP retries).
 *
 * Prepend mirrors that work from mainland via `MODAL_BACKEND_CN_EXTRA` / `BACKEND_URL_CN_EXTRA`.
 *
 * Configure on CF Pages / Worker (same pattern as MODAL_BACKEND_URL):
 * - MODAL_BACKEND_URL           — primary Modal base (no trailing slash)
 * - MODAL_BACKEND_FALLBACKS     — comma-separated extra Modal bases
 * - MODAL_BACKEND_CN_EXTRA      — when client country is CN, tried before primary (e.g. mirror)
 * - NEXT_PUBLIC_BACKEND_URL     — primary Render (or other) API base
 * - BACKEND_URL_FALLBACKS       — comma-separated extra Render/API bases
 * - BACKEND_URL_CN_EXTRA        — CN-only, prepended before primary + global fallbacks
 */

/** Shipped default when env has no MODAL_BACKEND_URL; keep in sync with Modal dashboard deployment. */
export const DEFAULT_PRO_V2_MODAL_URL = "https://dytsui--stellar-ai-fastapi-app.modal.run";

/** Pro v3（`/pro-v3/analyze`）与上述共用 Modal 基址与环境变量。 */
export const DEFAULT_PROV3_MODAL_URL = DEFAULT_PRO_V2_MODAL_URL;
const DEFAULT_RENDER = "https://stellar1-backend.onrender.com";

function splitCsv(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim().replace(/\/+$/, ""))
    .filter(Boolean);
}

function dedupePreserveOrder(urls: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const u of urls) {
    const n = u.replace(/\/+$/, "");
    if (!n || seen.has(n)) continue;
    seen.add(n);
    out.push(n);
  }
  return out;
}

export function buildProV2ModalUrlList(
  getCfEnv: (key: string) => string,
  isCn: boolean,
  processEnv: Record<string, string | undefined> = process.env,
): string[] {
  const primary = (
    getCfEnv("MODAL_BACKEND_URL") ||
    processEnv.MODAL_BACKEND_URL ||
    DEFAULT_PRO_V2_MODAL_URL
  )
    .trim()
    .replace(/\/+$/, "");
  const globalFb = splitCsv(getCfEnv("MODAL_BACKEND_FALLBACKS") || processEnv.MODAL_BACKEND_FALLBACKS || "");
  const cnExtra = isCn
    ? splitCsv(getCfEnv("MODAL_BACKEND_CN_EXTRA") || processEnv.MODAL_BACKEND_CN_EXTRA || "")
    : [];
  return dedupePreserveOrder([...cnExtra, primary, ...globalFb]);
}

export function buildProV2BackendUrlList(
  getCfEnv: (key: string) => string,
  isCn: boolean,
  primaryFallback: string,
  processEnv: Record<string, string | undefined> = process.env,
): string[] {
  const primary = (
    getCfEnv("NEXT_PUBLIC_BACKEND_URL") ||
    processEnv.NEXT_PUBLIC_BACKEND_URL ||
    primaryFallback
  )
    .trim()
    .replace(/\/+$/, "");
  const globalFb = splitCsv(getCfEnv("BACKEND_URL_FALLBACKS") || processEnv.BACKEND_URL_FALLBACKS || "");
  const cnExtra = isCn
    ? splitCsv(getCfEnv("BACKEND_URL_CN_EXTRA") || processEnv.BACKEND_URL_CN_EXTRA || "")
    : [];
  return dedupePreserveOrder([...cnExtra, primary, ...globalFb]);
}

export function normalizeProV2UrlListsFromPrecheck(data: {
  modal_url?: string;
  backend_url?: string;
  modal_urls?: unknown;
  backend_urls?: unknown;
}): { modalUrls: string[]; backendUrls: string[] } {
  const modalFromArr = Array.isArray(data.modal_urls)
    ? (data.modal_urls as unknown[]).filter((x): x is string => typeof x === "string")
    : [];
  const backendFromArr = Array.isArray(data.backend_urls)
    ? (data.backend_urls as unknown[]).filter((x): x is string => typeof x === "string")
    : [];
  const modalUrls = dedupePreserveOrder(
    modalFromArr.length > 0
      ? modalFromArr.map((u) => u.replace(/\/+$/, ""))
      : data.modal_url
        ? [String(data.modal_url).replace(/\/+$/, "")]
        : [],
  );
  let backendUrls = dedupePreserveOrder(
    backendFromArr.length > 0
      ? backendFromArr.map((u) => u.replace(/\/+$/, ""))
      : data.backend_url
        ? [String(data.backend_url).replace(/\/+$/, "")]
        : [DEFAULT_RENDER],
  );
  if (backendUrls.length === 0) {
    backendUrls = [DEFAULT_RENDER];
  }
  return { modalUrls, backendUrls };
}

/** Pro v3：与 `normalizeProV2UrlListsFromPrecheck` 相同（precheck 契约一致）。 */
export function normalizeProv3UrlListsFromPrecheck(
  data: Parameters<typeof normalizeProV2UrlListsFromPrecheck>[0],
): ReturnType<typeof normalizeProV2UrlListsFromPrecheck> {
  return normalizeProV2UrlListsFromPrecheck(data);
}
