/**
 * Pro v3 upload targets: Modal first + optional Render fallbacks (CN mirrors, etc.).
 *
 * **All regions (including China):** the client always attempts **Modal (GPU) first**. China only adds
 * `X-Stellar-Network-Hint: cn` and slightly more aggressive **Render** connect retries **when**
 * `NEXT_PUBLIC_PROV3_RENDER_FALLBACK=true` — off by default (Modal-only + Modal HTTP retries).
 *
 * Prepend mirrors via `MODAL_BACKEND_CN_EXTRA` / `BACKEND_URL_CN_EXTRA`.
 *
 * Configure on CF Pages / Worker (same pattern as MODAL_BACKEND_URL):
 * - MODAL_BACKEND_URL           — primary Modal base (no trailing slash)
 * - MODAL_BACKEND_FALLBACKS     — comma-separated extra Modal bases
 * - MODAL_BACKEND_CN_EXTRA      — when client country is CN, tried before primary
 * - NEXT_PUBLIC_BACKEND_URL     — primary Render (or other) API base
 * - BACKEND_URL_FALLBACKS       — comma-separated extra Render/API bases
 * - BACKEND_URL_CN_EXTRA        — CN-only, prepended before primary + global fallbacks
 *
 * **Lite** ``POST /analyze/lite`` uses the same Modal base as Pro unless ``LITE_BACKEND_URL`` (Edge) or
 * ``NEXT_PUBLIC_LITE_BACKEND_URL`` (legacy) overrides — see ``resolveLiteAnalyzeClientOrigin`` /
 * ``resolveLiteAnalyzeUpstreamBase``.
 */

/** Next.js Edge route: Pro JWT + Modal/Render URL lists (see ``app/api/prov3/precheck``). */
export const PRO_V3_EDGE_PRECHECK_PATH = "/api/prov3/precheck";

/** FastAPI Pro v3 API — single prefix on Modal/Render (``backend/routers/prov3_api.py``). */
export const PRO_V3_HTTP_PREFIX = "/pro-v3";
export const PRO_V3_ANALYZE_PATH = `${PRO_V3_HTTP_PREFIX}/analyze`;

/** Raw keyframe pipeline (no Gemini / no product media). Product use ``PRO_V3_ANALYZE_PATH``. */
export const PRO_V3_KEYFRAMES_PATHS = {
  preprocess: `${PRO_V3_HTTP_PREFIX}/keyframes/preprocess`,
  extract: `${PRO_V3_HTTP_PREFIX}/keyframes/extract`,
  refine: `${PRO_V3_HTTP_PREFIX}/keyframes/refine`,
  analyze: `${PRO_V3_HTTP_PREFIX}/keyframes/analyze`,
} as const;

/** Shipped default when env has no MODAL_BACKEND_URL; keep in sync with Modal deployment. */
export const DEFAULT_PROV3_MODAL_URL = "https://dytsui--stellar-ai-fastapi-app.modal.run";

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
    const n = normalizeProHttpApiBase(u);
    if (!n || seen.has(n)) continue;
    seen.add(n);
    out.push(n);
  }
  return out;
}

/**
 * Pro Modal/HTTP 基址必须是 **origin**（如 `https://xxx.modal.run`），不要带路由后缀。
 * 误配 `.../pro-v3` 或 `.../pro-v3/analyze` 会导致请求 `.../pro-v3/pro-v3/analyze` → 404。
 */
export function normalizeProHttpApiBase(url: string): string {
  let u = url.trim().replace(/\/+$/, "");
  u = u.replace(/\/pro-v3\/analyze$/i, "");
  u = u.replace(/\/pro-v3$/i, "");
  return u.replace(/\/+$/, "");
}

export function buildProv3ModalUrlList(
  getCfEnv: (key: string) => string,
  isCn: boolean,
  processEnv: Record<string, string | undefined> = process.env,
): string[] {
  const primary = normalizeProHttpApiBase(
    getCfEnv("MODAL_BACKEND_URL") || processEnv.MODAL_BACKEND_URL || DEFAULT_PROV3_MODAL_URL,
  );
  const cnExtra = isCn
    ? splitCsv(getCfEnv("MODAL_BACKEND_CN_EXTRA") || processEnv.MODAL_BACKEND_CN_EXTRA || "").map(
        normalizeProHttpApiBase,
      )
    : [];
  const globalFbNorm = splitCsv(
    getCfEnv("MODAL_BACKEND_FALLBACKS") || processEnv.MODAL_BACKEND_FALLBACKS || "",
  ).map(normalizeProHttpApiBase);
  return dedupePreserveOrder([...cnExtra, primary, ...globalFbNorm]);
}

export function buildProv3BackendUrlList(
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

export function normalizeProv3UrlListsFromPrecheck(data: {
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

/**
 * Browser origin for ``POST {origin}/analyze/lite``.
 * Priority: ``NEXT_PUBLIC_MODAL_BACKEND_URL`` (same as Pro) → legacy ``NEXT_PUBLIC_LITE_BACKEND_URL`` → shipped default Modal base.
 */
export function resolveLiteAnalyzeClientOrigin(
  processEnv: Record<string, string | undefined> = process.env,
): string {
  const modalPub = normalizeProHttpApiBase(processEnv.NEXT_PUBLIC_MODAL_BACKEND_URL || "");
  if (modalPub) return modalPub;
  const litePub = normalizeProHttpApiBase(processEnv.NEXT_PUBLIC_LITE_BACKEND_URL || "");
  if (litePub) return litePub;
  return normalizeProHttpApiBase(DEFAULT_PROV3_MODAL_URL);
}

/**
 * Edge / server origin for Lite analyze upstream (CN proxy or ops).
 * Priority: ``LITE_BACKEND_URL`` (dedicated override) → Pro Modal list (``MODAL_BACKEND_URL`` + fallbacks) → same public env order as client → default.
 */
export function resolveLiteAnalyzeUpstreamBase(
  getCfEnv: (key: string) => string,
  options?: { clientCountryCode?: string },
  processEnv: Record<string, string | undefined> = process.env,
): string {
  const secretLite = (getCfEnv("LITE_BACKEND_URL") || processEnv.LITE_BACKEND_URL || "").trim();
  if (secretLite) {
    return normalizeProHttpApiBase(secretLite);
  }
  const cn = (options?.clientCountryCode || "").trim().toUpperCase() === "CN";
  const modalPrimary = buildProv3ModalUrlList(getCfEnv, cn, processEnv)[0];
  if (modalPrimary) {
    return modalPrimary;
  }
  const modalPub = normalizeProHttpApiBase(processEnv.NEXT_PUBLIC_MODAL_BACKEND_URL || "");
  if (modalPub) return modalPub;
  const litePub = normalizeProHttpApiBase(processEnv.NEXT_PUBLIC_LITE_BACKEND_URL || "");
  if (litePub) return litePub;
  return normalizeProHttpApiBase(DEFAULT_PROV3_MODAL_URL);
}

/** Full URL for ``POST /pro-v3/analyze`` given an API origin (Modal or Render). */
export function buildProV3AnalyzeRequestUrl(apiBaseOrigin: string): string {
  return `${normalizeProHttpApiBase(apiBaseOrigin)}${PRO_V3_ANALYZE_PATH}`;
}
