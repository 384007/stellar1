import "server-only";

import { PRO_V3_ANALYZE_PATH } from "@/lib/prov3-endpoints";

/** Shipped default when env has no MODAL_BACKEND_URL; keep in sync with Modal deployment. */
export const DEFAULT_PROV3_MODAL_URL = "https://suitsui72--stellar-ai-fastapi-app.modal.run";

/**
 * Lite-only Modal worker (`modal deploy modal_app_lite` → app `stellar-ai-lite`, ASGI `fastapi_app_lite`).
 * Override with ``LITE_BACKEND_URL`` in Cloudflare Pages if your workspace slug differs.
 */
export const DEFAULT_LITE_MODAL_URL = "https://suitsui72--stellar-ai-lite-fastapi-app-lite.modal.run";

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

/**
 * Render / secondary API bases (server-only). Uses ``BACKEND_URL`` / bindings only — never ``NEXT_PUBLIC_*``.
 */
export function buildProv3BackendUrlList(
  getCfEnv: (key: string) => string,
  isCn: boolean,
  primaryFallback: string,
  processEnv: Record<string, string | undefined> = process.env,
): string[] {
  const primary = (
    getCfEnv("BACKEND_URL") ||
    processEnv.BACKEND_URL ||
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
 * Pro / Plus stack (`modal deploy modal_app.py` → ``main:app``). CN-aware fallbacks.
 * Use for routes that only exist on the full app (e.g. ``/analyze/plus/posture-video``).
 */
export function resolveProModalUpstreamBase(
  getCfEnv: (key: string) => string,
  options?: { clientCountryCode?: string },
  processEnv: Record<string, string | undefined> = process.env,
): string {
  const cn = (options?.clientCountryCode || "").trim().toUpperCase() === "CN";
  const modalPrimary = buildProv3ModalUrlList(getCfEnv, cn, processEnv)[0];
  if (modalPrimary) {
    return modalPrimary;
  }
  return normalizeProHttpApiBase(DEFAULT_PROV3_MODAL_URL);
}

/**
 * Lite-only worker (`modal deploy modal_app_lite` → ``main_lite:app``): ``/analyze/lite``,
 * ``/analyze/vision-classic``, ``/analyze/recalculate``, ``/analyze/club-detect``,
 * ``/analyze/club-detect-batch``. Video Lite product path should rely on ``/analyze/lite`` only (one run).
 *
 * Priority: ``LITE_BACKEND_URL`` → ``DEFAULT_LITE_MODAL_URL`` (never the Pro Modal URL).
 */
export function resolveLiteModalWorkerBase(
  getCfEnv: (key: string) => string,
  processEnv: Record<string, string | undefined> = process.env,
): string {
  const configured = (getCfEnv("LITE_BACKEND_URL") || processEnv.LITE_BACKEND_URL || "").trim();
  // Preview deployments may retain an obsolete workspace URL. Only accept the
  // active Lite worker origin; otherwise use the shipped active default.
  if (configured.includes("suitsui72--stellar-ai-lite-fastapi-app-lite.modal.run")) {
    return normalizeProHttpApiBase(configured);
  }
  return normalizeProHttpApiBase(DEFAULT_LITE_MODAL_URL);
}

/**
 * Lite Modal worker only (same as ``resolveLiteModalWorkerBase``). ``clientCountryCode`` is ignored.
 */
export function resolveLiteAnalyzeUpstreamBase(
  getCfEnv: (key: string) => string,
  _options?: { clientCountryCode?: string },
  processEnv: Record<string, string | undefined> = process.env,
): string {
  return resolveLiteModalWorkerBase(getCfEnv, processEnv);
}

export function buildProV3AnalyzeRequestUrl(apiBaseOrigin: string): string {
  return `${normalizeProHttpApiBase(apiBaseOrigin)}${PRO_V3_ANALYZE_PATH}`;
}

/** First Modal origin in the configured list (CN-aware). */
export function prov3ModalPrimaryOrigin(
  getCfEnv: (key: string) => string,
  clientCountryCode: string,
  processEnv: Record<string, string | undefined> = process.env,
): string {
  const cn = clientCountryCode.trim().toUpperCase() === "CN";
  const list = buildProv3ModalUrlList(getCfEnv, cn, processEnv);
  const first = list[0] ? normalizeProHttpApiBase(list[0]) : "";
  return first;
}
