/**
 * MediaPipe Tasks Vision asset URLs.
 *
 * All assets are copied into public/mp/ at build time from node_modules
 * (by scripts/fetch-mediapipe-models.mjs). This makes them ship as CF Pages
 * static assets on the same domain / CDN — no cross-origin, no CORS, works
 * from China because CF Pages CDN serves from Chinese PoPs.
 *
 * Loading order:
 *   1. /mp/…            (same-origin static, CF Pages CDN — primary)
 *   2. R2 direct        (optional NEXT_PUBLIC_MEDIAPIPE_CDN_BASE)
 *   3. Same-origin ``/api/mp-ext/…`` Edge proxy (jsdelivr + npmmirror server-side only — browser never hits them)
 */

export const MEDIAPIPE_TASKS_VISION_VERSION = "0.10.33";

function trimBase(raw: string | undefined): string | null {
  if (!raw) return null;
  const b = raw.trim().replace(/\/+$/, "");
  return b.length > 0 ? b : null;
}

export function getMediaPipeSelfHostBase(): string | null {
  if (typeof process === "undefined") return null;
  return trimBase(process.env.NEXT_PUBLIC_MEDIAPIPE_CDN_BASE);
}

let _resolvedSelfHostBase: string | null | undefined;

export async function resolveMediaPipeSelfHostBase(): Promise<string | null> {
  const build = getMediaPipeSelfHostBase();
  if (build) return build;
  if (typeof window === "undefined") return null;
  if (_resolvedSelfHostBase !== undefined) return _resolvedSelfHostBase;
  try {
    const res = await fetch("/api/mediapipe-cdn", { credentials: "same-origin" });
    if (!res.ok) { _resolvedSelfHostBase = null; return null; }
    const data = (await res.json()) as { base?: unknown };
    const raw = typeof data.base === "string" ? data.base.trim().replace(/\/+$/, "") : "";
    _resolvedSelfHostBase = raw.length > 0 ? raw : null;
  } catch { _resolvedSelfHostBase = null; }
  return _resolvedSelfHostBase;
}

function effectiveSelfBase(override?: string | null): string | null {
  return override !== undefined ? override : getMediaPipeSelfHostBase();
}

/** Default true: npmmirror/jsdelivr race works in CN without setting env (same as “骨架零配置”). */
export function getMediaPipeAllowForeignFallback(): boolean {
  if (typeof process === "undefined") return true;
  const v = (process.env.NEXT_PUBLIC_MEDIAPIPE_ALLOW_FOREIGN_FALLBACK || "").toLowerCase();
  if (v === "0" || v === "false" || v === "no" || v === "off") return false;
  return true;
}

export function getMediaPipeBundleUrls(effectiveSelf?: string | null): string[] {
  const self = effectiveSelfBase(effectiveSelf);
  const urls: string[] = [
    "/mp/vision_bundle.mjs",
  ];
  if (self) urls.push(`${self}/vision_bundle.mjs`);
  urls.push("/api/mp-ext/vision_bundle.mjs");
  return urls;
}

export function getMediaPipeWasmBases(effectiveSelf?: string | null): string[] {
  const self = effectiveSelfBase(effectiveSelf);
  const bases: string[] = [
    "/mp/wasm",
  ];
  if (self) bases.push(`${self}/wasm`);
  bases.push("/api/mp-ext/wasm");
  return bases;
}

const PROXY_FULL = "/api/mp-model?m=pose_landmarker_full";
const PROXY_LITE = "/api/mp-model?m=pose_landmarker_lite";

export function getPoseModelUrls(effectiveSelf?: string | null): { full: string[]; lite: string[] } {
  const self = effectiveSelfBase(effectiveSelf);
  const liteList = ["/mp/models/pose_landmarker_lite.task", PROXY_LITE];
  const fullList = [PROXY_FULL];
  if (self) {
    liteList.push(`${self}/models/pose_landmarker_lite.task`);
    fullList.push(`${self}/models/pose_landmarker_full.task`);
  }
  return { full: fullList, lite: liteList };
}

export function preloadPoseModel(): void {
  if (typeof window === "undefined") return;
  resolveMediaPipeSelfHostBase().then((selfBase) => {
    const { lite } = getPoseModelUrls(selfBase);
    for (const url of lite) {
      try {
        const link = document.createElement("link");
        link.rel = "prefetch";
        link.href = url;
        link.as = "fetch";
        link.crossOrigin = "anonymous";
        document.head.appendChild(link);
      } catch { /* non-critical */ }
      break;
    }
  }).catch(() => { /* non-critical */ });
}
