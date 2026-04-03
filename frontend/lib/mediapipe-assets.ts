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
 *   2. R2 direct        (pub-xxx.r2.dev — fast for non-CN users)
 *   3. jsdelivr         (global CDN with some China coverage)
 *   4. npmmirror        (Alibaba China mirror)
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

export function getMediaPipeAllowForeignFallback(): boolean {
  if (typeof process === "undefined") return true;
  const v = (process.env.NEXT_PUBLIC_MEDIAPIPE_ALLOW_FOREIGN_FALLBACK || "").toLowerCase();
  return v === "1" || v === "true" || v === "yes";
}

const v = MEDIAPIPE_TASKS_VISION_VERSION;

export function getMediaPipeBundleUrls(effectiveSelf?: string | null): string[] {
  const self = effectiveSelfBase(effectiveSelf);
  const urls: string[] = [
    "/mp/vision_bundle.mjs",
  ];
  if (self) urls.push(`${self}/vision_bundle.mjs`);
  urls.push(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${v}/vision_bundle.mjs`,
    `https://registry.npmmirror.com/@mediapipe/tasks-vision/${v}/files/vision_bundle.mjs`,
  );
  return urls;
}

export function getMediaPipeWasmBases(effectiveSelf?: string | null): string[] {
  const self = effectiveSelfBase(effectiveSelf);
  const bases: string[] = [
    "/mp/wasm",
  ];
  if (self) bases.push(`${self}/wasm`);
  bases.push(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${v}/wasm`,
    `https://registry.npmmirror.com/@mediapipe/tasks-vision/${v}/files/wasm`,
  );
  return bases;
}

const GOOGLE_FULL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";
const GOOGLE_LITE =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

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
  liteList.push(GOOGLE_LITE);
  fullList.push(GOOGLE_FULL);
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
