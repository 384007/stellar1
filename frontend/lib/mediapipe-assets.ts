/**
 * MediaPipe Tasks Vision asset URLs — **browser only uses same-origin paths.**
 *
 * 1. ``/mp/…`` — static from ``public/mp`` (build: ``scripts/fetch-mediapipe-models.mjs``)
 * 2. ``/api/mp-ext/…`` — Edge proxy (upstream hosts resolved server-side only)
 *
 * R2 / external CDN bases are not exposed to the client bundle.
 */

export const MEDIAPIPE_TASKS_VISION_VERSION = "0.10.33";

export function getMediaPipeBundleUrls(): string[] {
  return ["/mp/vision_bundle.mjs", "/api/mp-ext/vision_bundle.mjs"];
}

export function getMediaPipeWasmBases(): string[] {
  return ["/mp/wasm", "/api/mp-ext/wasm"];
}

const PROXY_FULL = "/api/mp-model?m=pose_landmarker_full";
const PROXY_LITE = "/api/mp-model?m=pose_landmarker_lite";

export function getPoseModelUrls(): { full: string[]; lite: string[] } {
  return {
    full: [PROXY_FULL],
    lite: ["/mp/models/pose_landmarker_lite.task", PROXY_LITE],
  };
}

export function preloadPoseModel(): void {
  if (typeof window === "undefined") return;
  const { lite } = getPoseModelUrls();
  for (const url of lite) {
    try {
      const link = document.createElement("link");
      link.rel = "prefetch";
      link.href = url;
      link.as = "fetch";
      link.crossOrigin = "anonymous";
      document.head.appendChild(link);
    } catch {
      /* non-critical */
    }
    break;
  }
}
