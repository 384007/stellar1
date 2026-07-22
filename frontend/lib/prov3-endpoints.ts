/**
 * Pro v3 — **browser-safe** path constants only.
 *
 * Upstream URL resolution, Modal/Render defaults, and env reads live in ``@/lib/server/prov3-upstream`` (server-only).
 */

/** Next.js Edge route: Pro JWT + optional ``network_hint`` (no upstream URLs in the JSON body). */
export const PRO_V3_EDGE_PRECHECK_PATH = "/api/prov3/precheck";

/** FastAPI Pro v3 API prefix on Modal/Render (``backend/routers/prov3_api.py``). */
export const PRO_V3_HTTP_PREFIX = "/pro-v3";
export const PRO_V3_ANALYZE_PATH = `${PRO_V3_HTTP_PREFIX}/analyze`;

/** Raw keyframe pipeline (no Gemini / no product media). Product use ``PRO_V3_ANALYZE_PATH``. */
export const PRO_V3_KEYFRAMES_PATHS = {
  preprocess: `${PRO_V3_HTTP_PREFIX}/keyframes/preprocess`,
  extract: `${PRO_V3_HTTP_PREFIX}/keyframes/extract`,
  refine: `${PRO_V3_HTTP_PREFIX}/keyframes/refine`,
  analyze: `${PRO_V3_HTTP_PREFIX}/keyframes/analyze`,
} as const;
