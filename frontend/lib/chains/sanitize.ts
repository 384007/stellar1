/**
 * Single source of truth: strip implementation details from JSON sent to browsers.
 * Modal/FastAPI logs may still contain provider/key info; responses must not.
 */

/** Keys removed at any depth in plain objects (not applied inside string values). */
export const PRODUCT_RESPONSE_DENYLIST = new Set([
  /** Lab / Lite product label — still omit from browser JSON per private-API plan */
  "ai_provider",
  "model_used",
  "host_used",
  "key_used",
  "gemini_key_index",
  "modal_url",
  "backend_url",
  "modal_urls",
  "backend_urls",
  /** Request flag to upstream — not for browser product JSON */
  "cn_network_hint",
  "file_uri",
  "runtime",
  "ai_key",
  "prov3_debug",
  "internal_route",
  "upstream_host",
  /** CF→Gemini forward internal slot; must not appear in browser JSON */
  "_stellar_key_slot",
  /** Common provider / SDK metadata */
  "prompt_feedback",
  "promptFeedback",
  "usage_metadata",
  "usageMetadata",
  "citation_metadata",
  "citationMetadata",
  "safety_ratings",
  "safetyRatings",
  "response_mime_type",
  "responseMimeType",
  "model_version",
  "modelVersion",
  "thought_signature",
  "thoughtSignature",
]);

/** Browser-visible keys starting with `_` are treated as internal unless listed here. */
export const PRODUCT_UNDERSCORE_PUBLIC_KEYS = new Set<string>(["_plus_usage"]);

export type ProductChain = "analysis" | "lab" | "plus" | "render" | "record" | "share" | "auth" | "generic";

/**
 * Deep-clone plain JSON-ish structures and drop denylisted keys.
 */
export function sanitizeProductJson(value: unknown, _chain: ProductChain = "generic"): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value !== "object") return value;
  if (Array.isArray(value)) {
    return value.map((x) => sanitizeProductJson(x, _chain));
  }
  const obj = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (PRODUCT_RESPONSE_DENYLIST.has(k)) continue;
    if (k.startsWith("_") && !PRODUCT_UNDERSCORE_PUBLIC_KEYS.has(k)) continue;
    out[k] = sanitizeProductJson(v, _chain);
  }
  return out;
}

/** Strip known internal keys from `prediction` objects (nested). */
export function sanitizePredictionObject(pred: Record<string, unknown> | null | undefined): Record<string, unknown> {
  if (!pred || typeof pred !== "object") return {};
  const deny = new Set([
    ...PRODUCT_RESPONSE_DENYLIST,
    "distance_debug",
    "fusion_weights",
  ]);
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(pred)) {
    if (deny.has(k)) continue;
    if (k.startsWith("_") && !PRODUCT_UNDERSCORE_PUBLIC_KEYS.has(k)) continue;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      out[k] = sanitizeProductJson(v, "analysis");
    } else if (Array.isArray(v)) {
      out[k] = sanitizeProductJson(v, "analysis");
    } else {
      out[k] = v;
    }
  }
  return out;
}
