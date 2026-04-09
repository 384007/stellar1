/**
 * Pro v3 / R2 media: rewrite to same-origin ``/api/cdn/p`` so the browser never loads Modal/R2 hosts directly.
 * Only ``m=1&p=`` (Modal ``/pro-v3/media/``) and ``r=1&p=`` (``/prov3-media/``) — no reversible encoding of full upstream URLs.
 */

function cdnModalPath(pathWithSearch: string): string {
  return `/api/cdn/p?m=1&p=${encodeURIComponent(pathWithSearch)}`;
}

function cdnR2Path(pathWithSearch: string): string {
  return `/api/cdn/p?r=1&p=${encodeURIComponent(pathWithSearch)}`;
}

/** ``/prov3-media/...`` segment inside pathname (any host). */
function pathFromProv3MediaPathname(pathname: string, search: string): string | null {
  const low = pathname.toLowerCase();
  const idx = low.indexOf("/prov3-media/");
  if (idx === -1) return null;
  return pathname.slice(idx) + search;
}

/**
 * True when URL is same-origin proxy or legacy absolute product media — safe to skip redundant `<img>` probes.
 */
export function isProv3DurableR2ProductUrl(url: string): boolean {
  const u = String(url ?? "").trim();
  if (!u) return false;
  if (u.startsWith("/api/cdn/p?")) return true;
  try {
    const parsed = /^https?:\/\//i.test(u) ? new URL(u) : new URL(u, "https://invalid.local");
    const path = parsed.pathname.toLowerCase();
    return path.includes("/prov3-media/") || path.includes("/pro-v3/media/");
  } catch {
    return false;
  }
}

/** Rewrite Modal / R2 media references to same-origin CDN proxy paths. */
export function resolveProv3ProductMediaUrl(url: string | undefined | null): string {
  const u = String(url ?? "").trim();
  if (!u) return "";
  if (/^https?:\/\//i.test(u)) {
    try {
      const parsed = new URL(u);
      const ps = parsed.pathname + parsed.search;
      if (parsed.pathname.startsWith("/pro-v3/media/")) {
        return cdnModalPath(ps);
      }
      const r2Seg = pathFromProv3MediaPathname(parsed.pathname, parsed.search);
      if (r2Seg) {
        return cdnR2Path(r2Seg);
      }
    } catch {
      return u;
    }
    return u;
  }
  if (u.startsWith("/pro-v3/media/")) {
    return cdnModalPath(u);
  }
  if (u.startsWith("/prov3-media/")) {
    return cdnR2Path(u);
  }
  return u;
}

/** In-place: fix keyframe + top-level media strings for prov3 payloads (history, deep link, API). */
export function normalizeProv3MediaInRaw(raw: Record<string, unknown>): void {
  const pip = String(raw.pipeline ?? "");
  const aid = String(raw.analysis_id ?? "");
  if (pip !== "prov3" && !aid.startsWith("prov3_")) return;

  const topKeys = [
    "analysis_video_url",
    "video_url",
    "original_video_url",
    "playback_video_url",
    "screen_cropped_video_url",
    "contact_sheet_url",
    "screen_clean_video_url",
  ] as const;
  for (const key of topKeys) {
    const v = raw[key];
    if (typeof v === "string" && v.trim()) {
      const n = resolveProv3ProductMediaUrl(v);
      if (n) raw[key] = n;
    }
  }

  const lists = ["keyframes", "official_phase_keyframes", "preview_keyframes", "keyframe_images"] as const;
  for (const lk of lists) {
    const arr = raw[lk];
    if (!Array.isArray(arr)) continue;
    for (const row of arr) {
      if (!row || typeof row !== "object" || Array.isArray(row)) continue;
      const rec = row as Record<string, unknown>;
      const ku = rec.keyframe_image_url;
      if (typeof ku === "string" && ku.trim()) {
        const n = resolveProv3ProductMediaUrl(ku);
        if (n) rec.keyframe_image_url = n;
      }
    }
  }
}
