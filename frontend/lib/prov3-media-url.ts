/**
 * Pro v3 product media URLs must be absolute and point at durable storage (R2) or at least
 * at the Modal API origin — never path-only `/pro-v3/media/...` resolved against Pages (404).
 */

import { DEFAULT_PROV3_MODAL_URL, normalizeProHttpApiBase } from "@/lib/prov3-endpoints";

function modalApiOriginForMediaResolution(): string {
  if (typeof process !== "undefined" && process.env.NEXT_PUBLIC_MODAL_BACKEND_URL?.trim()) {
    return normalizeProHttpApiBase(process.env.NEXT_PUBLIC_MODAL_BACKEND_URL.trim());
  }
  return normalizeProHttpApiBase(DEFAULT_PROV3_MODAL_URL);
}

/**
 * True when URL targets R2-backed prov3 objects (`…/prov3-media/{analysis_id}/…`).
 * Backend HeadObject-verifies these before returning analyze success — safe to skip client
 * HEAD/img probes that can false-fail while `<img src>` still works.
 */
export function isProv3DurableR2ProductUrl(url: string): boolean {
  const u = String(url ?? "").trim();
  if (!u) return false;
  try {
    const parsed = /^https?:\/\//i.test(u) ? new URL(u) : new URL(u, "https://invalid.local");
    return parsed.pathname.includes("/prov3-media/");
  } catch {
    return false;
  }
}

/** Turn relative prov3 media paths into absolute Modal URLs (ephemeral fallback for old rows). */
export function resolveProv3ProductMediaUrl(url: string | undefined | null): string {
  const u = String(url ?? "").trim();
  if (!u) return "";
  if (/^https?:\/\//i.test(u)) {
    try {
      const parsed = new URL(u);
      // Wrong origin (e.g. Pages) stored in history — same path must hit Modal / R2 redirect, not 404 on Pages.
      if (parsed.pathname.startsWith("/pro-v3/media/")) {
        const origin = modalApiOriginForMediaResolution();
        return `${origin}${parsed.pathname}${parsed.search}`;
      }
    } catch {
      return u;
    }
    return u;
  }
  if (u.startsWith("/pro-v3/media/")) {
    const origin = modalApiOriginForMediaResolution();
    return `${origin}${u}`;
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
