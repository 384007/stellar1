"use client";

import { useEffect, useState } from "react";
import {
  isProv3DurableR2ProductUrl,
  normalizeProv3MediaInRaw,
  resolveProv3ProductMediaUrl,
} from "@/lib/prov3-media-url";

/** Session tab: URLs that already failed probe — avoid repeated img decode attempts on re-render. */
const prov3MediaProbeFailedUrls = new Set<string>();

export function markProv3MediaUrlsProbeFailed(urls: string[]): void {
  for (const u of urls) {
    const s = u.trim();
    if (s) prov3MediaProbeFailedUrls.add(s);
  }
}

export function prov3MediaUrlsHadProbeFailure(urls: string[]): boolean {
  return urls.some((u) => prov3MediaProbeFailedUrls.has(u.trim()));
}

/** User-facing copy when true-240 JPG URLs are missing or unloadable */
export const PROV3_KEYFRAME_MEDIA_FAIL_ZH =
  "该结果不满足真 240 分析时间线关键帧展示标准（缺少可访问的 timeline JPG）。请重新分析。";
export const PROV3_KEYFRAME_MEDIA_FAIL_EN =
  "This result does not meet the true-240 analysis timeline keyframe standard (missing reachable timeline JPGs). Please re-analyze.";

export type Prov3KeyframeRowLike = {
  keyframe_image_url?: string;
  image_base64?: string;
};

export type Prov3ResultLike = {
  pipeline?: string;
  analysis_id?: string;
  final_status?: string;
  analysis_trust?: string;
  trust_level?: string;
  low_trust_preview_only?: boolean;
  keyframes?: unknown[];
  official_phase_keyframes?: unknown[];
  preview_keyframes?: unknown[];
  keyframes_strip?: {
    timeline?: string;
    thumbnails_from_analysis_video?: boolean;
  };
};

/** Pro v3 product responses that must never use base64 as user-visible keyframe source */
export function isProv3StrictMediaPolicyResult(r: Prov3ResultLike | null | undefined): boolean {
  if (!r) return false;
  if (String(r.pipeline ?? "") === "prov3") return true;
  if (String(r.analysis_id ?? "").startsWith("prov3_")) return true;
  return false;
}

export function prov3DisplayKeyframeRows(r: Prov3ResultLike): Prov3KeyframeRowLike[] {
  const lowTrust =
    String(r.final_status ?? "") !== "pass" ||
    String(r.analysis_trust ?? r.trust_level ?? "") === "low_trust" ||
    r.low_trust_preview_only === true;
  const keyframes = Array.isArray(r.keyframes) ? (r.keyframes as Prov3KeyframeRowLike[]) : [];
  const official = Array.isArray(r.official_phase_keyframes)
    ? (r.official_phase_keyframes as Prov3KeyframeRowLike[])
    : [];
  const preview = Array.isArray(r.preview_keyframes) ? (r.preview_keyframes as Prov3KeyframeRowLike[]) : [];
  if (lowTrust) {
    return preview.length ? preview : [];
  }
  return official.length ? official : [];
}

export function isValidProv3KeyframeImageUrl(url: string): boolean {
  const u = url.trim();
  if (u.length < 12) return false;
  if (!/^https?:\/\//i.test(u) && !u.startsWith("/")) return false;
  const path = u.split("?")[0].toLowerCase();
  return path.endsWith(".jpg") || path.endsWith(".jpeg");
}

/** Static check before decode: every display row must have a plausible timeline JPG URL */
export function prov3RowsMeetStaticUrlPolicy(rows: Prov3KeyframeRowLike[]): boolean {
  if (!rows.length) return false;
  return rows.every((k) => isValidProv3KeyframeImageUrl(String(k?.keyframe_image_url ?? "")));
}

export async function verifyProv3KeyframeUrlsLoad(
  urls: string[],
  timeoutMs = 20000,
): Promise<{ ok: true } | { ok: false; url?: string }> {
  const resolved = urls
    .map((u) => resolveProv3ProductMediaUrl(u.trim()))
    .filter((u) => u.length > 0);
  if (resolved.length > 0 && resolved.every((u) => isProv3DurableR2ProductUrl(u))) {
    return { ok: true };
  }
  // One probe per URL: `<img>` only. A separate HEAD doubled Modal traffic (HEAD+GET) on the same path.
  for (const url of resolved) {
    const decoded = await new Promise<boolean>((resolve) => {
      const img = new Image();
      const t = window.setTimeout(() => {
        img.onload = null;
        img.onerror = null;
        resolve(false);
      }, timeoutMs);
      img.onload = () => {
        window.clearTimeout(t);
        resolve(true);
      };
      img.onerror = () => {
        window.clearTimeout(t);
        resolve(false);
      };
      img.src = url;
    });
    if (!decoded) {
      markProv3MediaUrlsProbeFailed(resolved);
      return { ok: false, url };
    }
  }
  return { ok: true };
}

/** Merge: prefer JSON that has full prov3 timeline JPG URLs over base64-heavy local rows */
export function prov3HistoryMergePayloadScore(json: string): number {
  try {
    const p = JSON.parse(json) as Record<string, unknown>;
    normalizeProv3MediaInRaw(p);
    if (!isProv3StrictMediaPolicyResult(p as Prov3ResultLike)) return 0;
    const rows = prov3DisplayKeyframeRows(p as Prov3ResultLike);
    let score = 0;
    for (const k of rows) {
      const u = String((k as Prov3KeyframeRowLike).keyframe_image_url ?? "").trim();
      if (isValidProv3KeyframeImageUrl(u)) score += 50_000 + Math.min(u.length, 2000);
    }
    const av = String(p.analysis_video_url ?? "").trim();
    if (av.length > 12) score += 25_000;
    return score;
  } catch {
    return 0;
  }
}

export type Prov3KeyframeGateState = "idle" | "checking" | "ok" | "fail";

/**
 * For prov3 results: static URL check + image decode probe. Non-prov3 → ok immediately.
 */
export function useProv3KeyframeDisplayGate(result: Prov3ResultLike): Prov3KeyframeGateState {
  const [state, setState] = useState<Prov3KeyframeGateState>("idle");
  const aid = String(result.analysis_id ?? "");
  const pipeline = String(result.pipeline ?? "");
  const trustKey = `${result.final_status ?? ""}|${result.analysis_trust ?? result.trust_level ?? ""}|${Boolean(result.low_trust_preview_only)}`;
  const urlSig = JSON.stringify(
    prov3DisplayKeyframeRows(result).map((k) =>
      resolveProv3ProductMediaUrl(String(k.keyframe_image_url ?? "").trim()),
    ),
  );

  useEffect(() => {
    if (!isProv3StrictMediaPolicyResult(result)) {
      setState("ok");
      return;
    }
    const rows = prov3DisplayKeyframeRows(result);
    if (!prov3RowsMeetStaticUrlPolicy(rows)) {
      setState("fail");
      return;
    }
    const urls = rows
      .map((row) => resolveProv3ProductMediaUrl(String(row.keyframe_image_url ?? "").trim()))
      .filter((u) => u.length > 0);
    if (urls.length < rows.length) {
      setState("fail");
      return;
    }
    if (urls.length > 0 && urls.every((u) => isProv3DurableR2ProductUrl(u))) {
      setState("ok");
      return;
    }
    if (prov3MediaUrlsHadProbeFailure(urls)) {
      setState("fail");
      return;
    }
    setState("checking");
    let cancelled = false;
    void verifyProv3KeyframeUrlsLoad(urls).then((res) => {
      if (cancelled) return;
      setState(res.ok ? "ok" : "fail");
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- trustKey/urlSig capture row + URL identity
  }, [aid, pipeline, trustKey, urlSig]);

  return state;
}

/** History list/detail: prov3 records need timeline JPG URLs, not base64 thumbnails */
export function prov3HistoryKeyframesIncomplete(parsed: Prov3ResultLike | null | undefined): boolean {
  if (!parsed || !isProv3StrictMediaPolicyResult(parsed)) return false;
  const rows = prov3DisplayKeyframeRows(parsed);
  if (rows.length < 6) return true;
  return !prov3RowsMeetStaticUrlPolicy(rows);
}
