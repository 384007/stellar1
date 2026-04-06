"use client";

import {
  isProv3StrictMediaPolicyResult,
  prov3HistoryKeyframesIncomplete,
  type Prov3ResultLike,
} from "@/lib/prov3-keyframe-media";
import { normalizeProv3MediaInRaw } from "@/lib/prov3-media-url";
import { getAnalysisVideoBlob } from "@/lib/video-store";

export type LoadedProAnalysis = {
  raw: Record<string, unknown>;
  videoBlob: Blob | null;
};

/**
 * Load a Pro analysis by id: local stellar_history_local first, then GET /api/history/:id (+ optional video blob).
 */
export async function loadProAnalysisById(
  analysisId: string,
  token: string | null,
): Promise<LoadedProAnalysis | null> {
  const id = analysisId.trim();
  if (!id || id.length > 128) return null;

  let incompleteLocalFallback: LoadedProAnalysis | null = null;

  try {
    const key = "stellar_history_local";
    const existing = JSON.parse(localStorage.getItem(key) || "[]") as Array<{
      id: string;
      type?: string;
      result_json?: string;
    }>;
    const hit = existing.find((r) => r.id === id && r.type === "pro");
    if (hit?.result_json) {
      const raw = JSON.parse(hit.result_json) as Record<string, unknown>;
      raw.analysis_id = String(raw.analysis_id || id);
      normalizeProv3MediaInRaw(raw);
      const videoBlob = await getAnalysisVideoBlob(id);
      const needCloudDetail =
        Boolean(token && !token.startsWith("local-")) &&
        isProv3StrictMediaPolicyResult(raw as Prov3ResultLike) &&
        prov3HistoryKeyframesIncomplete(raw as Prov3ResultLike);
      if (!needCloudDetail) {
        return { raw, videoBlob };
      }
      incompleteLocalFallback = { raw, videoBlob };
    }
  } catch {
    /* ignore */
  }

  if (token && !token.startsWith("local-")) {
    try {
      const res = await fetch(`/api/history/${encodeURIComponent(id)}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const row = (await res.json()) as {
          result_json?: string;
          video_url?: string;
          analysis_video_url?: string;
        };
        const raw = JSON.parse(String(row.result_json || "{}")) as Record<string, unknown>;
        raw.analysis_id = String(raw.analysis_id || id);
        normalizeProv3MediaInRaw(raw);
        if (row.video_url) raw.video_url = row.video_url;
        if (row.analysis_video_url) raw.analysis_video_url = row.analysis_video_url;
        // Row overlays can reintroduce relative / wrong-host prov3 paths — normalize again.
        normalizeProv3MediaInRaw(raw);

        let videoBlob: Blob | null = null;
        try {
          const vr = await fetch(`/api/history/video/${encodeURIComponent(id)}`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (vr.ok) videoBlob = await vr.blob();
        } catch {
          /* ignore */
        }
        return { raw, videoBlob };
      }
    } catch {
      /* ignore */
    }
  }

  return incompleteLocalFallback;
}
