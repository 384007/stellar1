"use client";

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
      const videoBlob = await getAnalysisVideoBlob(id);
      return { raw, videoBlob };
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
        };
        const raw = JSON.parse(String(row.result_json || "{}")) as Record<string, unknown>;
        raw.analysis_id = String(raw.analysis_id || id);
        if (row.video_url) raw.video_url = row.video_url;

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

  return null;
}
