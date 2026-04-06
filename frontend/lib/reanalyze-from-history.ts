import { getAnalysisVideoBlob } from "@/lib/video-store";

export const REANALYZE_FROM_HISTORY_KEY = "stellar_reanalyze_from_history_v1";

export type ReanalyzeFromHistoryPayload = {
  analysisId: string;
  page: "analyze" | "pro" | "plus" | "shot-lab";
  /** 仅 `/analyze`：与历史记录类型对应的模式（Pro 记录走 `/pro`，此处一般为 lite） */
  analysisMode?: "lite" | "pro";
  videoUrl?: string;
  analysisVideoUrl?: string;
  /**
   * Pro v3：原分析为屏幕模式时保留；再次分析时传给 `screen_mode` 表单字段。
   */
  prov3ScreenMode?: boolean;
};

/** 从重新分析 payload 解析是否对屏。 */
export function reanalyzePayloadProv3ScreenMode(p: ReanalyzeFromHistoryPayload): boolean {
  return Boolean(p.prov3ScreenMode);
}

export function queueReanalyzeFromHistory(p: ReanalyzeFromHistoryPayload): void {
  if (typeof sessionStorage === "undefined") return;
  sessionStorage.setItem(REANALYZE_FROM_HISTORY_KEY, JSON.stringify(p));
}

export function consumeReanalyzeFromHistoryPayload(): ReanalyzeFromHistoryPayload | null {
  if (typeof sessionStorage === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(REANALYZE_FROM_HISTORY_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(REANALYZE_FROM_HISTORY_KEY);
    const p = JSON.parse(raw) as ReanalyzeFromHistoryPayload;
    if (!p?.analysisId || !p?.page) return null;
    if (p.page !== "analyze" && p.page !== "pro" && p.page !== "plus" && p.page !== "shot-lab") {
      return null;
    }
    return p;
  } catch {
    return null;
  }
}

function extFromMime(mime: string): string {
  const m = (mime || "").toLowerCase();
  if (m.includes("webm")) return "webm";
  if (m.includes("quicktime")) return "mov";
  if (m.includes("png")) return "png";
  if (m.startsWith("image/")) return "jpg";
  return "mp4";
}

export function reanalyzeHistoryFilename(blob: Blob): string {
  return `history-reanalyze.${extFromMime(blob.type)}`;
}

/**
 * Shot Lab：从 R2 拉取该 job 备份的源媒体（需登录；与 analyses 的 IndexedDB 无关）。
 */
export async function fetchLabVideoBlobForReanalyze(jobId: string): Promise<Blob | null> {
  const token =
    typeof localStorage !== "undefined" ? localStorage.getItem("stellar_token") : null;
  if (!token) return null;
  try {
    const r = await fetch(`/api/lab/video/${encodeURIComponent(jobId)}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (r.ok) {
      const blob = await r.blob();
      if (blob.size > 0) return blob;
    }
  } catch {
    /* ignore */
  }
  return null;
}

/**
 * 按顺序：直链 URL → 本机 IndexedDB → 登录用户 R2（/api/history/video）
 */
export async function fetchVideoBlobForHistoryReanalyze(
  analysisId: string,
  videoUrl?: string,
  analysisVideoUrl?: string,
): Promise<Blob | null> {
  const candidates = [(analysisVideoUrl || "").trim(), (videoUrl || "").trim()].filter(Boolean);
  for (const u of candidates) {
    if (/^https?:\/\//i.test(u) || u.startsWith("/")) {
      try {
        const r = await fetch(u);
        if (r.ok) {
          const blob = await r.blob();
          if (blob.size > 0) return blob;
        }
      } catch {
        /* fall through */
      }
    }
  }

  try {
    const fromIdb = await getAnalysisVideoBlob(analysisId);
    if (fromIdb && fromIdb.size > 0) return fromIdb;
  } catch {
    /* ignore */
  }

  const token =
    typeof localStorage !== "undefined" ? localStorage.getItem("stellar_token") : null;
  if (token && !token.startsWith("local-")) {
    try {
      const r = await fetch(`/api/history/video/${encodeURIComponent(analysisId)}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (r.ok) {
        const blob = await r.blob();
        if (blob.size > 0) return blob;
      }
    } catch {
      /* ignore */
    }
  }
  return null;
}
