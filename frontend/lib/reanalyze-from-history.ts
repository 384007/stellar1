import { resolveProv3ProductMediaUrl } from "@/lib/prov3-media-url";
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

const REMOTE_VIDEO_FETCH_MS = 12_000;

function abortSignalForMs(ms: number): AbortSignal | undefined {
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
    return AbortSignal.timeout(ms);
  }
  return undefined;
}

/** Worker-local ``/pro-v3/media/…`` (ephemeral). R2 product URLs use ``/prov3-media/…``. */
function isEphemeralProv3WorkerMediaUrl(url: string): boolean {
  const lower = url.trim().toLowerCase();
  if (!lower.includes("/pro-v3/media/")) return false;
  if (lower.includes("/prov3-media/")) return false;
  return true;
}

async function tryCachedOrHistoryVideoBlob(analysisId: string): Promise<Blob | null> {
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

/**
 * History「重新分析」取视频：
 * - 解析相对 / 错源的 prov3 URL（与展示层一致）
 * - **不再**对同一资源先发 HEAD 再 GET（避免 Modal 上出现「两条 media run + 一条 analyze」）
 * - 若首选时间线是易失效的 worker ``/pro-v3/media/…``，先尝试 IndexedDB / 同源历史视频 API，再尝试远程
 * - 否则保持：远程（analysis → video）→ 本地缓存 → 历史 API
 *
 * 参数名与历史页 `queueReanalyzeFromHistory` 一致：`videoUrl` / `analysisVideoUrl`。
 */
export async function fetchVideoBlobForHistoryReanalyze(
  analysisId: string,
  videoUrl?: string,
  analysisVideoUrl?: string,
): Promise<Blob | null> {
  const rawA = (analysisVideoUrl || "").trim();
  const rawV = (videoUrl || "").trim();
  const remoteCandidates: string[] = [];
  const seen = new Set<string>();
  for (const raw of [rawA, rawV]) {
    if (!raw) continue;
    const u = resolveProv3ProductMediaUrl(raw);
    if (!u || seen.has(u)) continue;
    seen.add(u);
    remoteCandidates.push(u);
  }

  const firstRemote = remoteCandidates[0] || "";
  if (firstRemote && isEphemeralProv3WorkerMediaUrl(firstRemote)) {
    const early = await tryCachedOrHistoryVideoBlob(analysisId);
    if (early) return early;
  }

  for (const u of remoteCandidates) {
    if (!/^https?:\/\//i.test(u) && !u.startsWith("/")) continue;
    try {
      const sig = abortSignalForMs(REMOTE_VIDEO_FETCH_MS);
      const r = await fetch(u, {
        method: "GET",
        mode: "cors",
        cache: "no-store",
        ...(sig ? { signal: sig } : {}),
      });
      if (r.ok) {
        const blob = await r.blob();
        if (blob.size > 0) return blob;
      }
    } catch {
      /* fall through */
    }
  }

  return tryCachedOrHistoryVideoBlob(analysisId);
}
