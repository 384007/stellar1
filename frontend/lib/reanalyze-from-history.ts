import { resolveProv3ProductMediaUrl } from "@/lib/prov3-media-url";
import { getAnalysisVideoBlob } from "@/lib/video-store";

export const REANALYZE_FROM_HISTORY_KEY = "stellar_reanalyze_from_history_v1";

/** Max age for a queued history reanalyze payload on `/pro` (requires `?reanalyze=1`). */
export const REANALYZE_FROM_HISTORY_TTL_MS = 30 * 60 * 1000;

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
  /** Set when queued; required for `/pro` TTL validation (see `reconcileProPageReanalyzeSession`). */
  createdAt?: number;
};

/** 从重新分析 payload 解析是否对屏。 */
export function reanalyzePayloadProv3ScreenMode(p: ReanalyzeFromHistoryPayload): boolean {
  return Boolean(p.prov3ScreenMode);
}

export function queueReanalyzeFromHistory(p: ReanalyzeFromHistoryPayload): void {
  if (typeof sessionStorage === "undefined") return;
  const withMeta: ReanalyzeFromHistoryPayload = { ...p, createdAt: Date.now() };
  sessionStorage.setItem(REANALYZE_FROM_HISTORY_KEY, JSON.stringify(withMeta));
}

export function isReanalyzePayloadFresh(
  p: ReanalyzeFromHistoryPayload,
  nowMs: number = Date.now(),
): boolean {
  const t = p.createdAt;
  if (typeof t !== "number" || !Number.isFinite(t) || t <= 0) return false;
  return nowMs - t <= REANALYZE_FROM_HISTORY_TTL_MS;
}

/**
 * `/pro` only: avoid firing analyze on plain load from stale sessionStorage.
 * - Without `?reanalyze=1`: remove a queued `page: "pro"` payload and return null.
 * - With `?reanalyze=1`: return payload only if present, valid shape, and within TTL; otherwise remove and return null.
 * Payloads for other pages are left untouched.
 */
export function reconcileProPageReanalyzeSession(
  explicitReanalyzeNav: boolean,
): ReanalyzeFromHistoryPayload | null {
  if (typeof sessionStorage === "undefined") return null;
  const raw = sessionStorage.getItem(REANALYZE_FROM_HISTORY_KEY);
  if (!raw) return null;

  let p: ReanalyzeFromHistoryPayload;
  try {
    p = JSON.parse(raw) as ReanalyzeFromHistoryPayload;
  } catch {
    sessionStorage.removeItem(REANALYZE_FROM_HISTORY_KEY);
    return null;
  }

  if (!p?.analysisId || !p?.page) {
    sessionStorage.removeItem(REANALYZE_FROM_HISTORY_KEY);
    return null;
  }
  if (p.page !== "analyze" && p.page !== "pro" && p.page !== "plus" && p.page !== "shot-lab") {
    sessionStorage.removeItem(REANALYZE_FROM_HISTORY_KEY);
    return null;
  }
  if (p.page !== "pro") return null;

  if (!explicitReanalyzeNav) {
    sessionStorage.removeItem(REANALYZE_FROM_HISTORY_KEY);
    return null;
  }
  if (!isReanalyzePayloadFresh(p)) {
    sessionStorage.removeItem(REANALYZE_FROM_HISTORY_KEY);
    return null;
  }

  sessionStorage.removeItem(REANALYZE_FROM_HISTORY_KEY);
  return p;
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
 * History「重新分析」取视频（供再次 **上传** 走分析；尽量减少 Modal worker 上的多余 GET）：
 * 1. **先** IndexedDB + ``/api/history/video``（与用户当初上传同源，无 worker media run）
 * 2. 远程 URL：**原始 ``videoUrl`` 优先于** 派生 ``analysisVideoUrl``（避免先拉 240 时间线再拉原片 → 多次 Modal run）
 * 3. 同一域内 URL：非 ephemeral 的持久链接优先；易失效的 ``/pro-v3/media/…`` worker 放最后、且仅必要时尝试一次
 * 4. 不再 HEAD+GET；每个 URL 最多一次 GET，首个成功即返回
 *
 * 参数名与历史页 `queueReanalyzeFromHistory` 一致：`videoUrl` / `analysisVideoUrl`。
 */
export async function fetchVideoBlobForHistoryReanalyze(
  analysisId: string,
  videoUrl?: string,
  analysisVideoUrl?: string,
): Promise<Blob | null> {
  const localFirst = await tryCachedOrHistoryVideoBlob(analysisId);
  if (localFirst && localFirst.size > 0) return localFirst;

  const rawA = (analysisVideoUrl || "").trim();
  const rawV = (videoUrl || "").trim();
  const remoteCandidates: string[] = [];
  const seen = new Set<string>();
  for (const raw of [rawV, rawA]) {
    if (!raw) continue;
    const u = resolveProv3ProductMediaUrl(raw);
    if (!u || seen.has(u)) continue;
    seen.add(u);
    remoteCandidates.push(u);
  }

  const stableFirst = remoteCandidates.filter((u) => !isEphemeralProv3WorkerMediaUrl(u));
  const ephemeralLast = remoteCandidates.filter((u) => isEphemeralProv3WorkerMediaUrl(u));
  const orderedRemote = [...stableFirst, ...ephemeralLast];

  for (const u of orderedRemote) {
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

  return null;
}
