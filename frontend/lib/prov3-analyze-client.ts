import { makeFormData } from "@/lib/fetch-retry";
import { buildProV3AnalyzeRequestUrl, normalizeProHttpApiBase } from "@/lib/prov3-endpoints";

/** Modal/Render base URL used for the in-flight ``POST /pro-v3/analyze`` (for cancel). */
let _prov3ActiveAnalyzeBase: string | null = null;

export function clearProv3ActiveAnalyzeBase(): void {
  _prov3ActiveAnalyzeBase = null;
}

/** Ask the worker to cooperatively stop the current Pro analyze (same process as the active POST). */
export async function requestProv3AnalyzeCancel(
  authHeaders: Record<string, string>,
): Promise<{ ok: boolean }> {
  const raw = _prov3ActiveAnalyzeBase;
  if (!raw) return { ok: false };
  const base = normalizeProHttpApiBase(raw);
  if (!base) return { ok: false };
  const url = `${base.replace(/\/+$/, "")}/pro-v3/analyze/cancel`;
  try {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 12_000);
    const r = await fetch(url, {
      method: "POST",
      headers: { ...authHeaders },
      signal: ac.signal,
    });
    clearTimeout(timer);
    return { ok: r.ok };
  } catch {
    return { ok: false };
  }
}

export type RunProv3AnalyzeOptions = {
  modalUrls: string[];
  backendUrls: string[];
  cnNetworkHint: boolean;
  screenMode: boolean;
  /** Modal wall-clock timeout for the full analyze POST (must cover long Pro v3 runs). */
  modalTimeoutMs: number;
  renderTimeoutMs: number;
  logPrefix: string;
  /** Abort stops the client fetch; pair with ``requestProv3AnalyzeCancel`` so the worker stops too. */
  abortSignal?: AbortSignal;
  /** Shown when ``abortSignal`` fires (user clicked Stop). */
  userCancelledMessage?: string;
};

export type Prov3AnalyzeResult = {
  response: Response;
  route: "modal" | "render";
};

/** Lets the browser paint (e.g. progress 96%) before synchronous JSON parse on the main thread. */
export function yieldUiBeforeHeavyParse(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}

/**
 * Reads `NEXT_PUBLIC_PROV3_RENDER_FALLBACK`. Pro v3 multipart analyze no longer performs a Render
 * fallback POST (single Modal POST only); this remains for any other callers that check the flag.
 */
export function prov3RenderFallbackEnabled(): boolean {
  return process.env.NEXT_PUBLIC_PROV3_RENDER_FALLBACK === "true";
}

/** 将 FastAPI `detail` 与常见 404（错误 Modal 基址、未部署路由）说明合并为一条用户可读文案。 */
export function formatProAnalyzeHttpError(status: number, detail: string): string {
  const d = (detail || "").trim() || `HTTP ${status}`;
  if (status === 404) {
    return `Pro分析失败 [404]: ${d}。请确认 MODAL_BACKEND_URL 为 Modal 根地址（不要带 /pro-v3），且已部署含 POST /pro-v3/analyze 的镜像。`;
  }
  if (status === 422 && (d.includes("取消") || /cancel/i.test(d))) {
    return d;
  }
  return `Pro分析失败 [${status}]: ${d}`;
}

function makeProv3FormData(blob: Blob, filename: string, screenMode: boolean): FormData {
  const fd = makeFormData(blob, filename);
  fd.append("screen_mode", screenMode ? "true" : "false");
  return fd;
}

/** One in-flight Pro multipart analyze per tab (avoids duplicate POST → Modal 409). */
let _prov3AnalyzeClientBusy = false;

/**
 * Single POST to the first valid Modal base URL — no same-host HTTP retries, no connection retries,
 * no failover to other Modal hosts, no Render fallback (keyframe / Pro v3 debug auditing).
 */
async function tryProv3ModalHosts(
  blob: Blob,
  filename: string,
  screenMode: boolean,
  headers: Record<string, string>,
  modalUrls: string[],
  modalTimeoutMs: number,
  logPrefix: string,
  abortSignal: AbortSignal | undefined,
  userCancelledMessage: string | undefined,
): Promise<Prov3AnalyzeResult | null> {
  let mUrlRaw: string | null = null;
  let mUrl: string | null = null;
  for (const raw of modalUrls) {
    const n = normalizeProHttpApiBase(raw);
    if (n) {
      mUrlRaw = raw;
      mUrl = n;
      break;
    }
  }
  if (!mUrl || !mUrlRaw) {
    return null;
  }

  const analyzeUrl = buildProV3AnalyzeRequestUrl(mUrl);
  console.log(
    `${logPrefix} Pro v3 analyze: exactly one POST /pro-v3/analyze (no client retries) → ${analyzeUrl}`,
  );

  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), modalTimeoutMs);
  const onUserAbort = () => {
    clearTimeout(t);
    ctrl.abort();
  };
  if (abortSignal) {
    if (abortSignal.aborted) {
      clearTimeout(t);
      throw new Error(userCancelledMessage || "分析已停止");
    }
    abortSignal.addEventListener("abort", onUserAbort);
  }

  try {
    _prov3ActiveAnalyzeBase = mUrlRaw;
    const mRes = await fetch(analyzeUrl, {
      method: "POST",
      headers,
      body: makeProv3FormData(blob, filename, screenMode),
      signal: ctrl.signal,
    });
    clearTimeout(t);
    if (!mRes.ok) {
      console.warn(
        `${logPrefix} Pro v3 analyze: single POST completed with HTTP ${mRes.status} (no automatic retry)`,
      );
    }
    return { response: mRes, route: "modal" };
  } catch (e) {
    clearTimeout(t);
    const isAbort = e instanceof DOMException && e.name === "AbortError";
    const msg = e instanceof Error ? e.message : String(e);
    if (isAbort) {
      if (abortSignal?.aborted) {
        throw new Error(userCancelledMessage || "分析已停止");
      }
      throw new Error(
        "分析等待超时：请勿重复提交。可稍后从历史查看是否已完成，或缩短视频后重试。（客户端已禁用自动重试，便于对照单次请求。）",
      );
    }
    console.error(`${logPrefix} Pro v3 analyze: single POST failed (network/throw): ${msg}`);
    throw new Error(
      `Pro 分析请求失败（已禁用自动重试）：${msg}。请检查网络与 Modal 地址，查看上方日志中的 POST URL。`,
    );
  } finally {
    if (abortSignal) {
      abortSignal.removeEventListener("abort", onUserAbort);
    }
  }
}

/**
 * Pro v3 routing (debug-audit mode):
 * **One user action → one POST** to the first valid Modal URL. No Render fallback.
 * `cnNetworkHint` still adds `X-Stellar-Network-Hint: cn` for that single request.
 */
export async function runProv3AnalyzeMultipart(
  blob: Blob,
  filename: string,
  authHeaders: Record<string, string>,
  opts: RunProv3AnalyzeOptions,
): Promise<Prov3AnalyzeResult> {
  if (_prov3AnalyzeClientBusy) {
    throw new Error(
      "已有一次 Pro 分析正在进行，请等待结束后再试（重复请求会导致服务器返回 409）。",
    );
  }
  _prov3AnalyzeClientBusy = true;
  try {
  const modalUrls = opts.modalUrls.map((u) => u.replace(/\/+$/, "")).filter(Boolean);
  const headers: Record<string, string> = {
    ...authHeaders,
    ...(opts.cnNetworkHint ? { "X-Stellar-Network-Hint": "cn" } : {}),
  };
  const abortSignal = opts.abortSignal;
  const userCancelledMessage = opts.userCancelledMessage;

  const fromModal = await tryProv3ModalHosts(
    blob,
    filename,
    opts.screenMode,
    headers,
    modalUrls,
    opts.modalTimeoutMs,
    opts.logPrefix,
    abortSignal,
    userCancelledMessage,
  );
  if (fromModal) return fromModal;

  if (modalUrls.length === 0) {
    throw new Error("Pro 分析失败：未配置 Modal 地址（检查 precheck / MODAL_BACKEND_URL）");
  }
  throw new Error(
    "Pro 分析失败：没有可用的 Modal 基址（请检查 MODAL_BACKEND_URL 是否为有效 URL，不要带 /pro-v3）。",
  );
  } finally {
    _prov3AnalyzeClientBusy = false;
    clearProv3ActiveAnalyzeBase();
  }
}
