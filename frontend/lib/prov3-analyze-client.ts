import { makeFormData } from "@/lib/fetch-retry";
import { buildProV3AnalyzeRequestUrl, normalizeProHttpApiBase } from "@/lib/prov3-endpoints";

export type RunProv3AnalyzeOptions = {
  modalUrls: string[];
  backendUrls: string[];
  cnNetworkHint: boolean;
  screenMode: boolean;
  /** Modal wall-clock timeout for the full analyze POST (must cover long Pro v3 runs). */
  modalTimeoutMs: number;
  renderTimeoutMs: number;
  logPrefix: string;
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

/** Opt-in only: default is Modal-only so traffic and errors stay on Modal (Render skipped). */
export function prov3RenderFallbackEnabled(): boolean {
  return process.env.NEXT_PUBLIC_PROV3_RENDER_FALLBACK === "true";
}

/** 将 FastAPI `detail` 与常见 404（错误 Modal 基址、未部署路由）说明合并为一条用户可读文案。 */
export function formatProAnalyzeHttpError(status: number, detail: string): string {
  const d = (detail || "").trim() || `HTTP ${status}`;
  if (status === 404) {
    return `Pro分析失败 [404]: ${d}。请确认 MODAL_BACKEND_URL 为 Modal 根地址（不要带 /pro-v3），且已部署含 POST /pro-v3/analyze 的镜像。`;
  }
  return `Pro分析失败 [${status}]: ${d}`;
}

function makeProv3FormData(blob: Blob, filename: string, screenMode: boolean): FormData {
  const fd = makeFormData(blob, filename);
  fd.append("screen_mode", screenMode ? "true" : "false");
  return fd;
}

/** Pro analyze POST is long-running; do not retry 422 (often validation — second POST can 409 single-flight). */
function shouldRetryModalHttp(status: number): boolean {
  return status === 429 || status === 502 || status === 503 || status === 504;
}

/** One in-flight Pro multipart analyze per tab (avoids duplicate POST → Modal 409). */
let _prov3AnalyzeClientBusy = false;

/**
 * Hit Modal only; retry transient HTTP statuses on the same host before trying the next Modal URL.
 */
async function tryProv3ModalHosts(
  blob: Blob,
  filename: string,
  screenMode: boolean,
  headers: Record<string, string>,
  modalUrls: string[],
  modalTimeoutMs: number,
  logPrefix: string,
): Promise<Prov3AnalyzeResult | null> {
  let lastResponse: Response | null = null;

  for (const mUrlRaw of modalUrls) {
    const mUrl = normalizeProHttpApiBase(mUrlRaw);
    if (!mUrl) continue;
    hrLoop: for (let hr = 0; hr < 5; hr++) {
      if (hr > 0) {
        await new Promise((r) => setTimeout(r, 5_000));
        console.log(`${logPrefix} Modal HTTP retry round ${hr + 1}/5 → ${mUrl}`);
      }
      let modalConnCtrl: AbortController | null = null;
      for (let connAttempt = 0; connAttempt < 2; connAttempt++) {
        if (connAttempt > 0) {
          modalConnCtrl?.abort();
          modalConnCtrl = null;
          console.log(`${logPrefix} Modal connection retry (${mUrl}), waiting 8s…`);
          await new Promise((r) => setTimeout(r, 8_000));
        }
        const ctrl = new AbortController();
        modalConnCtrl = ctrl;
        try {
          const t = setTimeout(() => ctrl.abort(), modalTimeoutMs);
          const analyzeUrl = buildProV3AnalyzeRequestUrl(mUrl);
          console.log(`${logPrefix} Pro Modal → ${analyzeUrl} (round ${hr + 1}, conn ${connAttempt + 1})`);
          const mRes = await fetch(analyzeUrl, {
            method: "POST",
            headers,
            body: makeProv3FormData(blob, filename, screenMode),
            signal: ctrl.signal,
          });
          clearTimeout(t);
          modalConnCtrl = null;

          lastResponse = mRes;
          if (mRes.ok) {
            return { response: mRes, route: "modal" };
          }
          if (shouldRetryModalHttp(mRes.status) && hr < 4) {
            console.warn(
              `${logPrefix} Modal ${mRes.status} on ${mUrl} — retrying same host (${hr + 1}/5)`,
            );
            continue hrLoop;
          }
          return { response: mRes, route: "modal" };
        } catch (e) {
          const isAbort = e instanceof DOMException && e.name === "AbortError";
          const isConnect = !isAbort && e instanceof TypeError;
          console.warn(
            `${logPrefix} Modal ${isAbort ? "timed out" : "unreachable"} (${mUrl}): ${e instanceof Error ? e.message : e}`,
          );
          if (isAbort) {
            throw new Error(
              "分析等待超时：请勿重复提交。可稍后从历史查看是否已完成，或缩短视频后重试。",
            );
          }
          if (isConnect && connAttempt === 0) continue;
          if (!isConnect) break;
        }
      }
    }
    if (lastResponse && !lastResponse.ok) {
      console.warn(`${logPrefix} Modal host exhausted: ${mUrl} (last HTTP ${lastResponse.status})`);
    }
  }

  return lastResponse ? { response: lastResponse, route: "modal" } : null;
}

async function tryProv3BackendHosts(
  blob: Blob,
  filename: string,
  screenMode: boolean,
  headers: Record<string, string>,
  backendUrls: string[],
  renderTimeoutMs: number,
  maxConn: number,
  logPrefix: string,
): Promise<Prov3AnalyzeResult | null> {
  if (backendUrls.length === 0) {
    return null;
  }
  for (const bUrlRaw of backendUrls) {
    const bUrl = normalizeProHttpApiBase(bUrlRaw);
    if (!bUrl) continue;
    console.log(`${logPrefix} Render warm-up → ${bUrl}`);
    for (let w = 0; w < 12; w++) {
      try {
        const hc = await fetch(`${bUrl}/health`, { signal: AbortSignal.timeout(10_000) });
        if (hc.ok) break;
      } catch {
        /* waking */
      }
      if (w < 11) await new Promise((r) => setTimeout(r, 5_000));
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), renderTimeoutMs);
    let connectAttempts = 0;
    let bRes: Response | null = null;
    while (!bRes) {
      try {
        const analyzeUrl = buildProV3AnalyzeRequestUrl(bUrl);
        console.log(`${logPrefix} Pro Render → ${analyzeUrl}`);
        bRes = await fetch(analyzeUrl, {
          method: "POST",
          headers,
          body: makeProv3FormData(blob, filename, screenMode),
          signal: controller.signal,
        });
      } catch (e) {
        const isAbort = e instanceof DOMException && e.name === "AbortError";
        if (!isAbort && e instanceof TypeError && connectAttempts < maxConn) {
          connectAttempts++;
          await new Promise((r) => setTimeout(r, 10_000));
          continue;
        }
        clearTimeout(timer);
        if (isAbort) {
          throw new Error("Pro分析超时（6分钟），请压缩视频后重试");
        }
        throw new Error(`网络错误：${e instanceof Error ? e.message : "无法连接服务器"}`);
      }
      break;
    }
    clearTimeout(timer);
    if (!bRes) continue;
    if (bRes.ok) return { response: bRes, route: "render" };
    if (bRes.status === 422 || bRes.status >= 500) {
      console.warn(`${logPrefix} Render ${bRes.status} on ${bUrl}, try next backend if any`);
      continue;
    }
    return { response: bRes, route: "render" };
  }
  return null;
}

/**
 * Pro v3 routing:
 * 1. **Modal first** — same host order everywhere; `cnNetworkHint` does not reorder Modal vs Render.
 * 2. **cnNetworkHint** — adds `X-Stellar-Network-Hint: cn` and raises Render `maxConn` when Render is used.
 * 3. **Render** — only after Modal fails **and** `NEXT_PUBLIC_PROV3_RENDER_FALLBACK=true`.
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
  const backendUrls = opts.backendUrls.map((u) => u.replace(/\/+$/, "")).filter(Boolean);
  const maxConn = opts.cnNetworkHint ? 2 : 1;
  const headers: Record<string, string> = {
    ...authHeaders,
    ...(opts.cnNetworkHint ? { "X-Stellar-Network-Hint": "cn" } : {}),
  };

  const fromModal = await tryProv3ModalHosts(
    blob,
    filename,
    opts.screenMode,
    headers,
    modalUrls,
    opts.modalTimeoutMs,
    opts.logPrefix,
  );
  if (fromModal) return fromModal;

  if (prov3RenderFallbackEnabled() && backendUrls.length > 0) {
    console.log(`${opts.logPrefix} Modal gave no usable response; Render fallback enabled via env`);
    const fromRender = await tryProv3BackendHosts(
      blob,
      filename,
      opts.screenMode,
      headers,
      backendUrls,
      opts.renderTimeoutMs,
      maxConn,
      opts.logPrefix,
    );
    if (fromRender) return fromRender;
  } else if (backendUrls.length > 0) {
    console.log(
      `${opts.logPrefix} Modal-only mode (set NEXT_PUBLIC_PROV3_RENDER_FALLBACK=true to use Render after Modal fails)`,
    );
  }

  if (modalUrls.length === 0) {
    throw new Error("Pro 分析失败：未配置 Modal 地址（检查 precheck / MODAL_BACKEND_URL）");
  }
  throw new Error(
    "Pro 分析失败：Modal 不可用。请稍后重试；若需临时走 Render，设置 NEXT_PUBLIC_PROV3_RENDER_FALLBACK=true 并重新部署。",
  );
  } finally {
    _prov3AnalyzeClientBusy = false;
  }
}
