import { makeFormData } from "@/lib/fetch-retry";

export type RunProV2AnalyzeOptions = {
  modalUrls: string[];
  backendUrls: string[];
  cnNetworkHint: boolean;
  screenMode: boolean;
  /** Modal wall-clock timeout for the full analyze POST (must cover long Pro v3 runs). */
  modalTimeoutMs: number;
  renderTimeoutMs: number;
  logPrefix: string;
};

export type ProV2AnalyzeResult = {
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
export function proV2RenderFallbackEnabled(): boolean {
  return process.env.NEXT_PUBLIC_PRO_V2_RENDER_FALLBACK === "true";
}

function makeProV2FormData(blob: Blob, filename: string, screenMode: boolean): FormData {
  const fd = makeFormData(blob, filename);
  fd.append("screen_mode", screenMode ? "true" : "false");
  return fd;
}

function shouldRetryModalHttp(status: number): boolean {
  return status === 422 || status === 429 || status >= 500;
}

/**
 * Hit Modal only; retry transient HTTP statuses on the same host before trying the next Modal URL.
 * Returns the last Modal response when exhausted so the UI shows Modal's error instead of silently using Render.
 */
async function tryProV2ModalHosts(
  blob: Blob,
  filename: string,
  screenMode: boolean,
  headers: Record<string, string>,
  modalUrls: string[],
  modalTimeoutMs: number,
  logPrefix: string,
): Promise<ProV2AnalyzeResult | null> {
  let lastResponse: Response | null = null;

  for (const mUrl of modalUrls) {
    hrLoop: for (let hr = 0; hr < 5; hr++) {
      if (hr > 0) {
        await new Promise((r) => setTimeout(r, 5_000));
        console.log(`${logPrefix} Modal HTTP retry round ${hr + 1}/5 → ${mUrl}`);
      }
      for (let connAttempt = 0; connAttempt < 2; connAttempt++) {
        if (connAttempt > 0) {
          console.log(`${logPrefix} Modal connection retry (${mUrl}), waiting 8s…`);
          await new Promise((r) => setTimeout(r, 8_000));
        }
        try {
          const ctrl = new AbortController();
          const t = setTimeout(() => ctrl.abort(), modalTimeoutMs);
          console.log(`${logPrefix} Pro Modal → ${mUrl}/pro-v3/analyze (round ${hr + 1}, conn ${connAttempt + 1})`);
          const mRes = await fetch(`${mUrl}/pro-v3/analyze`, {
            method: "POST",
            headers,
            body: makeProV2FormData(blob, filename, screenMode),
            signal: ctrl.signal,
          });
          clearTimeout(t);

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
          // Abort = wall-clock timeout: server may already be analyzing — do NOT re-POST (was causing duplicate full analyses).
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

async function tryProV2BackendHosts(
  blob: Blob,
  filename: string,
  screenMode: boolean,
  headers: Record<string, string>,
  backendUrls: string[],
  renderTimeoutMs: number,
  maxConn: number,
  logPrefix: string,
): Promise<ProV2AnalyzeResult | null> {
  if (backendUrls.length === 0) {
    return null;
  }
  for (const bUrl of backendUrls) {
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
        console.log(`${logPrefix} Pro Render → ${bUrl}/pro-v3/analyze`);
        bRes = await fetch(`${bUrl}/pro-v3/analyze`, {
          method: "POST",
          headers,
          body: makeProV2FormData(blob, filename, screenMode),
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
 * Pro v2 routing (same for CN and non-CN):
 *
 * 1. **Modal first** — every region uses the same host order. `cnNetworkHint` does **not** reorder Modal vs Render.
 * 2. **cnNetworkHint** — only adds `X-Stellar-Network-Hint: cn` and raises Render `maxConn` when Render is used.
 * 3. **Render** — attempted only after Modal returns no usable in-process result **and**
 *    `NEXT_PUBLIC_PRO_V2_RENDER_FALLBACK=true` (default off: Modal-only traffic, Modal retries 422/429/5xx on-host).
 *
 * Default remains Modal-only (no Render) unless that env flag is set.
 */
export async function runProV2AnalyzeMultipart(
  blob: Blob,
  filename: string,
  authHeaders: Record<string, string>,
  opts: RunProV2AnalyzeOptions,
): Promise<ProV2AnalyzeResult> {
  const modalUrls = opts.modalUrls.map((u) => u.replace(/\/+$/, "")).filter(Boolean);
  const backendUrls = opts.backendUrls.map((u) => u.replace(/\/+$/, "")).filter(Boolean);
  const maxConn = opts.cnNetworkHint ? 2 : 1;
  const headers: Record<string, string> = {
    ...authHeaders,
    ...(opts.cnNetworkHint ? { "X-Stellar-Network-Hint": "cn" } : {}),
  };

  // Pass 1 — Modal (GPU); exhaustive retries on same URL before next Modal host.
  const fromModal = await tryProV2ModalHosts(
    blob,
    filename,
    opts.screenMode,
    headers,
    modalUrls,
    opts.modalTimeoutMs,
    opts.logPrefix,
  );
  if (fromModal) return fromModal;

  // Pass 2 — Render/API only when explicitly enabled (still never CN-first).
  if (proV2RenderFallbackEnabled() && backendUrls.length > 0) {
    console.log(`${opts.logPrefix} Modal gave no usable response; Render fallback enabled via env`);
    const fromRender = await tryProV2BackendHosts(
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
      `${opts.logPrefix} Modal-only mode (set NEXT_PUBLIC_PRO_V2_RENDER_FALLBACK=true to use Render after Modal fails)`,
    );
  }

  if (modalUrls.length === 0) {
    throw new Error("Pro 分析失败：未配置 Modal 地址（检查 precheck / MODAL_BACKEND_URL）");
  }
  throw new Error(
    "Pro 分析失败：Modal 不可用。请稍后重试；若需临时走 Render，设置 NEXT_PUBLIC_PRO_V2_RENDER_FALLBACK=true 并重新部署。",
  );
}
