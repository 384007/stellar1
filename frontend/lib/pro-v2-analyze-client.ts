import { makeFormData } from "@/lib/fetch-retry";

export type RunProV2AnalyzeOptions = {
  modalUrls: string[];
  backendUrls: string[];
  cnNetworkHint: boolean;
  screenMode: boolean;
  /** Modal attempt timeout (analyze page uses shorter cold-start budget). */
  modalTimeoutMs: number;
  renderTimeoutMs: number;
  logPrefix: string;
};

export type ProV2AnalyzeResult = {
  response: Response;
  route: "modal" | "render";
};

function makeProV2FormData(blob: Blob, filename: string, screenMode: boolean): FormData {
  const fd = makeFormData(blob, filename);
  fd.append("screen_mode", screenMode ? "true" : "false");
  return fd;
}

async function tryProV2ModalHosts(
  blob: Blob,
  filename: string,
  screenMode: boolean,
  headers: Record<string, string>,
  modalUrls: string[],
  modalTimeoutMs: number,
  logPrefix: string,
): Promise<ProV2AnalyzeResult | null> {
  for (const mUrl of modalUrls) {
    for (let mAttempt = 0; mAttempt < 2; mAttempt++) {
      try {
        if (mAttempt > 0) {
          console.log(`${logPrefix} Modal retry (${mUrl}) after connection failure, waiting 8s…`);
          await new Promise((r) => setTimeout(r, 8_000));
        }
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), modalTimeoutMs);
        console.log(`${logPrefix} Pro Modal → ${mUrl}/pro-v2/analyze (attempt ${mAttempt + 1})`);
        const mRes = await fetch(`${mUrl}/pro-v2/analyze`, {
          method: "POST",
          headers,
          body: makeProV2FormData(blob, filename, screenMode),
          signal: ctrl.signal,
        });
        clearTimeout(t);
        if (mRes.ok) {
          return { response: mRes, route: "modal" };
        }
        if (mRes.status === 422 || mRes.status >= 500) {
          console.warn(`${logPrefix} Modal ${mRes.status} on ${mUrl}, try next host or fallback tier`);
        } else {
          return { response: mRes, route: "modal" };
        }
        break;
      } catch (e) {
        const isAbort = e instanceof DOMException && e.name === "AbortError";
        const isConnect = !isAbort && e instanceof TypeError;
        console.warn(
          `${logPrefix} Modal ${isAbort ? "timed out" : "unreachable"} (${mUrl}): ${e instanceof Error ? e.message : e}`,
        );
        if (!isConnect) break;
      }
    }
  }
  return null;
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
        console.log(`${logPrefix} Pro Render → ${bUrl}/pro-v2/analyze`);
        bRes = await fetch(`${bUrl}/pro-v2/analyze`, {
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
 * Pro v2: Modal hosts (GPU) then Render/API fallbacks by default.
 * **China (cnNetworkHint):** Render/API first — modal.run is often blocked or flaky from mainland;
 * Modal is still tried after Render if all backends fail or return retryable errors.
 * Sends `X-Stellar-Network-Hint: cn` so workers can align Gemini / copy even without CF-IPCountry.
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

  const tryModal = () =>
    tryProV2ModalHosts(
      blob,
      filename,
      opts.screenMode,
      headers,
      modalUrls,
      opts.modalTimeoutMs,
      opts.logPrefix,
    );
  const tryBackends = () =>
    tryProV2BackendHosts(
      blob,
      filename,
      opts.screenMode,
      headers,
      backendUrls,
      opts.renderTimeoutMs,
      maxConn,
      opts.logPrefix,
    );

  if (opts.cnNetworkHint) {
    const fromRender = await tryBackends();
    if (fromRender) return fromRender;
    const fromModal = await tryModal();
    if (fromModal) return fromModal;
  } else {
    const fromModal = await tryModal();
    if (fromModal) return fromModal;
    const fromRender = await tryBackends();
    if (fromRender) return fromRender;
  }

  if (backendUrls.length === 0 && modalUrls.length === 0) {
    throw new Error("Pro 分析失败：未配置可用后端地址");
  }
  throw new Error("Pro 分析失败：所有可用后端均无可用响应，请稍后重试");
}
