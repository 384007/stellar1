/** Same-origin Pro v3 orchestration (upload → one Modal ``/analyze/start`` → R2 job poll). */
const PRO_V3_EDGE_ANALYZE_START = "/api/prov3/analyze/start";
const PRO_V3_EDGE_ANALYZE_CANCEL = "/api/prov3/analyze/cancel";

export function clearProv3ActiveAnalyzeBase(): void {
  /* legacy no-op: cancel is always same-origin */
}

/**
 * Ask Modal (via same-origin proxy) to cooperatively stop the current Pro analyze worker.
 */
export async function requestProv3AnalyzeCancel(
  authHeaders: Record<string, string>,
): Promise<{ ok: boolean }> {
  try {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 12_000);
    const r = await fetch(PRO_V3_EDGE_ANALYZE_CANCEL, {
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
  /** Wall-clock budget for the whole job (upload + start + polling until result). */
  modalTimeoutMs: number;
  renderTimeoutMs: number;
  logPrefix: string;
  abortSignal?: AbortSignal;
  userCancelledMessage?: string;
  /** UI hint: uploading → starting → polling → reconnecting (GET poll only; never a second analyze POST). */
  onJobPhase?: (phase: "uploading" | "starting" | "polling" | "reconnecting") => void;
};

export type Prov3AnalyzeResult = {
  /** Full product JSON (same shape as legacy sync ``POST /pro-v3/analyze``). */
  raw: Record<string, unknown>;
  route: "edge-job";
};

/** Lets the browser paint (e.g. progress 96%) before synchronous JSON parse on the main thread. */
export function yieldUiBeforeHeavyParse(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}

/**
 * Reads `NEXT_PUBLIC_PROV3_RENDER_FALLBACK`. The edge-job path does not use Render fallback;
 * kept for callers that branch on this flag elsewhere.
 */
export function prov3RenderFallbackEnabled(): boolean {
  return process.env.NEXT_PUBLIC_PROV3_RENDER_FALLBACK === "true";
}

/** 将 FastAPI `detail` 与常见 404 说明合并为一条用户可读文案。 */
export function formatProAnalyzeHttpError(status: number, detail: string): string {
  const d = (detail || "").trim() || `HTTP ${status}`;
  if (status === 404) {
    return `Pro分析失败 [404]: ${d}。请确认已部署含 Pro v3 异步分析的 Modal 镜像，且 R2 已配置。`;
  }
  if (status === 422 && (d.includes("取消") || /cancel/i.test(d))) {
    return d;
  }
  return `Pro分析失败 [${status}]: ${d}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** One in-flight Pro analyze orchestration per tab (avoids duplicate Modal job → 409). */
let _prov3AnalyzeClientBusy = false;

/**
 * Pro v3 product path: same-origin upload to R2 → **one** short ``POST /pro-v3/analyze/start`` on Modal
 * (via Edge proxy) → poll same-origin ``GET /api/prov3/analyze/job/:id`` (reads R2).
 * No long browser→Modal analyze connection; no automatic duplicate analyze POST retries (GET poll may retry on network blur).
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
    const abortSignal = opts.abortSignal;
    const userCancelledMessage = opts.userCancelledMessage;
    const deadline = Date.now() + opts.modalTimeoutMs;
    const pollIntervalMs = 2500;

    const bumpAbort = () => {
      if (abortSignal?.aborted) {
        throw new Error(userCancelledMessage || "分析已停止");
      }
    };

    opts.onJobPhase?.("uploading");
    bumpAbort();
    const uploadId = crypto.randomUUID();
    const upForm = new FormData();
    upForm.append("analysis_id", uploadId);
    upForm.append("file", blob, filename);
    const upRes = await fetch("/api/history/upload-video", {
      method: "POST",
      headers: { ...authHeaders },
      body: upForm,
      signal: abortSignal,
    });
    if (!upRes.ok) {
      let detail = `HTTP ${upRes.status}`;
      try {
        const j = (await upRes.json()) as { detail?: string };
        if (typeof j.detail === "string") detail = j.detail;
      } catch {
        /* ignore */
      }
      throw new Error(
        upRes.status === 401 || upRes.status === 403
          ? detail
          : `视频上传失败：${detail}`,
      );
    }
    const upJson = (await upRes.json()) as { video_r2_key?: string };
    const videoR2Key = String(upJson.video_r2_key || "").trim();
    if (!videoR2Key) {
      throw new Error("视频上传成功但未返回存储键，请重试。");
    }

    opts.onJobPhase?.("starting");
    bumpAbort();
    console.info(`${opts.logPrefix} [PROV3_EDGE_JOB] one analyze/start after R2 upload`, {
      uploadId,
      video_r2_key: videoR2Key,
    });

    const startHeaders: Record<string, string> = {
      ...authHeaders,
      "Content-Type": "application/json",
      ...(opts.cnNetworkHint ? { "X-Stellar-Network-Hint": "cn" } : {}),
    };
    const startRes = await fetch(PRO_V3_EDGE_ANALYZE_START, {
      method: "POST",
      headers: startHeaders,
      body: JSON.stringify({
        source_r2_key: videoR2Key,
        screen_mode: opts.screenMode,
      }),
      signal: abortSignal,
    });
    if (!startRes.ok) {
      let detail = `HTTP ${startRes.status}`;
      try {
        const j = (await startRes.json()) as { detail?: unknown };
        if (typeof j.detail === "string") detail = j.detail;
        else if (Array.isArray(j.detail)) detail = JSON.stringify(j.detail);
      } catch {
        /* ignore */
      }
      throw new Error(formatProAnalyzeHttpError(startRes.status, detail));
    }
    const startJson = (await startRes.json()) as { job_id?: string };
    const jobId = String(startJson.job_id || "").trim();
    if (!jobId) {
      throw new Error("分析任务未返回 job_id，请重试或联系支持。");
    }

    while (Date.now() < deadline) {
      bumpAbort();
      opts.onJobPhase?.("polling");
      let pollRes: Response;
      try {
        pollRes = await fetch(`/api/prov3/analyze/job/${encodeURIComponent(jobId)}`, {
          headers: { ...authHeaders },
          signal: abortSignal,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.warn(`${opts.logPrefix} [PROV3_EDGE_JOB] poll network error (will retry GET only):`, msg);
        opts.onJobPhase?.("reconnecting");
        await sleep(3000);
        continue;
      }

      if (pollRes.status === 401 || pollRes.status === 403) {
        let detail = `HTTP ${pollRes.status}`;
        try {
          const j = (await pollRes.json()) as { detail?: string };
          if (typeof j.detail === "string") detail = j.detail;
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }

      if (!pollRes.ok) {
        await sleep(pollIntervalMs);
        continue;
      }

      const j = (await pollRes.json()) as {
        status?: string;
        result?: Record<string, unknown> | null;
        detail?: string;
        job_id?: string;
      };
      const st = String(j.status || "");
      if (st === "completed") {
        if (j.result && typeof j.result === "object") {
          return { raw: j.result, route: "edge-job" };
        }
        throw new Error(
          "分析已完成但结果暂不可用。请稍后打开「历史记录」查看是否已同步；若仍没有，请重新分析。",
        );
      }
      if (st === "failed") {
        throw new Error(
          typeof j.detail === "string" && j.detail.trim()
            ? j.detail
            : "Pro 分析失败",
        );
      }

      await sleep(pollIntervalMs);
    }

    throw new Error(
      "等待分析结果超时。服务端可能仍在处理，请稍后到「历史记录」查看；请勿重复点击上传以免占用队列。",
    );
  } finally {
    _prov3AnalyzeClientBusy = false;
  }
}
