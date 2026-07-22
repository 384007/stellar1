/**
 * Lite analyze Edge proxy: while waiting for Modal cold start / long connect,
 * emit SSE comment lines so the browser ↔ Pages leg does not look "idle" to CF/proxies.
 * Upstream body is then forwarded (SSE pipe) or wrapped as a single final `data:` JSON envelope.
 */

import { sanitizeLiteSseStream } from "@/lib/chains";
import { sanitizeProductJson } from "@/lib/chains/sanitize";

const EDGE_SSE_PING_MS = 3500;

function sseDataLine(obj: unknown): Uint8Array {
  const enc = new TextEncoder();
  return enc.encode(`data: ${JSON.stringify(obj)}\n\n`);
}

/**
 * Build a ReadableStream of ``text/event-stream`` for the browser.
 * Always emits ``: edge-open`` first, then ``: edge-wait-upstream`` on an interval until ``fetch`` settles.
 */
export function createLiteAnalyzeEdgeSseStream(
  upstreamFetch: () => Promise<Response>,
): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  let pingTimer: ReturnType<typeof setInterval> | null = null;

  return new ReadableStream({
    async start(controller) {
      const clearPing = () => {
        if (pingTimer) {
          clearInterval(pingTimer);
          pingTimer = null;
        }
      };
      const safeClose = () => {
        try {
          controller.close();
        } catch {
          /* already closed */
        }
      };

      try {
        controller.enqueue(enc.encode(": edge-open\n\n"));
        pingTimer = setInterval(() => {
          try {
            controller.enqueue(enc.encode(": edge-wait-upstream\n\n"));
          } catch {
            clearPing();
          }
        }, EDGE_SSE_PING_MS);

        const upstream = await upstreamFetch();
        clearPing();

        const ct = (upstream.headers.get("content-type") || "").toLowerCase();

        if (upstream.ok && upstream.body && ct.includes("text/event-stream")) {
          const reader = sanitizeLiteSseStream(upstream.body, "analysis").getReader();
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            if (value) controller.enqueue(value);
          }
          safeClose();
          return;
        }

        const buf = new Uint8Array(await upstream.arrayBuffer());

        if (upstream.ok && ct.includes("application/json")) {
          try {
            const data = JSON.parse(new TextDecoder().decode(buf)) as unknown;
            const result = sanitizeProductJson(data, "analysis");
            controller.enqueue(sseDataLine({ ok: true as const, result }));
            safeClose();
            return;
          } catch {
            /* fall through */
          }
        }

        let detail = `Upstream HTTP ${upstream.status}`;
        let code = "LITE_UPSTREAM_ERROR";
        try {
          const j = JSON.parse(new TextDecoder().decode(buf)) as {
            detail?: unknown;
            code?: string;
            request_id?: string;
          };
          if (j.code) code = String(j.code);
          if (j.detail !== undefined) {
            detail =
              typeof j.detail === "string"
                ? j.detail
                : JSON.stringify(j.detail).slice(0, 4000);
          }
        } catch {
          const text = new TextDecoder().decode(buf).trim();
          if (text) detail = text.slice(0, 4000);
        }

        controller.enqueue(
          sseDataLine({
            ok: false as const,
            status: upstream.status,
            detail,
            code,
          }),
        );
        safeClose();
      } catch {
        clearPing();
        try {
          controller.enqueue(
            sseDataLine({
              ok: false as const,
              status: 502,
              detail: "分析服务连接失败，请稍后重试。",
              code: "LITE_PROXY_UPSTREAM_FAILED",
            }),
          );
        } catch {
          /* stream broken */
        }
        safeClose();
      }
    },
  });
}
