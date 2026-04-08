/**
 * ``POST /analyze/lite`` returns either cached ``application/json`` or ``text/event-stream`` (SSE keepalives
 * + final ``data:`` JSON) so intermediaries do not close long idle connections before the body starts.
 */

export type LiteSseEnvelope =
  | { ok: true; result: Record<string, unknown> }
  | {
      ok: false;
      status?: number;
      detail?: string;
      code?: string;
      request_id?: string;
    };

function parseLastSseDataBlock(buffer: string): string | null {
  let last: string | null = null;
  const blocks = buffer.split(/\r?\n\r?\n/);
  for (const block of blocks) {
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("data: ")) {
        last = line.slice(6).trim();
      }
    }
  }
  return last;
}

async function readSseLiteEnvelope(res: Response): Promise<LiteSseEnvelope> {
  if (!res.body) {
    throw new Error("Empty response body");
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    buffer += dec.decode(value || new Uint8Array(), { stream: !done });
    if (done) break;
  }
  const raw = parseLastSseDataBlock(buffer);
  if (!raw) {
    throw new Error("Lite analyze stream ended without a data payload");
  }
  try {
    return JSON.parse(raw) as LiteSseEnvelope;
  } catch {
    throw new Error("Lite analyze stream returned invalid JSON");
  }
}

/**
 * Read Lite analyze response body (JSON cache hit or SSE final envelope).
 * Caller should only use when ``res.ok`` or when handling errors that still use JSON.
 */
export async function readLiteAnalyzeResult(res: Response): Promise<Record<string, unknown>> {
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("text/event-stream")) {
    const env = await readSseLiteEnvelope(res);
    if (!env.ok) {
      const msg = env.detail || env.code || "Lite analyze failed";
      const err = new Error(msg);
      (err as Error & { liteCode?: string; liteStatus?: number }).liteCode = env.code;
      (err as Error & { liteCode?: string; liteStatus?: number }).liteStatus = env.status;
      throw err;
    }
    return env.result;
  }
  return (await res.json()) as Record<string, unknown>;
}
