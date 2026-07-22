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
  const norm = buffer.replace(/\r\n/g, "\n");
  const blocks = norm.split(/\n\n+/);
  for (const block of blocks) {
    for (const line of block.split("\n")) {
      if (line.startsWith("data: ")) {
        last = line.slice(6).trim();
      }
    }
  }
  return last;
}

function bodyLooksLikeSse(raw: string): boolean {
  const t = raw.trimStart();
  return t.startsWith(":") || t.startsWith("data:") || t.startsWith("event:");
}

function parseLiteEnvelopeFromSseText(raw: string): Record<string, unknown> {
  const last = parseLastSseDataBlock(raw);
  if (!last) {
    throw new Error("Lite analyze stream ended without a data payload");
  }
  let env: LiteSseEnvelope;
  try {
    env = JSON.parse(last) as LiteSseEnvelope;
  } catch {
    throw new Error("Lite analyze stream returned invalid JSON");
  }
  if (!env.ok) {
    const msg = env.detail || env.code || "Lite analyze failed";
    const err = new Error(msg);
    (err as Error & { liteCode?: string; liteStatus?: number }).liteCode = env.code;
    (err as Error & { liteCode?: string; liteStatus?: number }).liteStatus = env.status;
    throw err;
  }
  return env.result;
}

/**
 * Read Lite analyze response body (SSE final envelope, or legacy JSON).
 * Sniffs SSE when ``Content-Type`` is wrong but the body starts with ``:`` / ``data:`` (some proxies).
 */
export async function readLiteAnalyzeResult(res: Response): Promise<Record<string, unknown>> {
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  const raw = await res.text();
  const sseLike =
    ct.includes("text/event-stream") || bodyLooksLikeSse(raw);
  if (sseLike) {
    return parseLiteEnvelopeFromSseText(raw);
  }
  if (!raw.trim()) {
    throw new Error("Empty response body");
  }
  return JSON.parse(raw) as Record<string, unknown>;
}
