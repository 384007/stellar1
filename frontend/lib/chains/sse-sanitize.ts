import { sanitizeProductJson, type ProductChain } from "./sanitize";

/**
 * Sanitize JSON in each `data: {...}` line of an SSE stream (Lite analyze, etc.).
 * Comment lines (`:`) and non-JSON payloads pass through unchanged.
 */
export function sanitizeLiteSseStream(
  source: ReadableStream<Uint8Array>,
  chain: ProductChain = "analysis",
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";

  return source.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        buffer += decoder.decode(chunk, { stream: true });
        const parts = buffer.split("\n");
        buffer = parts.pop() ?? "";
        for (const line of parts) {
          controller.enqueue(encoder.encode(transformSseLine(line, chain) + "\n"));
        }
      },
      flush(controller) {
        if (buffer.length) {
          controller.enqueue(encoder.encode(transformSseLine(buffer, chain)));
        }
      },
    }),
  );
}

function transformSseLine(line: string, chain: ProductChain): string {
  const trimmed = line.replace(/\r$/, "");
  if (!trimmed.startsWith("data:")) {
    return line;
  }
  const after = trimmed.slice(5).trim();
  if (!after || after.startsWith(":")) {
    return line;
  }
  try {
    const j = JSON.parse(after) as unknown;
    return `data: ${JSON.stringify(sanitizeProductJson(j, chain))}`;
  } catch {
    return line;
  }
}
