/**
 * Backend / caches sometimes store raw base64; other paths may store a full data URL.
 * Strip an optional data:image/...;base64, prefix so callers can safely prefix jpeg.
 */
export function rawBase64ImagePayload(s: string): string {
  return s.trim().replace(/^data:image\/\w+;base64,/i, "");
}
