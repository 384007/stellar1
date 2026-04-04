/**
 * Backend / caches sometimes store raw base64; other paths may store a full data URL.
 * Strip an optional data:image/...;base64, prefix so callers can safely prefix jpeg.
 */
export function rawBase64ImagePayload(s: string): string {
  return s.trim().replace(/^data:image\/[\w+.+-]+;base64,/i, "");
}

/**
 * Detect image MIME from the first bytes of a **raw** base64 payload (no `data:` prefix).
 * Falls back to `image/jpeg` when sniffing fails (matches OpenCV JPEG keyframes).
 */
export function sniffImageMimeFromBase64Payload(rawB64: string): string {
  const s = rawB64.replace(/\s/g, "");
  if (s.length < 12) return "image/jpeg";
  try {
    const padLen = (4 - (s.length % 4)) % 4;
    const sample = s.slice(0, Math.min(s.length, 96));
    const padded = sample + (padLen === 0 ? "" : padLen === 2 ? "==" : "=");
    const binary = atob(padded);
    const b0 = binary.charCodeAt(0);
    const b1 = binary.charCodeAt(1);
    const b2 = binary.charCodeAt(2);
    const b3 = binary.charCodeAt(3);
    if (b0 === 0xff && b1 === 0xd8 && b2 === 0xff) return "image/jpeg";
    if (b0 === 0x89 && b1 === 0x50 && b2 === 0x4e && b3 === 0x47) return "image/png";
    if (b0 === 0x47 && b1 === 0x49 && b2 === 0x46) return "image/gif";
    if (b0 === 0x52 && b1 === 0x49 && b2 === 0x46 && b3 === 0x46 && binary.length >= 12) {
      const tag = binary.slice(8, 12);
      if (tag === "WEBP") return "image/webp";
    }
    return "image/jpeg";
  } catch {
    return "image/jpeg";
  }
}

/**
 * Build a correct `data:` URL for keyframe / thumbnail fields.
 * Preserves an existing `data:image/…;base64,` MIME; otherwise sniffs from payload.
 * Returns `null` if the payload is missing or too short to be a real bitmap.
 */
export function keyframeImageDataUrl(imageField: string | undefined | null): string | null {
  if (typeof imageField !== "string") return null;
  const t = imageField.trim();
  if (!t) return null;
  const prefixed = t.match(/^data:(image\/[\w+.+-]+);base64,([\s\S]+)$/i);
  if (prefixed) {
    const mime = prefixed[1].toLowerCase();
    const body = prefixed[2].replace(/\s/g, "");
    if (body.length < 40) return null;
    return `data:${mime};base64,${body}`;
  }
  const raw = rawBase64ImagePayload(t).replace(/\s/g, "");
  if (raw.length < 40) return null;
  const mime = sniffImageMimeFromBase64Payload(raw);
  return `data:${mime};base64,${raw}`;
}
