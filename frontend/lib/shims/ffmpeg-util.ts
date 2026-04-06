/**
 * `toBlobURL` compatible with `@ffmpeg/util` (fetch → Blob URL).
 * Resolved via `next.config.js` alias so `import("@ffmpeg/util")` does not pull the
 * published package into Edge/server bundles (its module graph references `document`).
 * The `@ffmpeg/util` package remains installed for version alignment with `@ffmpeg/ffmpeg`.
 */
export async function toBlobURL(
  url: string,
  mimeType: string,
  _progress?: boolean,
  _cb?: unknown,
): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`toBlobURL: ${res.status}`);
  const buf = await res.arrayBuffer();
  return URL.createObjectURL(new Blob([buf], { type: mimeType }));
}
