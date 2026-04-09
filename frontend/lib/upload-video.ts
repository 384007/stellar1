/**
 * Client-side helper: stream upload via same-origin `/api/upload-video`, poll until ready,
 * return opaque `upload_token` for `/api/lab` or `/api/analyze`.
 */

export function isVideoFile(file: File | Blob, filename: string): boolean {
  const t = (file as File).type || "";
  return t.startsWith("video/") || /\.(mp4|mov|webm|avi)$/i.test(filename);
}

export interface UploadResult {
  /** Opaque server-held file reference (browser never sees raw URI). */
  upload_token: string;
  mime_type: string;
}

export async function uploadVideoForAnalysis(
  file: File | Blob,
  filename: string,
  authHeaders: Record<string, string>,
  onProgress?: (pct: number) => void,
  signal?: AbortSignal,
): Promise<UploadResult> {
  const rawType = (file as File).type || "";
  const mimeType =
    rawType === "video/quicktime"
      ? "video/mp4"
      : rawType || "video/mp4";

  onProgress?.(8);

  const uploadRes = await fetch("/api/upload-video", {
    method: "POST",
    headers: {
      ...authHeaders,
      "Content-Type": mimeType,
      "X-Upload-Content-Type": mimeType,
      "X-Upload-Filename": filename || "video.mp4",
      "X-Upload-Byte-Length": String(file.size),
    },
    body: file,
    signal,
  });

  if (!uploadRes.ok) {
    const d = await uploadRes
      .json()
      .catch(() => ({ detail: `HTTP ${uploadRes.status}` }));
    throw new Error(
      (d as Record<string, string>).detail ||
        `视频上传失败 [${uploadRes.status}]`,
    );
  }

  const raw = await uploadRes.json() as Record<string, unknown>;
  const upload_token = raw.upload_token as string | undefined;
  const mime_type = raw.mime_type as string | undefined;

  if (!upload_token) throw new Error("上传成功但未获取会话令牌");

  onProgress?.(55);

  const pollQs = `session=${encodeURIComponent(upload_token)}`;
  for (let i = 0; i < 90; i++) {
    const sr = await fetch(`/api/upload-video?${pollQs}`, {
      headers: authHeaders,
      signal,
    });
    if (sr.ok) {
      const info = await sr.json() as { state?: string };
      if (info.state === "ACTIVE") break;
      if (info.state === "FAILED")
        throw new Error("视频处理失败，请重试");
    }
    onProgress?.(55 + Math.min(30, i * 0.4));
    await new Promise((r) => setTimeout(r, 2000));
  }

  onProgress?.(88);
  return {
    upload_token,
    mime_type: mime_type || mimeType,
  };
}
