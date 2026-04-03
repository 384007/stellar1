/**
 * Client-side helper: upload video to Gemini via the streaming proxy,
 * poll until processed, return file_uri for use in analysis endpoints.
 *
 * Each step is a lightweight Edge Worker call (<30 s each), so the
 * combined pipeline supports files up to 100 MB+ without hitting
 * Cloudflare's wall-clock timeout.
 */

export function isVideoFile(file: File | Blob, filename: string): boolean {
  const t = (file as File).type || "";
  return t.startsWith("video/") || /\.(mp4|mov|webm|avi)$/i.test(filename);
}

export interface UploadResult {
  file_uri: string;
  mime_type: string;
  /** 0 = GEMINI_API_KEY, 1 = GEMINI_API_KEY_2 — must match /api/analyze when using file_uri. */
  gemini_key_index?: number;
}

export async function uploadVideoToGemini(
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

  // Phase 1: Stream upload via proxy (Worker does NOT buffer the file)
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
  const file_uri = raw.file_uri as string;
  const file_name = raw.file_name as string | null;
  const mime_type = raw.mime_type as string | undefined;
  const gemini_key_index =
    typeof raw.gemini_key_index === "number" ? raw.gemini_key_index : 0;

  if (!file_uri) throw new Error("上传成功但未获取文件标识");

  onProgress?.(55);

  // Phase 2: Poll until Gemini finishes processing the video
  if (file_name) {
    const pollQs = `name=${encodeURIComponent(file_name)}&key_index=${gemini_key_index}`;
    for (let i = 0; i < 90; i++) {
      const sr = await fetch(`/api/upload-video?${pollQs}`, {
        headers: authHeaders,
        signal,
      });
      if (sr.ok) {
        const info = await sr.json();
        if (info.state === "ACTIVE") break;
        if (info.state === "FAILED")
          throw new Error("视频处理失败，请重试");
      }
      onProgress?.(55 + Math.min(30, i * 0.4));
      await new Promise((r) => setTimeout(r, 2000));
    }
  }

  onProgress?.(88);
  return {
    file_uri,
    mime_type: mime_type || mimeType,
    gemini_key_index,
  };
}
