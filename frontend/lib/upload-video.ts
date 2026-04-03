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

  const { file_uri, file_name, mime_type } = await uploadRes.json();
  if (!file_uri) throw new Error("上传成功但未获取文件标识");

  onProgress?.(55);

  // Phase 2: Poll until Gemini finishes processing the video
  if (file_name) {
    for (let i = 0; i < 90; i++) {
      const sr = await fetch(
        `/api/upload-video?name=${encodeURIComponent(file_name)}`,
        { headers: authHeaders, signal },
      );
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
  return { file_uri, mime_type: mime_type || mimeType };
}
