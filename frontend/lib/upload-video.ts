/**
 * Client-side helper: stream upload via same-origin `/api/upload-video`, poll until ready,
 * return opaque `upload_token` for `/api/lab` or `/api/analyze`.
 */

/** Broad extensions — browsers often omit MIME; iOS/Android may report empty type. */
const VIDEO_FILENAME_EXT_PATTERN =
  /\.(mp4|m4v|mov|qt|webm|mkv|avi|wmv|flv|3gp|3g2|ts|mts|m2ts|mpg|mpeg|vob|ogv|f4v|asf|divx|xvid|rm|rmvb|mxf|nut)$/i;

/**
 * Append ``file`` + ``request_id`` for ``POST /api/lite/analyze-proxy``.
 * Ensures Blobs get a ``File`` with a stable MIME (e.g. ``.mov`` → ``video/mp4``) so multipart parts are valid during Modal cold start.
 */
export function appendLiteAnalyzeFileToFormData(
  fd: FormData,
  file: Blob,
  filename: string,
  requestId: string,
): void {
  const name = (filename || "video.mp4").trim() || "video.mp4";
  const video = isVideoFile(file as File, name);
  const mime = normalizeVideoMimeForUpload((file as File).type || "", name);
  if (video && typeof File !== "undefined" && file instanceof File) {
    fd.append("file", file, name);
  } else if (video) {
    fd.append("file", new File([file], name, { type: mime }), name);
  } else {
    fd.append("file", file, name);
  }
  fd.append("request_id", requestId);
}

export function isVideoFile(file: File | Blob, filename: string): boolean {
  const t = ((file as File).type || "").trim().toLowerCase();
  if (t.startsWith("video/")) return true;
  if (t === "application/octet-stream" && VIDEO_FILENAME_EXT_PATTERN.test(filename)) return true;
  return VIDEO_FILENAME_EXT_PATTERN.test(filename);
}

/**
 * MIME for Gemini / R2: map quicktime → mp4; infer from filename when type missing.
 */
export function normalizeVideoMimeForUpload(rawType: string, filename: string): string {
  const t = rawType?.trim() || "";
  const base = t.split(";")[0]?.trim().toLowerCase() || "";
  if (base.startsWith("video/")) {
    if (base === "video/quicktime") return "video/mp4";
    return base || "video/mp4";
  }
  const low = filename.toLowerCase();
  if (low.endsWith(".webm")) return "video/webm";
  if (low.endsWith(".mkv")) return "video/x-matroska";
  if (low.endsWith(".mov") || low.endsWith(".qt")) return "video/mp4";
  if (low.endsWith(".avi")) return "video/x-msvideo";
  if (VIDEO_FILENAME_EXT_PATTERN.test(filename)) return "video/mp4";
  return "video/mp4";
}

/** R2 / history metadata: pick a reasonable video/* from filename. */
export function contentTypeForStoredVideo(filename: string): string {
  return normalizeVideoMimeForUpload("", filename);
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
  const mimeType = normalizeVideoMimeForUpload((file as File).type || "", filename || "video.mp4");

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
