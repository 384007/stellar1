/**
 * Cloudflare R2 storage helpers.
 * When deployed to Cloudflare Pages, the R2 binding is available through the platform context.
 * For local development and direct API usage, uses the S3-compatible API.
 */

export interface R2Bucket {
  put(
    key: string,
    value: ReadableStream | ArrayBuffer | string,
    options?: R2PutOptions
  ): Promise<R2Object>;
  get(key: string): Promise<R2ObjectBody | null>;
  delete(key: string): Promise<void>;
  list(options?: R2ListOptions): Promise<R2ObjectList>;
}

export interface R2PutOptions {
  httpMetadata?: {
    contentType?: string;
    contentDisposition?: string;
  };
  customMetadata?: Record<string, string>;
}

export interface R2Object {
  key: string;
  size: number;
  etag: string;
  uploaded: Date;
}

export interface R2ObjectBody extends R2Object {
  body: ReadableStream;
  arrayBuffer(): Promise<ArrayBuffer>;
  text(): Promise<string>;
  json<T>(): Promise<T>;
}

export interface R2ListOptions {
  prefix?: string;
  limit?: number;
  cursor?: string;
}

export interface R2ObjectList {
  objects: R2Object[];
  truncated: boolean;
  cursor?: string;
}

export function generateVideoKey(
  userId: string,
  fileName: string
): string {
  const timestamp = Date.now();
  const ext = fileName.split(".").pop() || "mp4";
  return `videos/${userId}/${timestamp}.${ext}`;
}

export function generateKeyframeKey(
  analysisId: string,
  frameIndex: number
): string {
  return `keyframes/${analysisId}/frame_${frameIndex}.jpg`;
}

export function getPublicUrl(key: string): string {
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID || "";
  const bucket = process.env.CLOUDFLARE_R2_BUCKET || "stellar-golf-media";
  return `https://${bucket}.${accountId}.r2.cloudflarestorage.com/${key}`;
}

export async function uploadToR2(
  bucket: R2Bucket,
  key: string,
  data: ArrayBuffer | ReadableStream,
  contentType: string
): Promise<R2Object> {
  return bucket.put(key, data, {
    httpMetadata: { contentType },
  });
}

export async function uploadVideoToR2(
  bucket: R2Bucket,
  userId: string,
  fileName: string,
  data: ArrayBuffer
): Promise<{ key: string; url: string }> {
  const key = generateVideoKey(userId, fileName);
  const contentType = fileName.endsWith(".mov") ? "video/quicktime" : "video/mp4";

  await uploadToR2(bucket, key, data, contentType);

  return {
    key,
    url: getPublicUrl(key),
  };
}

export async function getVideoFromR2(
  bucket: R2Bucket,
  key: string
): Promise<ArrayBuffer | null> {
  const obj = await bucket.get(key);
  if (!obj) return null;
  return obj.arrayBuffer();
}

export async function deleteFromR2(
  bucket: R2Bucket,
  key: string
): Promise<void> {
  await bucket.delete(key);
}

/**
 * Client-side upload using presigned URL or direct fetch.
 * For production, generate presigned URLs via a server endpoint.
 */
export async function uploadVideoClient(
  file: File,
  uploadUrl: string
): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(uploadUrl, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Upload failed");
  }

  const data = await response.json();
  return data.url;
}
